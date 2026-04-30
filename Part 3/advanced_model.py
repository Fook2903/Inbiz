import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# 1. CẤU HÌNH & TÍNH TÁI LẬP
# ============================================
RANDOM_SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_SEED)

DATA_DIR = "data"
SUBMISSION_DIR = "submissions"
PLOT_DIR = "plots"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================
# 2. FEATURE ENGINEERING
# ============================================
def get_special_days(df):
    """Khai thác triệt để các ngày lễ và sự kiện mua sắm tại Việt Nam"""
    dates = pd.to_datetime(df['Date'])
    is_double_day = (dates.dt.day == dates.dt.month).astype(int)
    fixed_holidays = [(1, 1), (4, 30), (5, 1), (9, 2), (12, 24), (12, 25), (12, 31)]
    is_holiday = dates.apply(lambda x: 1 if (x.month, x.day) in fixed_holidays else 0)
    women_days = [(2, 14), (3, 8), (10, 20)]
    is_women_day = dates.apply(lambda x: 1 if (x.month, x.day) in women_days else 0)
    is_big_sale = dates.apply(lambda x: 1 if (x.month == 11 and x.day == 11) or (x.month == 12 and x.day == 12) else 0)
    is_payday = dates.apply(lambda x: 1 if x.is_month_end or x.day == 5 else 0)
    return is_double_day, is_holiday, is_women_day, is_big_sale, is_payday


def prepare_advanced_features(df_sales, df_traffic=None, is_train=True):
    df = df_sales.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if df_traffic is not None:
        df_traffic = df_traffic.copy()
        df_traffic['date'] = pd.to_datetime(df_traffic['date'])
        traffic_daily = df_traffic.groupby('date').agg({
            'sessions': 'sum',
            'page_views': 'sum'
        }).reset_index().rename(columns={'date': 'Date'})
        df = pd.merge(df, traffic_daily, on='Date', how='left')
        df['sessions'] = df['sessions'].ffill().fillna(df['sessions'].median())
        df['page_views'] = df['page_views'].ffill().fillna(df['page_views'].median())

    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    res = get_special_days(df)
    df['is_double_day'], df['is_holiday'], df['is_women_day'], df['is_big_sale'], df['is_payday'] = res

    for lag in [1, 7, 14, 30]:
        df[f'rev_lag_{lag}'] = df['Revenue'].shift(lag)
        if 'sessions' in df.columns:
            df[f'traffic_lag_{lag}'] = df['sessions'].shift(lag)

    df['rev_roll_mean_7'] = df['Revenue'].shift(1).rolling(window=7).mean()
    df['rev_roll_mean_30'] = df['Revenue'].shift(1).rolling(window=30).mean()
    df['rev_roll_std_7'] = df['Revenue'].shift(1).rolling(window=7).std()

    if is_train:
        df = df.dropna().reset_index(drop=True)
    else:
        df = df.ffill().fillna(0)
    return df


# ============================================
# 3. HUẤN LUYỆN & PHÂN TÍCH GIẢI THÍCH (SHAP/Importance)
# ============================================
def train_balanced_model(df):
    features = [c for c in df.columns if c not in ['Date', 'Revenue', 'COGS']]
    X = df[features]
    y = np.log1p(df['Revenue'])

    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []

    print(f"\n--- Bắt đầu huấn luyện XGBoost Balanced ---")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.015,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=100
        )

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = np.expm1(model.predict(X_val))
        actual = np.expm1(y_val)

        mae = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        r2 = r2_score(actual, preds)
        cv_results.append([mae, rmse, r2])
        print(f"Fold {fold + 1} | MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | R2: {r2:.4f}")

    avg = np.mean(cv_results, axis=0)
    print(
        "\n" + "=" * 45 + f"\nKẾT QUẢ CV TRUNG BÌNH:\nMAE: {avg[0]:,.2f} | RMSE: {avg[1]:,.2f} | R2: {avg[2]:.4f}\n" + "=" * 45)

    # Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu
    final_model = xgb.XGBRegressor(n_estimators=model.best_iteration, learning_rate=0.015, random_state=RANDOM_SEED)
    final_model.fit(X, y)

    # 1. Feature Importance (XGBoost Native)
    print("\n[Đang tạo] Biểu đồ Feature Importance...")
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(final_model, max_num_features=15, importance_type='gain', title='Top 15 Features (Gain)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"))
    plt.close()

    # 2. SHAP Values (Model Explainability)
    print("[Đang tạo] Biểu đồ SHAP Summary...")
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title("SHAP Feature Impact Analysis")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "shap_summary.png"))
        plt.close()
    except Exception as e:
        print(f"Lưu ý: Không thể tạo SHAP plot do {e}")

    return final_model, features


# ============================================
# 4. THỰC THI & KIỂM SOÁT SUBMISSION
# ============================================
if __name__ == "__main__":
    try:
        df_sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
        df_traffic = pd.read_csv(os.path.join(DATA_DIR, "web_traffic.csv"))
        df_sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    except Exception as e:
        print(f"[LỖI] {e}");
        exit()

    # 1. Feature Engineering & Training
    train_proc = prepare_advanced_features(df_sales, df_traffic, is_train=True)
    model, feat_cols = train_balanced_model(train_proc)

    # 2. Dự báo tập Test (Đảm bảo thứ tự tuyệt đối bằng Merge)
    df_test_template = df_sample[['Date']].copy()
    df_test_template['Date'] = pd.to_datetime(df_test_template['Date'])

    full_df = pd.concat([df_sales, df_sample], axis=0).reset_index(drop=True)
    processed_all = prepare_advanced_features(full_df, df_traffic, is_train=False)

    test_final = pd.merge(df_test_template, processed_all, on='Date', how='left')
    X_test = test_final[feat_cols]
    y_pred_revenue = np.expm1(model.predict(X_test))

    # 3. Tính toán và Ràng buộc COGS (Dùng 90 ngày gần nhất)
    recent_90 = df_sales.tail(90)
    ratio = min(recent_90['COGS'].sum() / recent_90['Revenue'].sum(), 0.9)
    y_pred_cogs = y_pred_revenue * ratio

    # 4. Lưu kết quả
    submission = pd.DataFrame({
        'Date': df_sample['Date'],
        'Revenue': np.round(y_pred_revenue, 2),
        'COGS': np.round(y_pred_cogs, 2)
    })

    assert len(submission) == len(df_sample)
    assert (submission['Revenue'] > submission['COGS']).all()

    path = os.path.join(SUBMISSION_DIR, "advanced_model.csv")
    submission.to_csv(path, index=False)
    print(f"\n[V] Đã lưu file kết quả: {path}")
    print(f"[V] Biểu đồ phân tích đã được lưu trong thư mục: {PLOT_DIR}")