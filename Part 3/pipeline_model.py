import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# ============================================
# 1. CẤU HÌNH & TÍNH TÁI LẬP
# ============================================
RANDOM_SEED = 42
DATA_DIR = "data"
SUBMISSION_DIR = "submissions"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(RANDOM_SEED)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# ============================================
# 2. XÂY DỰNG TIME-SERIES PIPELINE
# ============================================
def prepare_features(df, is_train=True):
    df = df.copy()

    # Lưu lại thứ tự ban đầu để đảm bảo không bị xáo trộn file nộp bài
    df['row_id'] = range(len(df))
    df['Date'] = pd.to_datetime(df['Date'])

    # Sắp xếp theo ngày để tính Lag và Rolling
    df = df.sort_values('Date')

    # a. Đặc trưng thời gian
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # b. Tạo Lag Features
    for lag in [1, 7, 14]:
        df[f'rev_lag_{lag}'] = df['Revenue'].shift(lag)

    # c. Rolling Statistics (Trung bình trượt 7 ngày và 30 ngày)
    df['rev_roll_mean_7'] = df['Revenue'].shift(1).rolling(window=7).mean()
    df['rev_roll_mean_30'] = df['Revenue'].shift(1).rolling(window=30).mean()

    if is_train:
        # Xóa dòng NaN do Lag tạo ra để không làm nhiễu mô hình
        df = df.dropna().reset_index(drop=True)
    else:
        # Trả về thứ tự dòng gốc trước khi kết thúc Pipeline cho tập Test
        df = df.sort_values('row_id')
        df = df.ffill().fillna(0)

    return df


# ============================================
# 3. HUẤN LUYỆN (MODEL PIPELINE)
# ============================================
def train_with_time_cv(df):
    # Loại bỏ row_id và các cột mục tiêu khỏi đặc trưng huấn luyện
    features = [col for col in df.columns if col not in ['Date', 'Revenue', 'COGS', 'row_id']]
    X = df[features]
    y = df['Revenue']

    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []

    print(f"\n--- Đang chạy Time-Series CV ---")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        model.fit(X_train_cv, y_train_cv)

        preds = model.predict(X_val_cv)
        mae = mean_absolute_error(y_val_cv, preds)
        mae_scores.append(mae)
        print(f"Fold {fold + 1} MAE: {mae:,.2f}")

    print(f"==> Average CV MAE: {np.mean(mae_scores):,.2f}")

    # Huấn luyện Final Model
    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    final_model.fit(X, y)
    return final_model, features


# ============================================
# 4. HÀM GIẢI THÍCH MÔ HÌNH
# ============================================
def export_interpretability(model, X_train):
    os.makedirs('plots', exist_ok=True)

    # 1. Feature Importance
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 5))
    plt.barh(importances['Feature'][:10][::-1], importances['Importance'][:10][::-1], color='skyblue')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

    # 2. SHAP Summary
    print("Đang tính toán SHAP values...")
    explainer = shap.TreeExplainer(model)
    X_sample = X_train.iloc[:200]
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig('plots/shap_summary.png', bbox_inches='tight')
    plt.close()
    print("[V] Đã lưu biểu đồ giải thích vào thư mục plots/")


# ============================================
# 5. THỰC THI
# ============================================
if __name__ == "__main__":
    train_path = os.path.join(DATA_DIR, "sales.csv")
    sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

    if os.path.exists(train_path) and os.path.exists(sample_path):
        df_sales = pd.read_csv(train_path)
        df_sample = pd.read_csv(sample_path)

        # 1. Huấn luyện Revenue
        processed_train = prepare_features(df_sales, is_train=True)
        model, feat_cols = train_with_time_cv(processed_train)

        # 2. Xuất biểu đồ SHAP & Importance
        export_interpretability(model, processed_train[feat_cols])

        # 3. KIỂM TRA ĐỘ LỆCH NGÀY
        last_train_date = pd.to_datetime(df_sales['Date']).max()
        first_test_date = pd.to_datetime(df_sample['Date']).min()
        gap = (first_test_date - last_train_date).days

        print(f"\n[CHECK] Ngày cuối Train: {last_train_date.date()} | Ngày đầu Test: {first_test_date.date()}")
        if gap > 1:
            print(f"[CẢNH BÁO] Khoảng trống {gap} ngày! Lag features có thể không chính xác.")
        else:
            print(f"[OK] Dữ liệu thời gian liên tiếp.")

        # 4. Chuẩn bị tập Test (Bảo vệ thứ tự dòng tuyệt đối)
        full_df = pd.concat([df_sales, df_sample], axis=0).reset_index(drop=True)
        processed_full = prepare_features(full_df, is_train=False)

        # Lấy đúng số lượng dòng cuối cùng tương ứng với sample_submission
        df_test = processed_full.tail(len(df_sample))

        # 5. Dự báo Revenue
        y_pred_revenue = model.predict(df_test[feat_cols])

        # 6. TÍNH RATIO COGS (CÓ XỬ LÝ NGOẠI LỆ CHIA CHO 0)
        last_30_days = df_sales.tail(30)
        sum_rev_30 = last_30_days['Revenue'].sum()

        if sum_rev_30 > 0:
            ratio_cogs = last_30_days['COGS'].sum() / sum_rev_30
        else:
            # Fallback: Nếu 30 ngày qua ko có doanh thu, dùng tỷ lệ tổng thể
            ratio_cogs = df_sales['COGS'].sum() / df_sales['Revenue'].sum()
            print("[LƯU Ý] Doanh thu 30 ngày qua bằng 0, sử dụng tỷ lệ COGS tổng quát.")

        y_pred_cogs = y_pred_revenue * ratio_cogs

        # 7. Lưu kết quả nộp bài
        submission = pd.DataFrame({
            'Date': df_sample['Date'],
            'Revenue': np.round(y_pred_revenue, 2),
            'COGS': np.round(y_pred_cogs, 2)
        })

        sub_path = os.path.join(SUBMISSION_DIR, "pipeline_model.csv")
        submission.to_csv(sub_path, index=False)

        print(f"\n[THÀNH CÔNG] File kết quả: {sub_path}")
        print(f"Tỷ lệ COGS áp dụng (30 ngày gần nhất): {ratio_cogs:.4f}")
    else:
        print("[LỖI] Kiểm tra lại đường dẫn file trong thư mục data/")
