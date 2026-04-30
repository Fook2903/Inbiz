import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ============================================
# 1. MÔI TRƯỜNG & RANDOM SEED
# ============================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
print(f"Random seed = {RANDOM_SEED}")

# ============================================
# 2. THƯ MỤC SUBMISSION & KIỂM TRA DỮ LIỆU
# ============================================
DATA_DIR = "data"
SALES_PATH = os.path.join(DATA_DIR, "sales.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUBMISSION_DIR = "submissions"

if not os.path.exists(SALES_PATH):
    raise FileNotFoundError(f"Không tìm thấy {SALES_PATH}")
if not os.path.exists(SAMPLE_PATH):
    raise FileNotFoundError(f"Không tìm thấy {SAMPLE_PATH}")

os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ============================================
# 3. ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# ============================================
train = pd.read_csv(SALES_PATH)
train['Date'] = pd.to_datetime(train['Date'])
train = train.sort_values('Date')
# Xóa dòng trùng ngày (nếu có) và xóa dòng thiếu Revenue/COGS
train = train.drop_duplicates(subset='Date').dropna(subset=['Revenue', 'COGS'])

sample = pd.read_csv(SAMPLE_PATH)
test_dates = pd.to_datetime(sample['Date'])

print(f"Train: {len(train)} ngày, từ {train['Date'].min()} đến {train['Date'].max()}")
print(f"Test: {len(test_dates)} ngày, từ {test_dates.min()} đến {test_dates.max()}")

# ============================================
# 4. TẠO ĐẶC TRƯNG THỜI GIAN (giữ nguyên 7 đặc trưng)
# ============================================
def create_date_features(dates):
    df = pd.DataFrame({'Date': pd.to_datetime(dates)})
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekend'] = (df['dayofweek'] >= 5).astype(int)
    return df.drop('Date', axis=1)

X_train = create_date_features(train['Date'])
y_train = train['Revenue']
X_test = create_date_features(test_dates)

# ============================================
# 5. HUẤN LUYỆN RANDOM FOREST
# ============================================
model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
model.fit(X_train, y_train)

# Dự báo doanh thu
y_pred_revenue = model.predict(X_test)

# Dự báo COGS theo tỷ lệ trung bình trên toàn bộ train
ratio_cogs = train['COGS'].sum() / train['Revenue'].sum()
y_pred_cogs = y_pred_revenue * ratio_cogs

# ============================================
# 6. LƯU SUBMISSION
# ============================================
submission = pd.DataFrame({
    'Date': sample['Date'],
    'Revenue': np.round(y_pred_revenue, 2),
    'COGS': np.round(y_pred_cogs, 2)
})
submission_path = os.path.join(SUBMISSION_DIR, "baseline.csv")
submission.to_csv(submission_path, index=False)

print(f"File kết quả: {submission_path}")
