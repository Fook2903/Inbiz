# 📦 Inbiz Repository

Đây là repository chính thức của **Team Inbiz** chứa toàn bộ mã nguồn, dữ liệu đã làm sạch, notebooks phân tích, dashboard Power BI và kết quả dự đoán (submissions).
Mọi nội dung trong này đều bám sát cấu trúc của báo cáo (định dạng NeurIPS) đính kèm.

---

## 🌟 1. Tính Năng Nổi Bật (Key Features)

* ⚡ **Pipeline Tự Động:** Quy trình xử lý dữ liệu và huấn luyện mô hình được đóng gói khép kín.
* 📊 **Trực Quan Hóa Chuyên Sâu:** Bao gồm Feature Importance, SHAP và Dashboard Power BI.
* 🛠️ **Tính Tái Lập:** Cố định Random Seed đảm bảo kết quả đồng nhất.
* 📂 **Cấu Trúc Sạch:** Dễ theo dõi, mở rộng và bảo trì.

---

## 📁 2. Cấu Trúc Thư Mục (Project Structure)

> ⚠️ Giữ nguyên cấu trúc để tránh lỗi đường dẫn

```text
Inbiz-main/
├── README.md
├── reports/
│   └── Inbiz_Datathon_NeurIPS_2025.pdf
├── data/
│   ├── customers_clean.csv
│   ├── data
│   ├── geography_clean.csv
│   ├── inventory_clean.csv
│   ├── order_items_clean&order_clean.zip
│   ├── payments_clean.csv
│   ├── products_clean.csv
│   ├── promotions_clean.csv
│   ├── returns_clean.csv
│   ├── reviews_clean.csv
│   ├── sales_clean.csv
│   ├── shipments_clean.csv
│   └── web_traffic_clean.csv
└── Part 3/
    ├── advanced_model.py
    ├── pipeline_model.py
    ├── data/
    │   ├── sales.csv
    │   ├── web_traffic.csv
    │   └── sample_submission.csv
    ├── notebooks/
    │   ├── baseline.ipynb
    │   └── baseline_model.py
    ├── plots/
    │   ├── feature_importance.png
    │   └── shap_summary.png
    └── submissions/
        ├── baseline.csv
        ├── pipeline_model.csv
        └── advanced_model.csv
```

---

## 💾 3. Dữ Liệu Đầu Vào (Data)

Toàn bộ dữ liệu trong `data/` đã được làm sạch.

### 🔑 Dữ liệu cốt lõi:

* `sales.csv`: Doanh thu theo thời gian (target)
* `web_traffic.csv`: Hành vi người dùng
* `sample_submission.csv`: Format output

---

## 📊 4. Phân Tích & Trực Quan Hóa (Power BI)

### 4.1 Data Relationships

| From        | Column        | Relation | To        | Column      | Status |
| ----------- | ------------- | -------- | --------- | ----------- | ------ |
| inventory   | product_id    | *:1      | products  | product_id  | Active |
| inventory   | snapshot_date | *:1      | Dim_Date  | Date        | Active |
| order_items | order_id      | *:1      | orders    | order_id    | Active |
| order_items | product_id    | *:1      | products  | product_id  | Active |
| order_items | promo_id      | *:1      | promotion | promo_id    | Active |
| orders      | customer_id   | *:1      | customers | customer_id | Active |
| orders      | order_date    | *:1      | Dim_Date  | Date        | Active |
| orders      | zip           | *:1      | geography | zip         | Active |
| payments    | order_id      | 1:1      | orders    | order_id    | Active |
| returns     | order_id      | *:1      | orders    | order_id    | Active |
| returns     | product_id    | *:1      | products  | product_id  | Active |
| reviews     | order_id      | *:1      | orders    | order_id    | Active |
| reviews     | product_id    | *:1      | products  | product_id  | Active |
| sales       | Date          | *:1      | Dim_Date  | Date        | Active |
| shipments   | order_id      | 1:1      | orders    | order_id    | Active |
| web_traffic | date          | *:1      | Dim_Date  | Date        | Active |

---

### 4.2 DAX Measures

```DAX
-- 1. Tạo bảng Dim_Date (Theo giai đoạn sales_train)
Dim_Date = 
VAR BaseCalendar = CALENDAR(DATE(2012, 7, 4), DATE(2022, 12, 31)) 
RETURN
    ADDCOLUMNS (
        BaseCalendar,
        "Year", YEAR([Date]),
        "Month No", MONTH([Date]),
        "Month Name", FORMAT([Date], "MMMM"),
        "Quarter", "Q" & FORMAT([Date], "Q"),
        "Month-Year", FORMAT([Date], "MMM yyyy"),
        "YearMonth_Sort", FORMAT([Date], "yyyyMM") -- Dùng để sắp xếp cột Month-Year
    )

-- 2. Các Measures Tính Toán Chỉ Số Kinh Doanh & Vận Hành
Avg Delivery Days = AVERAGE('shipments'[Wait Days])

Avg_Rating = AVERAGE(reviews[rating])

Conversion Rate = DIVIDE(DISTINCTCOUNT('orders'[order_id]), SUM('web_traffic'[sessions]), 0)

Gross Margin % = DIVIDE([Total Revenue] - [Total COGS Calculated], [Total Revenue], 0)

Late Delivery Rate = 
VAR LateOrders = COUNTROWS(FILTER('shipments', 'shipments'[Wait Days] > 6))
VAR TotalOrders = COUNTROWS('shipments')
RETURN DIVIDE(LateOrders, TotalOrders, 0)

Return Rate = DIVIDE(COUNTROWS('returns'), COUNTROWS('order_items'), 0)

SD_Delivery = STDEV.S(shipments[Wait Days])

Total COGS Calculated = 
SUMX(
    'order_items', 
    'order_items'[quantity] * RELATED('products'[cogs])
)

Total Revenue = 
SUMX(
    'order_items', 
    'order_items'[quantity] * 'order_items'[unit_price]
)

Total_Return_Qty = SUM(returns[return_quantity])

Wait Days = DATEDIFF(RELATED('orders'[order_date]), 'shipments'[delivery_date], DAY)

Lead_Time_Group = 
VAR LeadTime = DATEDIFF(RELATED(orders[order_date]), shipments[delivery_date], DAY)
RETURN
SWITCH(TRUE(), 
    LeadTime <= 2, "1-2 Ngày", 
    LeadTime <= 4, "3-4 Ngày", 
    LeadTime <= 7, "5-7 Ngày (Quá tải)", 
    "> 7 Ngày"
)
```

---

## 🤖 5. Mô Hình Dự Báo

### 🎯 Mục tiêu

Dự báo doanh thu từ dữ liệu lịch sử + hành vi user

### ⚙️ Pipeline

* Baseline Model
* Pipeline Model
* Advanced Model

---

## ⚙️ 6. How to Run

```bash
pip install -r requirements.txt
cd "Part 3"

python notebooks/baseline_model.py
python pipeline_model.py
python advanced_model.py
```

---

## 🎯 7. Output

### 📂 Submissions

* baseline.csv
* pipeline_model.csv
* advanced_model.csv ✅

### 📊 Plots

* feature_importance.png
* shap_summary.png

---

## 👥 Team

**Team Inbiz**
