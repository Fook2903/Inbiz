# 🚀 Dự Án Inbiz - Team Inbiz

Đây là repository chính thức của **Team Inbiz** chứa toàn bộ mã nguồn, dữ liệu, notebooks phân tích và kết quả dự đoán (submissions). Mọi nội dung trong này đều bám sát cấu trúc của báo cáo (định dạng NeurIPS) đính kèm.

---

## 🌟 1. Tính Năng Nổi Bật (Key Features)

* **⚡ Pipeline Tự Động:** Quy trình xử lý dữ liệu và huấn luyện mô hình được đóng gói khép kín.
* **📊 Trực Quan Hóa Chuyên Sâu:** Bao gồm phân tích đặc trưng (Feature Importance) và giải thích mô hình bằng SHAP.
* **🛠️ Tính Tái Lập (Reproducibility):** Cố định Random Seed để đảm bảo kết quả đồng nhất trên mọi máy tính.
* **📂 Cấu Trúc Sạch:** Tổ chức thư mục khoa học, dễ dàng theo dõi và mở rộng.

---

## 📁 2. Cấu Trúc Thư Mục (Project Structure)

Vui lòng giữ nguyên sơ đồ thư mục này để các scripts không bị lỗi đường dẫn:
```text
Inbiz-main/
├── README.md                           # Tài liệu hướng dẫn này
├── data/                               # Thư mục chứa toàn bộ dữ liệu đã được làm sạch
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
└── Part 3/                             # Pipeline & Modeling (Trọng tâm báo cáo)
    ├── advanced_model.py               # Script mô hình tối ưu nhất
    ├── pipeline_model.py               # Script chạy pipeline tổng thể
    ├── data/                           # Dữ liệu bổ trợ huấn luyện
    │   ├── sales.csv                   # Doanh số
    │   ├── web_traffic.csv             # Lưu lượng truy cập
    │   └── sample_submission.csv       # Mẫu file nộp bài
    ├── notebooks/                      # Quá trình thử nghiệm (EDA)
    │   ├── baseline.ipynb              # Notebook xây dựng Baseline
    │   └── baseline_model.py           # Script Python cho Baseline
    ├── plots/                          # Hình ảnh trực quan (Sử dụng trong báo cáo)
    │   ├── feature_importance.png      # Độ quan trọng của đặc trưng
    │   └── shap_summary.png            # Giải thích mô hình SHAP
    └── submissions/                    # Kết quả đầu ra (CSV)
        ├── baseline.csv                # Kết quả từ mô hình cơ sở
        ├── pipeline_model.csv          # Kết quả từ mô hình pipeline
        └── advanced_model.csv          # Kết quả nộp bài cuối cùng

