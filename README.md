# SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection
**Đề tài:** Detecting the Spread of Misinformation in Social Network (Mở rộng Temporal Tree Transformer với User Credibility Features trên bộ dữ liệu PHEME)  
**Môn học:** Mạng Xã Hội | **GVHD:** TS. Trần Hưng Nghiệp  
**Nhóm:** Nhóm 6 (Trần Thiên Phú, Mạc Nguyễn Gia Huy, Võ Tấn Đạt, Nguyễn Thị Thanh Mai, Bùi Phạm Bích Phương)

### 📚 Tài liệu Quản lý Dự án & Phối hợp Nhóm:
- 📋 [Kế hoạch chi tiết toàn diện (Plan.md)](Plan.md)
- ✅ [Danh sách công việc & Backlog GitHub Issues (TODO.md)](TODO.md)
- 🤝 [Quy trình phối hợp làm việc nhóm & Git Guidelines (CONTRIBUTING.md)](CONTRIBUTING.md)
- 📝 [Nhật ký tiến độ & Checkpoint (PROGRESS_LOG.md)](PROGRESS_LOG.md)

---

## 1. Cấu trúc Dự án
```
PJ/
├── data/
│   ├── raw/                 # Nơi lưu archive gốc PHEME_veracity.tar.bz2
│   └── processed/           # Cache dữ liệu đã tiền xử lý: pheme_processed.pkl
├── src/
│   ├── data/
│   │   ├── pheme_parser.py           # Parser bóc tách threads, tweets, trees từ PHEME
│   │   ├── tree_builder.py           # Dựng quan hệ cây lan truyền & metrics cấu trúc
│   │   └── user_feature_builder.py   # Trích xuất 12 đặc trưng User Credibility (3 nhóm)
│   ├── models/
│   │   ├── user_encoder.py           # MLP & Gated Fusion cho User Credibility
│   │   ├── tree_transformer.py       # Tree Transformer Attention Layer
│   │   ├── temporal_gru.py           # Temporal GRU tổng hợp lát cắt thời gian
│   │   └── ucttt_classifier.py       # End-to-end model UC-TTT & TTT Baseline
│   ├── baselines/
│   │   ├── tfidf_lr.py               # Baseline 1: TF-IDF + Logistic Regression
│   │   └── propagation_rf.py         # Baseline 2: Propagation Features + Random Forest
│   └── utils/
│       ├── metrics.py                # Tính Accuracy, Macro-F1, Precision, Recall
│       └── loeo_splitter.py          # Bộ chia 9-fold Leave-One-Event-Out (LOEO)
├── experiments/
│   ├── preprocess_pheme.py           # Script tiền xử lý và đóng gói dataset
│   ├── run_baselines.py              # Script chạy benchmark Baseline 1 & 2
│   └── run_ucttt_experiment.py       # Script huấn luyện & đánh giá UC-TTT / TTT
├── results/                          # Chứa kết quả thực nghiệm dạng JSON và log
├── requirements.txt                  # Các thư viện phụ thuộc
└── README.md
```

---

## 2. Hướng dẫn Chạy Thực nghiệm

### 2.1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2.2. Tiền xử lý dữ liệu PHEME
Đọc từ archive `PHEME_veracity.tar.bz2`, trích xuất cây phân cấp, nhãn và 12 đặc trưng uy tín người dùng:
```bash
python experiments/preprocess_pheme.py
```
*Kết quả:* Tạo ra file `data/processed/pheme_processed.pkl` gồm 6,425 threads (2,402 Rumours, 4,023 Non-rumours) trải rộng trên 9 sự kiện.

### 2.3. Chạy Baselines truyền thống (9-fold LOEO)
```bash
python experiments/run_baselines.py
```
*Kết quả được lưu tại:* `results/baseline_results.json`

| Mô hình | Pooled Accuracy | Pooled Macro-F1 | Mean LOEO F1 (± std) |
| :--- | :---: | :---: | :---: |
| **B1a: TF-IDF + LR (Source Only)** | **71.77%** | **67.56%** | **56.17% ± 16.66%** |
| **B1b: TF-IDF + LR (Source + Reactions)** | 70.60% | 65.67% | 52.85% ± 16.25% |
| **B2: Propagation Features + RF** | 56.62% | 55.62% | 46.12% ± 8.88% |

---

## 3. Kiến trúc Mô hình Đề xuất: UC-TTT
Mô hình **User-Credibility-Aware Temporal Tree Transformer** giải quyết hạn chế của các phương pháp hiện tại bằng cách:
1. **Nội dung bài viết (Text Representation):** Vector hóa câu từ của tweet.
2. **Độ sâu cây lan truyền (Depth Encoding):** Sinusoidal Positional Encoding cho độ sâu trong cascade.
3. **Uy tín người dùng (User Credibility Encoder):** Mã hóa 12 đặc trưng (Profile Verification, Engagement Counts với Log-scale, Derived Ratios) và dung hợp thông qua cơ chế cổng động (**Gated Fusion**).
4. **Cấu trúc cây (Tree Transformer):** Multi-head self-attention mô hình hóa tương tác giữa các phản hồi.
5. **Động lực học thời gian (Temporal GRU):** Phân chia thành các lát cắt thời gian và tổng hợp tiến trình lan truyền theo thời gian.
