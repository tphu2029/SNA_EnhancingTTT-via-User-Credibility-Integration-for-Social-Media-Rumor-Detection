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
│   └── processed/           # Cache dữ liệu đã tiền xử lý: pheme_processed.pkl, text_embeddings.pt
├── src/
│   ├── data/
│   │   ├── pheme_parser.py           # Parser bóc tách threads, tweets, trees từ PHEME
│   │   ├── tree_builder.py           # Dựng quan hệ cây lan truyền & metrics cấu trúc
│   │   ├── user_feature_builder.py   # Trích xuất 12 đặc trưng User Credibility (3 nhóm)
│   │   └── dataset.py                # PyTorch Dataset & dynamic batch collate function
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
│   ├── extract_text_embeddings.py    # Script trích xuất offline sentence embeddings (MiniLM)
│   ├── run_baselines.py              # Script chạy benchmark Baseline 1 & 2
│   └── run_ucttt_experiment.py       # Script huấn luyện & đánh giá UC-TTT / TTT
├── tests/
│   └── test_dataloader.py            # Unit test cho dynamic batching & time window collate
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

### 2.3. Trích xuất Offline Text Embeddings
Chuyển đổi văn bản của 6,425 bài đăng gốc và hơn 105,000 phản hồi thành vector ngữ nghĩa tĩnh 384-d bằng `sentence-transformers/all-MiniLM-L6-v2`:
```bash
python experiments/extract_text_embeddings.py --batch_size 256
```
> **Mẹo (Tải nhanh từ GitHub Releases):** Nếu không muốn chạy trích xuất lại trên máy, bạn có thể tải trực tiếp file `text_embeddings.pt` đã sinh sẵn từ tag release [`v0.1-data-cache`](https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection/releases/tag/v0.1-data-cache) về thư mục `data/processed/`.

### 2.4. Kiểm thử DataLoader & Dynamic Tree Batching
```bash
pytest tests/test_dataloader.py -v
```

### 2.5. Chạy Baselines truyền thống (9-fold LOEO)
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
1. **Nội dung bài viết (Text Representation):** Vector hóa câu từ của tweet thông qua MiniLM Sentence Transformer.
2. **Độ sâu cây lan truyền (Depth Encoding):** Sinusoidal Positional Encoding cho độ sâu trong cascade.
3. **Uy tín người dùng (User Credibility Encoder):** Mã hóa 12 đặc trưng (Profile Verification, Engagement Counts với Log-scale, Derived Ratios) và dung hợp thông qua cơ chế cổng động (**Gated Fusion**).
4. **Cấu trúc cây (Tree Transformer):** Multi-head self-attention mô hình hóa tương tác giữa các phản hồi.
5. **Động lực học thời gian (Temporal GRU):** Phân chia thành các lát cắt thời gian và tổng hợp tiến trình lan truyền theo thời gian.

---

## 4. Bảng Theo dõi Tiến độ Dự án (Roadmap & Status)

| Giai đoạn | Nội dung công việc trọng tâm | Trạng thái | Người phụ trách |
| :---: | :--- | :---: | :--- |
| **GĐ 0** | Khảo sát, Tiền xử lý dữ liệu PHEME, Khối Deep Learning PyTorch & Baselines B1/B2 | ✅ **Hoàn thành** | Tất cả thành viên |
| **GĐ 1** | Offline Text Embeddings (`all-MiniLM-L6-v2`) & PyTorch Dynamic Tree DataLoader | ✅ **Hoàn thành** (11/10) | Gia Huy, Tấn Đạt |
| **GĐ 2** | Huấn luyện & Tái hiện Baseline 3: Original TTT (Wu et al. 2025) trên 9-fold LOEO | 🔄 **Đang triển khai** | Thiên Phú, Gia Huy |
| **GĐ 3** | Huấn luyện Mô hình Đề xuất UC-TTT & So sánh 3 Fusion Strategies (RQ1 & H1) | ⏳ Chờ thực hiện | Thiên Phú, Thanh Mai, Bích Phương |
| **GĐ 4** | Chuỗi thực nghiệm Ablation Study bóc tách Feature-level & Model-level (RQ2 & H2) | ⏳ Chờ thực hiện | Tấn Đạt, Bích Phương, Thanh Mai |
| **GĐ 5** | Phân tích Thống kê theo Sự kiện & Phân tích Lỗi sai / Case Studies (RQ3 & H3) | ⏳ Chờ thực hiện | Thanh Mai, Bích Phương |
| **GĐ 6** | Tổng hợp Báo cáo Cuối kỳ chuẩn IMRaD, Thiết kế Slide thuyết trình & Bàn giao Repo | ⏳ Chờ thực hiện | Tất cả thành viên |

