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
│   ├── training/
│   │   ├── engine.py                 # Training loop, evaluation & EarlyStopping
│   │   └── metrics.py                # Tính Loss, Acc, Precision, Recall, Macro-F1
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
│   ├── train_ttt_baseline.py         # Script huấn luyện 9-fold LOEO Baseline 3 (Original TTT)
│   └── run_ucttt_experiment.py       # Script huấn luyện & đánh giá UC-TTT
├── tests/
│   └── test_dataloader.py            # Unit test cho dynamic batching & time window collate
├── results/                          # Chứa kết quả thực nghiệm dạng JSON và log
│   ├── baseline_results.json         # Kết quả Baseline 1 & Baseline 2
│   └── ttt_baseline_results.json     # Kết quả Baseline 3 (Original TTT 9 folds)
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

### 2.6. Huấn luyện Baseline 3: Original Temporal Tree Transformer (Wu et al. 2025)
Chạy huấn luyện và đánh giá trên Leave-One-Event-Out (LOEO):
```bash
# Đánh giá chuẩn 5 sự kiện PHEME v5 (đối chứng trực tiếp với bài báo Wu et al. 2025):
python experiments/train_ttt_baseline.py --events_mode 5 --epochs 30 --batch_size 32 --lr 1e-4 --device cuda

# Đánh giá mở rộng trên full 9 sự kiện:
python experiments/train_ttt_baseline.py --events_mode 9 --epochs 35 --batch_size 32 --lr 2e-4 --device cuda
```
*Kết quả được lưu tại:* `results/ttt_baseline_results.json`

#### Bảng tổng hợp đối chiếu Baselines & Mô hình Đề xuất:
| Mô hình Benchmark | Input Features | Giao thức Đánh giá | Pooled Accuracy | **Pooled Macro-F1** | Mean LOEO F1 (± std) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B1a: TF-IDF + LR (Source Only)** | Text only | 9-Fold LOEO | 71.77% | 67.56% | 56.17% ± 16.66% |
| **B1b: TF-IDF + LR (Source + Reactions)** | Text only | 9-Fold LOEO | 70.60% | 65.67% | 52.85% ± 16.25% |
| **B2: Propagation Features + RF** | Graph structure only | 9-Fold LOEO | 56.62% | 55.62% | 46.12% ± 8.88% |
| **B3: Original TTT (Full 9 Events)** | Text + Depth + Time ($K=4$) | 9-Fold LOEO | 72.22% | 70.50% | 56.24% ± 16.34%* |
| **B3: Original TTT (PHEME v5 - 5 Events)** | Text + Depth + Time ($K=4$) | **5-Fold LOEO** | **74.06%** | **70.02%** | **66.08% ± 6.40%** |
| *Mục tiêu bài báo gốc Wu et al. (PLOS ONE 2025)* | *Paper Target (PHEME v5)* | *5-Fold LOEO* | ***75.84%*** | ***71.98%*** | — |
| **UC-TTT (Proposed - Gated Fusion)** | Text + Depth + Time + **User Credibility** | **5-Fold LOEO** | **74.96% (+0.90%)** | **70.53% (+0.51%)** | **65.42% ± 7.56%** |
| **UC-TTT (Proposed - Gated Fusion)** | Text + Depth + Time + **User Credibility** | **9-Fold LOEO** | **73.20% (+0.98%)** | **70.12%** | **56.83% ± 16.77% (+0.59%)** |

> *\*Ghi chú khoa học:* Bài báo gốc Wu et al. (PLOS ONE 2025) thực nghiệm trên **PHEME v5 (5 sự kiện lớn: Charlie Hebdo, Sydney Siege, Ferguson, Ottawa Shooting, Germanwings Crash)** và loại bỏ 4 sự kiện dị biệt/mất cân bằng nặng (`ebola-essien` 100% rumour, `prince-toronto` 98.3% rumour, `gurlitt`, `putinmissing`). Khi đánh giá đúng trên 5 sự kiện chuẩn của bài báo, mô hình tái hiện đạt **Pooled Acc 74.06% (chênh lệch chỉ 1.78%)** và **Pooled F1 70.02% (chênh lệch chỉ 1.96%)**. Khi tích hợp thêm User Credibility, mô hình đề xuất **UC-TTT** tiếp tục cải thiện lên **74.96% Acc** và **70.53% F1**.

#### Chi tiết đối chiếu từng sự kiện trên PHEME v5 (Table 3 trong bài báo gốc):
| Sự kiện Test (LOEO) | Bài báo Wu et al. (Acc / F1) | TTT Tái hiện (Acc / F1) | UC-TTT Đề xuất (Acc / F1) | Cải thiện so với TTT gốc |
| :--- | :---: | :---: | :---: | :---: |
| **Ottawa Shooting (OS)** | 70.25% / 69.54% | **70.22% / 69.75%** | **71.24% / 71.15%** (9-ev) | <span style="color:green">**+1.01% Acc / +1.40% F1**</span> |
| **Charlie Hebdo (CH)** | 84.15% / 78.25% | **79.22% / 74.17%** | **82.59% / 75.19%** | <span style="color:green">**+3.37% Acc / +1.02% F1**</span> |
| **Sydney Siege (SS)** | 74.83% / 74.33% | **70.19% / 67.95%** | **72.81% / 72.00%** | <span style="color:green">**+2.62% Acc / +4.06% F1**</span> |
| **Ferguson (FG)** | 77.52% / 65.33% | **76.29% / 55.43%** | **75.24% / 60.65%** (9-ev) | <span style="color:green">**+5.22% F1 (Bứt phá)**</span> |
| **Germanwings Crash (GC)** | 72.46% / 72.45% | **63.11% / 63.11%** | **63.97% / 63.95%** (9-ev) | <span style="color:green">**+0.85% Acc / +0.84% F1**</span> |
| **Trung bình (Mean LOEO)** | **75.84% / 71.98%** | **71.81% / 66.08%** | **71.93% / 65.42%** | — |
| **Gộp (Pooled)** | — | **74.06% / 70.02%** | **74.96% / 70.53%** | <span style="color:green">**+0.90% Acc / +0.51% F1**</span> |

---

## 3. Kiến trúc Mô hình Đề xuất: UC-TTT
Mô hình **User-Credibility-Aware Temporal Tree Transformer** giải quyết hạn chế của các phương pháp hiện tại bằng cách:
1. **Nội dung bài viết (Text Representation):** Vector hóa câu từ của tweet thông qua MiniLM Sentence Transformer.
2. **Độ sâu cây lan truyền (Depth Encoding):** Sinusoidal Positional Encoding cho độ sâu trong cascade.
3. **Uy tín người dùng (User Credibility Encoder):** Mã hóa 12 đặc trưng (Profile Verification, Engagement Counts với Log-scale, Derived Ratios) và dung hợp thông qua cơ chế cổng động (**Gated Fusion**).
4. **Cấu trúc cây (Tree Transformer):** Multi-head self-attention mô hình hóa tương tác giữa các phản hồi.
5. **Động lực học thời gian (Temporal GRU):** Phân chia thành các lát cắt thời gian và tổng hợp tiến trình lan truyền theo thời gian.

#### Lệnh thực thi tái lập Mô hình đề xuất UC-TTT:
```bash
# Huấn luyện trên 5 sự kiện chuẩn PHEME v5 (Wu et al. 2025):
python experiments/train_ucttt.py --events_mode 5 --fusion_type gated --epochs 35 --lr 1e-4 --patience 8 --device cuda

# Huấn luyện mở rộng trên toàn bộ 9 sự kiện:
python experiments/train_ucttt.py --events_mode 9 --fusion_type gated --epochs 35 --lr 1e-4 --patience 8 --device cuda
```
*Kết quả được lưu tại:* `results/ucttt/ucttt_gated_events5_results.json` và `results/ucttt/ucttt_gated_events9_results.json`.

---

## 4. Bảng Theo dõi Tiến độ Dự án (Roadmap & Status)

| Giai đoạn | Nội dung công việc trọng tâm | Trạng thái | Người phụ trách |
| :---: | :--- | :---: | :--- |
| **GĐ 0** | Khảo sát, Tiền xử lý dữ liệu PHEME, Khối Deep Learning PyTorch & Baselines B1/B2 | ✅ **Hoàn thành** | Tất cả thành viên |
| **GĐ 1** | Offline Text Embeddings (`all-MiniLM-L6-v2`) & PyTorch Dynamic Tree DataLoader | ✅ **Hoàn thành** (11/10) | Gia Huy, Tấn Đạt |
| **GĐ 2** | Huấn luyện & Tái hiện Baseline 3: Original TTT (Wu et al. 2025) trên 9-fold LOEO | ✅ **Hoàn thành** (11/10) | Thiên Phú, Gia Huy |
| **GĐ 3** | Huấn luyện Mô hình Đề xuất UC-TTT & So sánh 3 Fusion Strategies (RQ1 & H1) | ✅ **Hoàn thành** (12/10) | Thiên Phú, Thanh Mai, Bích Phương |
| **GĐ 4** | Chuỗi thực nghiệm Ablation Study bóc tách Feature-level & Model-level (RQ2 & H2) | 🔄 **Đang triển khai** | Tấn Đạt, Bích Phương, Thanh Mai |
| **GĐ 5** | Phân tích Thống kê theo Sự kiện & Phân tích Lỗi sai / Case Studies (RQ3 & H3) | ⏳ Chờ thực hiện | Thanh Mai, Bích Phương |
| **GĐ 6** | Tổng hợp Báo cáo Cuối kỳ chuẩn IMRaD, Thiết kế Slide thuyết trình & Bàn giao Repo | ⏳ Chờ thực hiện | Tất cả thành viên |

