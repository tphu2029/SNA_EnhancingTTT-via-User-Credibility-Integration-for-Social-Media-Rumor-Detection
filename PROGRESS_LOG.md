# NHẬT KÝ TIẾN ĐỘ DỰ ÁN & HỒ SƠ CHUYỂN GIAO (PROGRESS LOG & CHECKPOINT)

**Thời gian lưu:** Ngày 06/10/2026  
**Đề tài:** Detecting the Spread of Misinformation in Social Network (Mở rộng Temporal Tree Transformer với User Credibility Features trên PHEME)  
**Môn học:** Mạng Xã Hội | **GVHD:** TS. Trần Hưng Nghiệp  
**Nhóm:** Nhóm 6 (Trần Thiên Phú, Mạc Nguyễn Gia Huy, Võ Tấn Đạt, Nguyễn Thị Thanh Mai, Bùi Phạm Bích Phương)  
**GitHub Repository:** `https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection.git`  
**Nhánh Git:** `main`

---

## 1. TÓM TẮT TẤT CẢ NHỮNG GÌ ĐÃ HOÀN THÀNH HÔM NAY

### 1.1. Khảo sát đề tài & Lập kế hoạch hành động
- Đã đọc hiểu và trích xuất toàn bộ yêu cầu từ file đề xuất: `[Main Template] Mini Proposal - THN.docx` và tài liệu hướng dẫn môn học `Tong_hop_bai_toan_PHEME_Misinformation.docx`.
- Đã lập bản kế hoạch chi tiết 12 tuần tại [`Plan.md`](Plan.md) (đã commit & push lên GitHub).

### 1.2. Chuẩn hóa và Tiền xử lý Dữ liệu PHEME
- **Cấu trúc thư mục dữ liệu:**
  - `data/raw/PHEME_veracity/all-rnr-annotated-threads/`: Chứa toàn bộ dữ liệu JSON thô của 9 sự kiện (`charliehebdo`, `ferguson`, `germanwings-crash`, `sydneysiege`, `ottawashooting`, `prince-toronto`, `putinmissing`, `gurlitt`, `ebola-essien`). Đã xóa sạch các file rác `.DS_Store` và `._*`.
  - `data/processed/pheme_processed.pkl` (181.5 MB): File nhị phân lưu toàn bộ 6,425 threads đã bóc tách cấu trúc cây và vector đặc trưng, nạp vào bộ nhớ chỉ trong 1-2 giây.
  - `data/processed/threads_summary.csv`: Bảng dữ liệu 6,425 dòng, chứa `thread_id`, `event`, `label`, `source_text`, thông tin user và đặc trưng cây lan truyền (có thể mở xem trên Excel).
  - `data/processed/events_summary.csv`: Bảng thống kê chi tiết các chỉ số của từng sự kiện (tỷ lệ rumour, số reaction trung bình, độ sâu cây, tỷ lệ verified user).
  - `data/processed/sample_thread.json`: Mẫu cấu trúc 1 thread hoàn chỉnh dạng JSON dễ đọc.

### 1.3. Kỹ nghệ đặc trưng User Credibility (`src/data/user_feature_builder.py`)
- Đã lập trình class `UserCredibilityExtractor` trích xuất đầy đủ **12 đặc trưng** chia làm 3 nhóm:
  1. *Profile & Verification:* `is_verified`, `has_default_avatar`, `has_description`, `log_account_age_days`.
  2. *Engagement Counts (Log-scale):* `log_followers`, `log_friends`, `log_statuses`, `log_favourites`, `log_listed`.
  3. *Derived Behavioral Ratios:* `log_reputation_ratio` ($\frac{\text{Followers}}{\text{Friends}+1}$), `log_status_rate`, `log_favourite_rate`.
- Đã tích hợp sẵn bộ lọc `select_feature_group` để tự động hóa các bài thử nghiệm bóc tách (*Ablation Studies*).

### 1.4. Đánh giá 2 Mô hình Cơ sở truyền thống (9-Fold LOEO)
- Chạy benchmark nghiêm ngặt bằng Leave-One-Event-Out (kết quả lưu tại `results/baseline_results.json`):
  - **B1a: TF-IDF + Logistic Regression (Source-only):** Pooled Acc 71.77%, Pooled Macro-F1 67.56% (Mean LOEO F1 56.17% ± 16.66%).
  - **B1b: TF-IDF + Logistic Regression (Source + Reactions):** Pooled Acc 70.60%, Pooled Macro-F1 65.67% (Mean LOEO F1 52.85% ± 16.25%).
  - **B2: Propagation Features + Random Forest:** Pooled Acc 56.62%, Pooled Macro-F1 55.62% (Mean LOEO F1 46.12% ± 8.88%).

### 1.5. Thiết kế và Kiểm thử Kiến trúc Học Sâu UC-TTT (PyTorch)
- Đã lập trình đầy đủ các thành phần kiến trúc:
  - `src/models/user_encoder.py`: MLP mã hóa 12-d đặc trưng uy tín + cơ chế cổng động **Gated Fusion**.
  - `src/models/tree_transformer.py`: Multi-Head Attention layer duyệt cây lan truyền.
  - `src/models/temporal_gru.py`: Lát cắt thời gian Equal-depth Time Windows + Temporal GRU.
  - `src/models/ucttt_classifier.py`: Ghép nối end-to-end, có công tắc `use_user_credibility=False` (chạy TTT gốc) hoặc `True` (chạy mô hình đề xuất UC-TTT).
- Đã test forward pass thành công trên GPU CUDA (`device: cuda`, shape `[batch_size, 2]`).

### 1.6. Quản trị mã nguồn
- Đã thiết lập `.gitignore` an toàn (chặn các file `.pkl` nặng 180MB và 137k files thô để tránh vượt quota 100MB của GitHub).
- Đã khởi tạo Git và đẩy mã nguồn lên nhánh `main` của repository GitHub.

---

## 2. CHECKPOINT NGÀY 11/10/2026: HOÀN THÀNH GIAI ĐOẠN 1 (OFFLINE EMBEDDINGS & DATALOADER)

### 2.1. Đã hoàn thành các nhiệm vụ Giai đoạn 1 (Milestone 1)
- **Issue #1 / PR #9 (`feat(data): implement offline embedding extraction with MiniLM`):**
  - Đã xây dựng script [`experiments/extract_text_embeddings.py`](experiments/extract_text_embeddings.py) trích xuất offline sentence embedding (384 chiều) bằng `sentence-transformers/all-MiniLM-L6-v2`.
  - Hỗ trợ batch encoding trên GPU CUDA với thanh tiến trình `tqdm`.
  - Thiết lập quy trình phát hành cache qua **GitHub Releases Tag `v0.1-data-cache`** để chia sẻ file dữ liệu nặng `text_embeddings.pt` mà không làm phình repo Git.
  - Cập nhật [`.gitignore`](.gitignore) chặn file `data/processed/*.pt`.
- **Issue #2 / PR #10 (`feat(data): implement PHEMETreeDataset and collate_tree_batch`):**
  - Đã xây dựng module [`src/data/dataset.py`](src/data/dataset.py) định nghĩa lớp `PHEMETreeDataset` và hàm gom batch động `collate_tree_batch`.
  - Hỗ trợ dynamic padding ma trận attention, liên kết cạnh tổ tiên-con cháu, và phân bổ các node vào $K=4$ equal-depth time windows.
  - Đã viết unit test hoàn chỉnh tại [`tests/test_dataloader.py`](tests/test_dataloader.py) (test mock batch và test real batch).
- **Trạng thái Git:**
  - Cả 2 PR #9 và #10 đã được merge thành công vào nhánh `main` (commit `453b43d`).

---

## 3. KẾ HOẠCH BẮT ĐẦU NGAY CHO PHIÊN LÀM VIỆC TIẾP THEO (GIAI ĐOẠN 2)

Trọng tâm tiếp theo là **Giai đoạn 2: Huấn luyện & Tái hiện Baseline 3 — Original TTT (Wu et al. 2025)**:

### Bước 1: Chuẩn bị file Text Embeddings cục bộ
- Tải file `text_embeddings.pt` từ GitHub Release `v0.1-data-cache` về đặt tại `data/processed/text_embeddings.pt`:
  ```powershell
  Invoke-WebRequest -Uri "https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection/releases/download/v0.1-data-cache/text_embeddings.pt" -OutFile "data/processed/text_embeddings.pt"
  ```
  *(hoặc chạy trích xuất lại trên máy local: `python experiments/extract_text_embeddings.py`)*

### Bước 2: Xây dựng Training & Evaluation Engine cho 9-Fold LOEO
- **Nhiệm vụ P-01 / Issue #3:** Tạo `src/training/engine.py` quản lý training loop, validation early stopping, scheduler, và tính metrics chuẩn (Accuracy, Macro-F1).
- Tích hợp bộ chia 9-fold Leave-One-Event-Out (`src/utils/loeo_splitter.py`).

### Bước 3: Lập trình script huấn luyện Baseline 3 (`experiments/train_ttt_baseline.py`)
- Cấu hình mô hình `UCTTTClassifier` với cờ `use_user_credibility=False` để tái hiện mô hình TTT nguyên bản của Wu et al. 2025.
- Chạy huấn luyện trên 9 folds LOEO, lưu checkpoint tốt nhất vào `results/ttt_baseline/` và tổng hợp kết quả đối chứng (kỳ vọng Macro-F1 ~71.98%).

---

## 4. CÁC LỆNH ĐIỀU HÀNH NHANH (QUICK COMMANDS)

```bash
# Kiểm tra trạng thái Git
git status

# Kiểm tra dữ liệu cache & DataLoader
python -m pytest tests/test_dataloader.py -k test_toy_tree

# Chạy trích xuất text embeddings (nếu cần tự sinh local)
python experiments/extract_text_embeddings.py --batch_size 256

# Kiểm tra CUDA GPU
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
