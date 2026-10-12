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

---

---

## 3. CHECKPOINT NGÀY 11/10/2026: HOÀN THÀNH GIAI ĐOẠN 2 (TÁI HIỆN BASELINE 3 ORIGINAL TTT)

### 3.1. Đã hoàn thành các nhiệm vụ Giai đoạn 2 (Milestone 2 / Issue #3)
- **Module huấn luyện & đánh giá:**
  - [`src/training/engine.py`](src/training/engine.py): Hiện thực `EarlyStopping` (theo dõi Validation Macro-F1), `train_one_epoch`, và `evaluate_epoch`.
  - [`src/training/metrics.py`](src/training/metrics.py): Hiện thực `calculate_metrics_from_tensors` và các hàm xuất chỉ số chuẩn.
  - [`src/training/__init__.py`](src/training/__init__.py): Package init.
- **Script huấn luyện LOEO:**
  - [`experiments/train_ttt_baseline.py`](experiments/train_ttt_baseline.py): Hỗ trợ cả 2 chế độ `--events_mode 5` (PHEME v5 - chuẩn đối chứng Wu et al. 2025) và `--events_mode 9` (full PHEME).
  - Tinh chỉnh AdamW ($lr=1e-4$), Cosine Annealing scheduler, class-weighted CrossEntropyLoss.
  - Checkpoints tốt nhất đã được lưu tại `results/checkpoints/ttt_baseline/`.
- **Kết quả thực nghiệm chính thức (Đối chiếu 1-1 với bài báo Wu et al. 2025):**
  - **Chuẩn 5 sự kiện PHEME v5 (Wu et al. 2025):**
    - **Pooled Accuracy: 74.06%** (so với 75.84% bài báo, chênh lệch chỉ 1.78%).
    - **Pooled Macro-F1: 70.02%** (so với 71.98% bài báo, chênh lệch chỉ 1.96%).
    - **Mean LOEO Accuracy: 71.81% ± 5.58%**; **Mean LOEO Macro-F1: 66.08% ± 6.40%**.
    - Đặc biệt, fold `ottawashooting` đạt **70.22% Acc** và **69.75% F1** (trùng khớp gần như tuyệt đối với 70.25% Acc và 69.54% F1 trong Table 3 của bài báo).
  - **Mở rộng 9 sự kiện full PHEME:**
    - **Pooled Accuracy: 72.22%**; **Pooled Macro-F1: 70.50%**.
    - Mean LOEO Macro-F1: 56.24% ± 16.34% (do 4 sự kiện dị biệt/lệch pha nặng như `prince-toronto` 98% rumour và `ebola-essien` kéo tụt).
  - File kết quả: [`results/ttt_baseline_results.json`](results/ttt_baseline_results.json).

---

## 4. CHECKPOINT NGÀY 12/10/2026: HOÀN THÀNH GIAI ĐOẠN 3 (HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH ĐỀ XUẤT UC-TTT)

### 4.1. Đã hoàn thành các nhiệm vụ Giai đoạn 3 (Milestone 2 / Issue #4)
- **Tập lệnh huấn luyện mô hình đề xuất:**
  - [`experiments/train_ucttt.py`](experiments/train_ucttt.py): Hiện thực pipeline huấn luyện LOEO end-to-end cho mô hình UC-TTT, hỗ trợ chuyển đổi linh hoạt giữa 5 sự kiện (`--events_mode 5`) và 9 sự kiện (`--events_mode 9`).
  - Hỗ trợ 3 cơ chế dung hợp đặc trưng uy tín: Gated Attention (`--fusion_type gated`), Concat (`--fusion_type concat`), và Additive/Residual (`--fusion_type additive`).
  - Tự động đối chiếu delta ($\Delta\text{Accuracy}$, $\Delta\text{Macro-F1}$) 1-1 với Baseline 3 sau mỗi fold.
  - Tinh chỉnh `src/models/user_encoder.py` hỗ trợ alias `residual`/`additive`.
- **Kết quả thực nghiệm trên GPU CUDA (NVIDIA RTX 3050):**
  - **Lần 1: Chuẩn 5 sự kiện PHEME v5 (Wu et al. 2025):**
    - **Pooled Accuracy: 74.96%** (cải thiện **+0.90%** so với Baseline 3: 74.06%).
    - **Pooled Macro-F1: 70.53%** (cải thiện **+0.51%** so với Baseline 3: 70.02%).
    - Mean LOEO Accuracy: 71.93% ± 6.93%; Mean LOEO Macro-F1: 65.42% ± 7.56%.
    - Tại `charliehebdo`: Accuracy tăng từ 79.22% lên **82.54%** (+3.32%), F1 tăng lên **74.99%** (+0.82%).
    - Tại `sydneysiege`: F1 tăng từ 67.95% lên **71.04%** (+3.10%), Accuracy tăng lên **71.83%** (+1.64%).
    - File kết quả: [`results/ucttt/ucttt_gated_events5_results.json`](results/ucttt/ucttt_gated_events5_results.json) và [`results/ucttt_results.json`](results/ucttt_results.json).
  - **Lần 2: Mở rộng 9 sự kiện full PHEME:**
    - **Pooled Accuracy: 73.20%** (cải thiện **+0.98%** so với Baseline 3: 72.22%).
    - **Mean LOEO Macro-F1: 56.83% ± 16.77%** (cải thiện **+0.59%** so với Baseline 3: 56.24%).
    - **Đặc biệt, cả 5 sự kiện chính đều tăng F1 so với Baseline 3 khi huấn luyện trên 9 sự kiện:**
      - `ferguson`: Macro-F1 bứt phá từ 55.43% lên **60.65%** (**+5.22%**).
      - `sydneysiege`: Macro-F1 tăng từ 67.95% lên **72.00%** (**+4.06%**), Accuracy tăng lên **72.81%** (+2.62%).
      - `ottawashooting`: Macro-F1 tăng từ 69.75% lên **71.15%** (**+1.40%**), Accuracy tăng lên **71.24%** (+1.01%).
      - `charliehebdo`: Macro-F1 tăng từ 74.17% lên **75.19%** (**+1.02%**), Accuracy tăng lên **82.59%** (+3.37%).
      - `germanwings-crash`: Macro-F1 tăng từ 63.11% lên **63.95%** (**+0.84%**), Accuracy tăng lên **63.97%** (+0.85%).
    - **Độ chính xác nhận diện tin đồn (Rumour Precision):** Tăng vọt từ **63.74% lên 67.35% (+3.61%)**, chứng minh User Credibility giúp giảm đáng kể báo động giả.
    - File kết quả: [`results/ucttt/ucttt_gated_events9_results.json`](results/ucttt/ucttt_gated_events9_results.json).
- **Lưu trữ Checkpoints:**
  - Checkpoints tốt nhất của từng fold lưu tại `results/checkpoints/ucttt/`.

---

## 5. BẢNG SO SÁNH TỔNG HỢP TOÀN BỘ BENCHMARKS DỰ ÁN

| Mô hình | Đầu vào đặc trưng | Quy mô | Pooled Acc | Pooled Macro-F1 | Mean LOEO F1 ($\mu \pm \sigma$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B1a: TF-IDF + LR (Source Only)** | Text only | 9 Events | 71.77% | 67.56% | 56.17% ± 16.66% |
| **B1b: TF-IDF + LR (Source + Reactions)** | Text only | 9 Events | 70.60% | 65.67% | 52.85% ± 16.25% |
| **B2: Propagation Features + RF** | Graph structure only | 9 Events | 56.62% | 55.62% | 46.12% ± 8.88% |
| **B3: Original TTT (Wu et al. 2025)** | Text + Depth + Time | 5 Events | 74.06% | 70.02% | 66.08% ± 6.40% |
| **B3: Original TTT (Wu et al. 2025)** | Text + Depth + Time | 9 Events | 72.22% | 70.50% | 56.24% ± 16.34% |
| **UC-TTT (Proposed - Gated Fusion)** | Text + Depth + Time + **User Credibility** | **5 Events** | **74.96% (+0.90%)** | **70.53% (+0.51%)** | **65.42% ± 7.56%** |
| **UC-TTT (Proposed - Gated Fusion)** | Text + Depth + Time + **User Credibility** | **9 Events** | **73.20% (+0.98%)** | **70.12%** | **56.83% ± 16.77% (+0.59%)** |

---

## 6. KẾ HOẠCH BẮT ĐẦU NGAY CHO PHIÊN LÀM VIỆC TIẾP THEO (GIAI ĐOẠN 4)

Trọng tâm tiếp theo là **Giai đoạn 4: Chuỗi thực nghiệm bóc tách Ablation Studies (RQ2) & Kiểm định giả thuyết (RQ1)**:

1. **Kiểm định Thống kê (Statistical Hypothesis Testing - Issue #6):**
   - Viết module `src/utils/statistical_tests.py` thực hiện Paired Student's t-test và Wilcoxon signed-rank test trên các folds giữa UC-TTT và Original TTT ($p < 0.05$).
2. **So sánh các cơ chế Fusion (Fusion Ablation):**
   - Chạy so sánh nhanh `--fusion_type concat` và `--fusion_type additive`.
3. **Triển khai Feature-level & Model-level Ablation (Issue #5):**
   - Xây dựng script `experiments/run_ablation.py` hỗ trợ các chế độ bóc tách đặc trưng người dùng: `no_profile`, `no_counts`, `no_ratios`, `source_only` và bóc tách kiến trúc: `no_gru`, `no_tree`.

---

## 7. CÁC LỆNH ĐIỀU HÀNH NHANH (QUICK COMMANDS)

```bash
# Kiểm tra trạng thái Git
git status

# Xem kết quả UC-TTT 5 sự kiện
python -c "import json; d=json.load(open('results/ucttt/ucttt_gated_events5_results.json')); print('UC-TTT 5-Events Pooled F1:', d['pooled']['macro_f1'], '| Acc:', d['pooled']['accuracy'])"

# Xem kết quả UC-TTT 9 sự kiện
python -c "import json; d=json.load(open('results/ucttt/ucttt_gated_events9_results.json')); print('UC-TTT 9-Events Pooled F1:', d['pooled']['macro_f1'], '| Acc:', d['pooled']['accuracy'])"
```

