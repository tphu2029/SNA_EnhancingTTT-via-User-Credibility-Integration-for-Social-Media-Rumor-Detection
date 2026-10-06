# BẢNG NHIỆM VỤ DỰ ÁN & BACKLOG GITHUB ISSUES (PROJECT TODO & DETAILED ISSUES)

> **Đề tài:** Detecting the Spread of Misinformation in Social Network  
> **Hướng nghiên cứu:** Mở rộng Temporal Tree Transformer (TTT) với User Credibility Features trên PHEME  
> **Môn học:** Mạng Xã Hội | **GVHD:** TS. Trần Hưng Nghiệp | **Nhóm:** 6  
> **Tài liệu căn cứ:** [Plan.md](Plan.md), [PROGRESS_LOG.md](PROGRESS_LOG.md), `[Main Template] Mini Proposal - THN.docx`

---

## 📑 MỤC LỤC
1. [Ma trận phân công thành viên (RACI) & Các Giai đoạn theo Plan.md](#1-ma-trận-phân-công-thành-viên-raci--các-giai-đoạn-theo-planmd)
2. [Checklist công việc theo từng thành viên (Personal Backlogs)](#2-checklist-công-việc-theo-từng-thành-viên-personal-backlogs)
   - [Trần Thiên Phú (Lead & DL Architect)](#thành-viên-1-trần-thiên-phú---23521188-team-lead)
   - [Mạc Nguyễn Gia Huy (Data & Embedding Engineer)](#thành-viên-2-mạc-nguyễn-gia-huy---23520620)
   - [Võ Tấn Đạt (Ablation & Experiment Specialist)](#thành-viên-3-võ-tấn-đạt---23520278)
   - [Nguyễn Thị Thanh Mai (Feature Analysis & Statistical Lead)](#thành-viên-4-nguyễn-thị-thanh-mai---23520910)
   - [Bùi Phạm Bích Phương (Feature Ablation & Reporting Specialist)](#thành-viên-5-bùi-phạm-bích-phương---23521239)
3. [Bộ GitHub Issues mẫu chi tiết (Kèm Description kỹ thuật cụ thể)](#3-bộ-github-issues-mẫu-chi-tiết-kèm-description-kỹ-thuật-cụ-thể)
   - [Issue #1: [GĐ 1] Trích xuất Offline Text Embeddings bằng Sentence Transformer](#issue-1-gđ-1-trích-xuất-offline-text-embeddings-bằng-sentence-transformer)
   - [Issue #2: [GĐ 1] Xây dựng PyTorch Dataset & Dynamic Collate Function cho Tree Batching](#issue-2-gđ-1-xây-dựng-pytorch-dataset--dynamic-collate-function-cho-tree-batching)
   - [Issue #3: [GĐ 2] Huấn luyện & Tái hiện Baseline 3 — Original TTT (Wu et al. 2025)](#issue-3-gđ-2-huấn-luyện--tái-hiện-baseline-3--original-ttt-wu-et-al-2025)
   - [Issue #4: [GĐ 3] Huấn luyện Mô hình Đề xuất UC-TTT & Thử nghiệm Fusion Strategies (RQ1)](#issue-4-gđ-3-huấn-luyện-mô-hình-đề-xuất-uc-ttt--thử-nghiệm-fusion-strategies-rq1)
   - [Issue #5: [GĐ 4] Triển khai Chuỗi Thực nghiệm Ablation Study Chuyên sâu (RQ2)](#issue-5-gđ-4-triển-khai-chuỗi-thực-nghiệm-ablation-study-chuyên-sâu-rq2)
   - [Issue #6: [GĐ 3 & 5] Phân tích Thống kê EDA, Kiểm định Giả thuyết p-value & Event Analysis (RQ3)](#issue-6-gđ-3--5-phân-tích-thống-kê-eda-kiểm-định-giả-thuyết-p-value--event-analysis-rq3)
   - [Issue #7: [GĐ 4 & 5] Phân tích Lỗi sai (Error Analysis) & Tuyển chọn Case Studies](#issue-7-gđ-4--5-phân-tích-lỗi-sai-error-analysis--tuyển-chọn-case-studies)
   - [Issue #8: [GĐ 6] Hoàn thiện Báo cáo Cuối kỳ chuẩn IMRaD & Slide Thuyết trình](#issue-8-gđ-6-hoàn-thiện-báo-cáo-cuối-kỳ-chuẩn-imrad--slide-thuyết-trình)
4. [Quy chuẩn thực hiện & Định nghĩa hoàn thành (Definition of Done)](#4-quy-chuẩn-thực-hiện--định-nghĩa-hoàn-thành-definition-of-done)
   - [4.4. Quy trình chia sẻ & Tải file nặng (>100MB) bằng GitHub Releases](#44-quy-trình-chia-sẻ--tải-file-nặng-100mb-bằng-github-releases)

---

## 1. MA TRẬN PHÂN CÔNG THÀNH VIÊN (RACI) & CÁC GIAI ĐOẠN THEO PLAN.MD

### 1.1. Ma trận RACI (Kế thừa từ Mục 3 của Plan.md)
| MSSV | Họ và Tên | Vai trò chính | Trách nhiệm cụ thể theo Plan | Giai đoạn phụ trách chính |
| :---: | :--- | :--- | :--- | :---: |
| **23521188** | **Trần Thiên Phú** | Team Lead, Deep Learning Architect | Quản lý tiến độ; Thiết kế và tối ưu mô hình TTT & UC-TTT; Training 9-fold LOEO; Viết phần Proposed Method. | Giai đoạn 2, 3, 6 |
| **23520620** | **Mạc Nguyễn Gia Huy** | Data & Baseline Engineer | Trích xuất text embeddings offline; Xây dựng PyTorch DataLoader; Hỗ trợ chạy baseline TTT; Phân tích kết quả. | Giai đoạn 1, 2, 6 |
| **23520278** | **Võ Tấn Đạt** | Experiment & Ablation Specialist | Phụ trách thực hiện toàn bộ chuỗi thực nghiệm Ablation Study; Xây dựng biểu đồ nhạy tham số; Thống kê số liệu. | Giai đoạn 1, 4, 6 |
| **23520910** | **Nguyễn Thị Thanh Mai** | Feature Engineering & Analysis Lead | Phân tích phân bố User Credibility (EDA); Kiểm định thống kê (p-value); Phân tích RQ2 & RQ3; Viết Related Work & Discussion. | Giai đoạn 3, 4, 5, 6 |
| **23521239** | **Bùi Phạm Bích Phương** | Feature Ablation & Reporting Specialist | Phối hợp chạy Feature-level Ablation; Thực hiện Error Analysis & Case Studies; Thiết kế Slide & Biên soạn Báo cáo cuối kỳ. | Giai đoạn 3, 4, 5, 6 |

### 1.2. 6 Giai đoạn triển khai theo Plan.md
- **Giai đoạn 1:** Offline Text Feature Caching & PyTorch DataLoader. *(Phụ trách: Gia Huy, Tấn Đạt)*
- **Giai đoạn 2:** Huấn luyện & Tái hiện Baseline 3 — Original TTT (Wu et al. 2025). *(Phụ trách: Thiên Phú, Gia Huy)*
- **Giai đoạn 3:** Huấn luyện & Đánh giá Mô hình Đề xuất UC-TTT — RQ1 & H1. *(Phụ trách: Thiên Phú, Thanh Mai, Bích Phương)*
- **Giai đoạn 4:** Chuỗi Thực nghiệm Bóc tách — Ablation Studies — RQ2 & H2. *(Phụ trách: Tấn Đạt, Bích Phương, Thanh Mai)*
- **Giai đoạn 5:** Phân tích Từng Sự kiện & Error Analysis / Case Studies — RQ3 & H3. *(Phụ trách: Thanh Mai, Bích Phương)*
- **Giai đoạn 6:** Hoàn thiện Báo cáo cuối kỳ (IMRaD), Slide thuyết trình & Bàn giao Repo. *(Phụ trách: Tất cả thành viên)*

---

## 2. CHECKLIST CÔNG VIỆC THEO TỪNG THÀNH VIÊN (PERSONAL BACKLOGS)

### THÀNH VIÊN 1: TRẦN THIÊN PHÚ - 23521188 (TEAM LEAD)
- [ ] **Task P-01 (GĐ 2):** Xây dựng module huấn luyện và đánh giá chuẩn cho 9-Fold LOEO (`src/training/engine.py` & `src/training/metrics.py`).
- [ ] **Task P-02 (GĐ 2):** Lập trình script `experiments/train_ttt_baseline.py` và tái hiện Baseline 3 Original TTT (mục tiêu F1 ~71.98%).
- [ ] **Task P-03 (GĐ 3):** Lập trình script `experiments/train_ucttt.py`, huấn luyện 9 folds LOEO cho mô hình UC-TTT (mục tiêu F1 > 73.50%).
- [ ] **Task P-04 (GĐ 3):** Triển khai so sánh 3 cơ chế dung hợp: *Gated Attention*, *Concatenation*, và *Residual Additive*.
- [ ] **Task P-05 (GĐ 6):** Soạn thảo mục "Proposed Method" và phần mô tả kiến trúc toán học trong Báo cáo cuối kỳ.

---

### THÀNH VIÊN 2: MẠC NGUYỄN GIA HUY - 23520620
- [ ] **Task H-01 (GĐ 1):** Viết script `experiments/extract_text_embeddings.py` trích xuất offline embedding cho 6,425 source tweets và 105,354 reaction tweets bằng Sentence Transformer.
- [ ] **Task H-02 (GĐ 1):** Lưu trữ cache `data/processed/text_embeddings.pt` và kiểm tra tính toàn vẹn của tensor nhúng.
- [ ] **Task H-03 (GĐ 1):** Lập trình module `src/data/dataset.py` định nghĩa `PHEMETreeDataset` và `collate_fn` gom batch cây lan truyền động (padding, mask, time windows).
- [ ] **Task H-04 (GĐ 2):** Phối hợp cùng Trưởng nhóm tích hợp DataLoader vào pipeline huấn luyện Baseline 3 và tối ưu GPU memory.
- [ ] **Task H-05 (GĐ 6):** Hoàn thiện tài liệu hướng dẫn tiền xử lý dữ liệu và cấu trúc pipeline nạp batch trong `README.md`.

---

### THÀNH VIÊN 3: VÕ TẤN ĐẠT - 23520278
- [ ] **Task D-01 (GĐ 1):** Hỗ trợ kiểm thử và benchmark tốc độ gom batch động của DataLoader trên CPU và GPU.
- [ ] **Task D-02 (GĐ 4):** Xây dựng script điều phối ablation tự động `experiments/run_ablation.py` hỗ trợ các cờ lệnh `--ablation_mode`.
- [ ] **Task D-03 (GĐ 4):** Thực hiện Model-level Ablation trên 9 folds LOEO (`w/o Temporal GRU` và `w/o Tree Transformer`).
- [ ] **Task D-04 (GĐ 4):** Thống kê kết quả thực nghiệm bóc tách, viết script `experiments/plot_ablation_results.py` sinh biểu đồ cột độ suy giảm F1.
- [ ] **Task D-05 (GĐ 6):** Soạn thảo mục "Ablation Study Results" và lập các bảng biểu đối chiếu trong Báo cáo cuối kỳ.

---

### THÀNH VIÊN 4: NGUYỄN THỊ THANH MAI - 23520910
- [ ] **Task M-01 (GĐ 3):** Phân tích thống kê khám phá (EDA) phân bố 12 đặc trưng uy tín người dùng giữa Rumour và Non-rumour trên 9 sự kiện (`notebooks/user_credibility_eda.ipynb`).
- [ ] **Task M-02 (GĐ 3):** Viết script `src/utils/statistical_tests.py` thực hiện kiểm định ý nghĩa thống kê (*Paired t-test*, *Wilcoxon signed-rank test*, $p < 0.05$) khẳng định độ tin cậy của H1.
- [ ] **Task M-03 (GĐ 4):** Cùng Đạt và Phương phân tích kết quả Feature-level Ablation, lý giải sự đóng góp của nhóm Profile vs Counts vs Derived Ratios (RQ2 & H2).
- [ ] **Task M-04 (GĐ 5):** Thực hiện Event-level Analysis mổ xẻ sự khác biệt hành vi người dùng giữa 9 sự kiện PHEME để trả lời RQ3 & H3.
- [ ] **Task M-05 (GĐ 6):** Soạn thảo mục "Related Work" (phân tích TTT, SARD-Net, KG) và mục "Discussion" trong Báo cáo cuối kỳ.

---

### THÀNH VIÊN 5: BÙI PHẠM BÍCH PHƯƠNG - 23521239
- [ ] **Task BP-01 (GĐ 3):** Phối hợp cùng Trưởng nhóm theo dõi logs huấn luyện 9 folds LOEO của UC-TTT và tổng hợp kết quả từng fold.
- [ ] **Task BP-02 (GĐ 4):** Thực thi các cấu hình Feature-level Ablation: `Profile Only`, `Counts Only`, `Ratios Only`, `Source User Only`.
- [ ] **Task BP-03 (GĐ 5):** Tiến hành Error Analysis: Trích xuất các ca False Positives và False Negatives; lọc các mẫu UC-TTT sửa sai thành công cho TTT gốc.
- [ ] **Task BP-04 (GĐ 5):** Tuyển chọn và xây dựng 4-6 Case Studies tiêu biểu có hình ảnh tweet, bảng thông số user và phân tích định tính.
- [ ] **Task BP-05 (GĐ 6):** Thiết kế bộ Slide thuyết trình báo cáo đồ án và chủ trì tổng hợp định dạng Báo cáo cuối kỳ chuẩn IMRaD.

---

## 3. BỘ GITHUB ISSUES MẪU CHI TIẾT (KÈM DESCRIPTION KỸ THUẬT CỤ THỂ)

---

### Issue #1: [GĐ 1] Trích xuất Offline Text Embeddings bằng Sentence Transformer

```markdown
## Tiêu đề Issue:
[Data] [Giai đoạn 1] Xây dựng script trích xuất Offline Text Embeddings cho dữ liệu PHEME

## Người phụ trách (Assignee):
@GiaHuy (23520620 - Mạc Nguyễn Gia Huy)

## Người phối hợp (Reviewer):
@TanDat (23520278 - Võ Tấn Đạt)

## Nhãn (Labels):
`giai-doan-1`, `data-pipeline`, `enhancement`, `priority: high`

## Milestone:
Milestone 1: Data Pipeline & Baseline 3 (Tuần 5 - 6)

---

### 📖 DESCRIPTION (MÔ TẢ CHI TIẾT KỸ THUẬT):

#### 1. Bối cảnh & Mục tiêu:
Dữ liệu PHEME sau khi tiền xử lý tại `data/processed/pheme_processed.pkl` chứa **6,425 threads** với tổng cộng hơn **111,000 tweets** (gồm tweet gốc và reactions). Nếu để mô hình nạp raw text và fine-tune trực tiếp qua Transformer trong mỗi epoch của 9-fold LOEO, thời gian huấn luyện sẽ rất chậm và dễ gây tràn bộ nhớ GPU. Do đó, theo Mục 2 - Giai đoạn 1 của `Plan.md`, nhóm cần trích xuất offline sentence embeddings một lần duy nhất và lưu cache nhị phân ra đĩa.

#### 2. Dữ liệu đầu vào (Input):
- File cache: `data/processed/pheme_processed.pkl` (Chứa list các dict thread; mỗi thread có `source_tweet` và danh sách `reactions`, trong đó mỗi node có `tweet_id` và `text`).

#### 3. Nhiệm vụ cụ thể cần thực hiện:
1. Tạo tệp mã nguồn: `experiments/extract_text_embeddings.py`.
2. Sử dụng thư viện `sentence-transformers` hoặc HuggingFace `transformers` với pre-trained model:
   - Mô hình đề xuất chính: `sentence-transformers/all-MiniLM-L6-v2` (output 384 chiều, tốc độ trích xuất cực nhanh).
   - Thiết lập fallback dự phòng: `microsoft/deberta-v3-small`.
3. Viết vòng lặp thu gom toàn bộ văn bản của source tweets và reactions, loại bỏ trùng lặp tweet_id (nếu có), làm sạch cơ bản (loại bỏ URL, ký tự điều khiển rác).
4. Thực hiện batch encoding trên GPU CUDA (`batch_size=128` hoặc `256`) với `torch.no_grad()`.
5. Tạo dictionary ánh xạ: `{int(tweet_id): torch.FloatTensor(embedding_vector)}`.
6. Lưu tensor dictionary ra đĩa: `data/processed/text_embeddings.pt` bằng `torch.save()`.
7. Viết hàm kiểm tra (sanity check) đảm bảo:
   - Tổng số key trong dict khớp với tổng số tweet duy nhất trong `pheme_processed.pkl`.
8. **Upload file nặng lên GitHub Releases:** Đính kèm file `text_embeddings.pt` lên Release với tag `v0.1-data-cache` (xem [Hướng dẫn chi tiết tại Mục 4.4](#quy-trinh-releases) hoặc xem trên [GitHub](https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection/blob/main/TODO.md#quy-trinh-releases)):
   - **Bước 1:** Vào tab **Releases** trên repo $\rightarrow$ bấm **Draft a new release**.
   - **Bước 2:** Đặt tag `v0.1-data-cache`, tiêu đề `PHEME Processed Cache & Offline Text Embeddings`.
   - **Bước 3:** Kéo thả file `data/processed/text_embeddings.pt` vào ô binaries $\rightarrow$ bấm **Publish release**.

#### 4. Lệnh chạy thực thi & Kiểm thử:
```bash
python experiments/extract_text_embeddings.py --model_name all-MiniLM-L6-v2 --batch_size 256
```

#### 5. Sản phẩm bàn giao (Deliverables & DoD):
- [ ] Tệp `experiments/extract_text_embeddings.py` hoàn chỉnh, có log tiến độ `tqdm`.
- [ ] Tệp cache `data/processed/text_embeddings.pt` sẵn sàng trong thư mục dữ liệu cục bộ.
- [ ] Đã upload `text_embeddings.pt` lên **GitHub Releases (Tag `v0.1-data-cache`)**.
- [ ] Bổ sung đường dẫn `data/processed/*.pt` vào `.gitignore` để tránh đẩy file nặng lên Git commit.
```

---

### Issue #2: [GĐ 1] Xây dựng PyTorch Dataset & Dynamic Collate Function cho Tree Batching

```markdown
## Tiêu đề Issue:
[Data] [Giai đoạn 1] Lập trình PHEMETreeDataset và collate_fn gom batch cây lan truyền động

## Người phụ trách (Assignee):
@GiaHuy (23520620 - Mạc Nguyễn Gia Huy)

## Người phối hợp (Reviewer):
@ThienPhu (23521188 - Trần Thiên Phú), @TanDat (23520278 - Võ Tấn Đạt)

## Nhãn (Labels):
`giai-doan-1`, `data-pipeline`, `model-infra`, `priority: high`

## Milestone:
Milestone 1: Data Pipeline & Baseline 3 (Tuần 5 - 6)

---

### 📖 DESCRIPTION (MÔ TẢ CHI TIẾT KỸ THUẬT):

#### 1. Bối cảnh & Mục tiêu:
Cây lan truyền tin đồn (propagation tree) trong PHEME có kích thước rất biến thiên: có thread chỉ có 1 bài đăng gốc, có thread lên đến vài trăm reactions. Để đưa vào kiến trúc `TreeTransformer` và `TemporalGRU` (đã viết tại `src/models/`), chúng ta cần một PyTorch Dataset và hàm gom batch (`collate_fn`) tùy biến để padding ma trận kề, ma trận reachability mask và chia node vào các cửa sổ thời gian $K=4$.

#### 2. Dữ liệu & Module sử dụng:
- Cache dữ liệu: `data/processed/pheme_processed.pkl`.
- Cache text embeddings: `data/processed/text_embeddings.pt`.
- Module trích xuất đặc trưng người dùng: `src/data/user_feature_builder.py` (cung cấp 12 đặc trưng float).

#### 3. Nhiệm vụ cụ thể cần thực hiện:
1. Tạo tệp mã nguồn: `src/data/dataset.py`.
2. Xây dựng lớp `PHEMETreeDataset(torch.utils.data.Dataset)`:
   - `__init__(self, threads_data, text_embeddings, user_feature_extractor)`: Nhận danh sách các thread của các event chỉ định.
   - `__len__(self)`: Trả về số lượng thread.
   - `__getitem__(self, idx)`: Trích xuất thông tin của 1 thread:
     - `node_texts_emb`: Tensor `[N, D_text]` (lấy từ cache embedding theo `tweet_id`).
     - `node_users`: Tensor `[N, 12]` (lấy từ `UserCredibilityExtractor.extract_features()`).
     - `tree_structure`: Danh sách cạnh cha-con `(parent_idx, child_idx)` và độ sâu `depth`.
     - `timestamps`: Danh sách mốc thời gian đăng bài để phân chia time windows.
     - `label`: Nhãn phân loại (0: Non-rumour, 1: Rumour).
     - `event`: Tên sự kiện (phục vụ chia fold LOEO).
3. Xây dựng hàm `collate_tree_batch(batch)`:
   - Tìm số node lớn nhất trong batch: `max_nodes = max(item['num_nodes'] for item in batch)`.
   - Padding `text_features` và `user_features` lên `[B, max_nodes, D]`.
   - Tạo `attention_mask`: Tensor boolean `[B, max_nodes, max_nodes]` (chỉ cho phép các node có quan hệ tổ tiên-con cháu hoặc trong cùng cửa sổ chú ý đến nhau).
   - Phân bố node vào $K=4$ Equal-depth Time Windows: Tensor `time_window_assignments` shape `[B, max_nodes]`.
   - Gom `labels` thành Tensor `[B]`.
4. Tạo tệp kiểm thử `tests/test_dataloader.py`:
   - Nạp thử 1 batch 16 threads, in kích thước các tensor đầu ra, kiểm tra không bị lỗi CUDA OOM.

#### 4. Lệnh chạy thực thi & Kiểm thử:
```bash
python tests/test_dataloader.py
```

#### 5. Sản phẩm bàn giao (Deliverables & DoD):
- [ ] Tệp `src/data/dataset.py` hoạt động ổn định, nạp dữ liệu nhanh (< 0.1s / batch).
- [ ] Test suite `tests/test_dataloader.py` pass 100%, shape tensor khớp với input của `src/models/ucttt_classifier.py`.
```

---

### Issue #3: [GĐ 2] Huấn luyện & Tái hiện Baseline 3 — Original TTT (Wu et al. 2025)

```markdown
## Tiêu đề Issue:
[Model] [Giai đoạn 2] Huấn luyện 9-Fold LOEO tái hiện Baseline 3 — Original Temporal Tree Transformer

## Người phụ trách (Assignee):
@ThienPhu (23521188 - Trần Thiên Phú)

## Người phối hợp (Reviewer):
@GiaHuy (23520620 - Mạc Nguyễn Gia Huy)

## Nhãn (Labels):
`giai-doan-2`, `deep-learning`, `baseline`, `priority: high`

## Milestone:
Milestone 1: Data Pipeline & Baseline 3 (Tuần 5 - 6)

---

### 📖 DESCRIPTION (MÔ TẢ CHI TIẾT KỸ THUẬT):

#### 1. Bối cảnh & Mục tiêu:
Temporal Tree Transformer (TTT) của tác giả Wu et al. (PLOS ONE 2025) là mô hình nền tảng mà đề tài kế thừa và mở rộng. Trước khi chứng minh đóng góp của User Credibility, nhóm bắt buộc phải tái hiện lại kết quả của mô hình TTT gốc trên PHEME làm đối chứng chính thức (*Benchmarking Target*), với mục tiêu đạt Acc ~75.84% và Macro-F1 tiệm cận ~71.98%.

#### 2. Cấu hình mô hình & Dữ liệu:
- Sử dụng mô hình: `src/models/ucttt_classifier.py` với cờ `use_user_credibility=False`.
- Giao thức thực nghiệm: **9-Fold Leave-One-Event-Out (LOEO)**:
  - Danh sách 9 sự kiện: `charliehebdo`, `ferguson`, `germanwings-crash`, `sydneysiege`, `ottawashooting`, `prince-toronto`, `putinmissing`, `gurlitt`, `ebola-essien`.
  - Mỗi fold: 7 events train, 1 event validation (chọn checkpoint tốt nhất), 1 event test.

#### 3. Nhiệm vụ cụ thể cần thực hiện:
1. Viết engine huấn luyện: `src/training/engine.py` (hàm `train_one_epoch`, `evaluate`).
2. Viết module tính chỉ số: `src/training/metrics.py` (tính Loss, Accuracy, Precision, Recall, Macro-F1).
3. Tạo script thực thi: `experiments/train_ttt_baseline.py`.
4. Thiết lập siêu tham số chuẩn theo bài báo gốc:
   - Optimizer: `AdamW`, Learning rate: $2e-4$ (có Cosine Annealing LR Scheduler).
   - Batch size: 32, Weight decay: $1e-4$, Dropout: 0.2.
   - Số time windows: $K=4$, Hidden dimension: 128, Số Attention Heads: 4.
   - Early Stopping: Kiên nhẫn 10 epochs dựa trên `Validation Macro-F1`.
5. Huấn luyện lần lượt 9 folds trên GPU. Thu thập kết quả từng fold và tính:
   - **Mean LOEO Macro-F1 & Accuracy** ($\mu \pm \sigma$).
   - **Pooled Macro-F1 & Accuracy** (gộp toàn bộ dự đoán 9 folds).
6. Xuất toàn bộ kết quả chi tiết ra file JSON: `results/ttt_baseline_results.json`.

#### 4. Lệnh chạy thực thi & Kiểm thử:
```bash
python experiments/train_ttt_baseline.py --epochs 50 --batch_size 32 --lr 2e-4 --device cuda
```

#### 5. Sản phẩm bàn giao (Deliverables & DoD):
- [ ] Tệp `experiments/train_ttt_baseline.py` và `src/training/engine.py`.
- [ ] Tệp kết quả `results/ttt_baseline_results.json` ghi nhận đầy đủ 9 folds.
- [ ] Checkpoints mô hình được lưu trong `results/checkpoints/ttt_baseline/`.
- [ ] Bảng tổng hợp so sánh với 2 Baseline trước (TF-IDF+LR và Prop+RF) sẵn sàng cho báo cáo.
```

---

### Issue #4: [GĐ 3] Huấn luyện Mô hình Đề xuất UC-TTT & Thử nghiệm Fusion Strategies (RQ1)

```markdown
## Tiêu đề Issue:
[Model] [Giai đoạn 3] Huấn luyện mô hình đề xuất UC-TTT trên 9 Folds LOEO và thử nghiệm cơ chế Fusion (RQ1)

## Người phụ trách (Assignee):
@ThienPhu (23521188 - Trần Thiên Phú)

## Người phối hợp (Reviewer):
@ThanhMai (23520910 - Nguyễn Thị Thanh Mai), @BichPhuong (23521239 - Bùi Phạm Bích Phương)

## Nhãn (Labels):
`giai-doan-3`, `core-feature`, `deep-learning`, `priority: critical`

## Milestone:
Milestone 2: Proposed UC-TTT & RQ1 Validation (Tuần 7 - 8)

---

### 📖 DESCRIPTION (MÔ TẢ CHI TIẾT KỸ THUẬT):

#### 1. Bối cảnh & Mục tiêu:
Đây là đóng góp khoa học cốt lõi của đề tài nhằm trả lời **Research Question 1 (RQ1)** và kiểm chứng **Hypothesis H1**: *"Bổ sung User Credibility vào TTT giúp cải thiện Macro-F1 so với TTT gốc"*. Mục tiêu là đưa Macro-F1 từ ~71.98% lên **$\ge 73.50\%$** trên cùng thể thức đánh giá 9-Fold LOEO.

#### 2. Cấu trúc mô hình & Điểm mới:
- Sử dụng mô hình: `src/models/ucttt_classifier.py` với cờ `use_user_credibility=True`.
- Vector 12 đặc trưng từ `src/data/user_feature_builder.py` được đưa qua `UserCredibilityEncoder` để chiếu lên không gian tiềm ẩn, sau đó dung hợp với Text Embedding trước khi đi vào `TreeTransformer`.

#### 3. Nhiệm vụ cụ thể cần thực hiện:
1. Tạo script huấn luyện: `experiments/train_ucttt.py`.
2. Cài đặt và thực nghiệm so sánh 3 chiến lược dung hợp (*Fusion Strategy*) tại `src/models/user_encoder.py`:
   - **Gated Attention Fusion (Đề xuất chính):** Dùng vector cổng điều khiển $\mathbf{g} = \sigma(W_g [\mathbf{x}_t; \mathbf{x}_u] + b_g)$, đầu ra $\mathbf{h} = \mathbf{g} \odot \mathbf{x}_t + (1 - \mathbf{g}) \odot \mathbf{x}_u$.
   - **Concatenation + Linear Projection:** Nối $[\mathbf{x}_t; \mathbf{x}_u]$ rồi nhân qua ma trận trọng số $W_c$.
   - **Residual Additive:** Cộng trực tiếp $\mathbf{x}_t + \text{MLP}(\mathbf{x}_u)$.
3. Huấn luyện trọn vẹn 9 folds LOEO cho UC-TTT với cơ chế Gated Fusion (cố định seed 42, cùng split dữ liệu với Baseline 3).
4. Lưu logs chi tiết per-epoch, validation curves và checkpoints tốt nhất cho từng fold.
5. Xuất báo cáo kết quả chi tiết: `results/ucttt_results.json` (chứa Accuracy, Precision, Recall, Macro-F1 của từng fold).
6. Tính độ chênh lệch hiệu năng: $\Delta \text{Macro-F1} = \text{F1}_{\text{UC-TTT}} - \text{F1}_{\text{TTT}}$.

#### 4. Lệnh chạy thực thi & Kiểm thử:
```bash
# Huấn luyện mô hình đề xuất UC-TTT
python experiments/train_ucttt.py --fusion_type gated --epochs 50 --lr 2e-4 --device cuda

# Thử nghiệm so sánh các cơ chế fusion khác
python experiments/train_ucttt.py --fusion_type concat --epochs 50 --device cuda
python experiments/train_ucttt.py --fusion_type residual --epochs 50 --device cuda
```

#### 5. Sản phẩm bàn giao (Deliverables & DoD):
- [ ] Tệp `experiments/train_ucttt.py` và module fusion hoàn chỉnh.
- [ ] File kết quả `results/ucttt_results.json` khẳng định Macro-F1 vượt trội so với baseline.
- [ ] Bảng số liệu so sánh giữa 3 cơ chế Fusion để đưa vào Báo cáo cuối kỳ.
```

---

### Issue #5: [GĐ 4] Triển khai Chuỗi Thực nghiệm Ablation Study Chuyên sâu (RQ2)

```markdown
## Tiêu đề Issue:
[Experiment] [Giai đoạn 4] Thực hiện chuỗi thực nghiệm Ablation Study bóc tách đặc trưng và mô hình (RQ2)

## Người phụ trách (Assignee):
@TanDat (23520278 - Võ Tấn Đạt)

## Người phối hợp (Reviewer):
@BichPhuong (23521239 - Bùi Phạm Bích Phương), @ThanhMai (23520910 - Nguyễn Thị Thanh Mai)

## Nhãn (Labels):
`giai-doan-4`, `ablation`, `experiments`, `priority: high`

## Milestone:
Milestone 3: Ablation Studies & Statistical Testing (Tuần 9 - 10)

---

### 📖 DESCRIPTION (MÔ TẢ CHI TIẾT KỸ THUẬT):

#### 1. Bối cảnh & Mục tiêu:
Nhằm trả lời **Research Question 2 (RQ2)** và kiểm chứng **Hypothesis H2**: *"Xác định nhóm đặc trưng người dùng và thành phần kiến trúc nào đóng vai trò then chốt trong việc cải thiện độ chính xác phát hiện tin đồn"*. Chuỗi thực nghiệm bóc tách cần được chạy nhất quán trên cùng giao thức 9-fold LOEO.

#### 2. Các biến thể cần thực nghiệm (Ablation Variants):
Theo Mục 2 - Giai đoạn 4 của `Plan.md`:
##### A. Nhóm bóc tách đặc trưng người dùng (Feature-level Ablation):
1. `UC-TTT (Full)`: Đầy đủ 12 đặc trưng (chuẩn đối chiếu).
2. `w/o Profile Features`: Bỏ nhóm xác minh & tuổi tài khoản (còn 8 đặc trưng).
3. `w/o Raw Counts`: Bỏ nhóm follower, friend, status log-scale (còn 7 đặc trưng).
4. `w/o Derived Ratios`: Bỏ các tỷ lệ chuẩn hóa reputation ratio, status rate (còn 9 đặc trưng).
5. `Profile Only` / `Counts Only` / `Ratios Only`: Chỉ dùng duy nhất từng nhóm đặc trưng tương ứng.
6. `Source User Only`: Chỉ dùng thông tin user của tweet gốc, đặt toàn bộ user phản hồi về 0 (đánh giá tầm quan trọng của người phát tán ban đầu so với người tương tác).

##### B. Nhóm bóc tách thành phần kiến trúc (Model-level Ablation):
1. `w/o Temporal GRU`: Giữ Tree Transformer tĩnh, bỏ chiều động lực học thời gian (chứng minh giá trị của tiến trình thời gian).
2. `w/o Tree Transformer`: Chỉ dùng GRU chuỗi thời gian, bỏ cấu trúc cây lan truyền (chứng minh giá trị của topo mạng lan truyền).

#### 3. Nhiệm vụ cụ thể cần thực hiện:
1. Viết script điều phối tự động: `experiments/run_ablation.py` hỗ trợ tham số `--feature_group` và `--model_component`.
2. Tận dụng hàm `select_feature_group()` đã có sẵn trong `src/data/user_feature_builder.py`.
3. Chạy 9-fold LOEO cho toàn bộ các cấu hình trên GPU.
4. Thu thập toàn bộ kết quả Accuracy, Macro-F1 của từng cấu hình vào: `results/ablation/ablation_summary.json`.
5. Viết script `experiments/plot_ablation_results.py` sinh biểu đồ cột thể hiện độ giảm Macro-F1 ($\Delta \text{F1}$) khi khuyết thiếu từng nhóm tính năng.

#### 4. Lệnh chạy thực thi & Kiểm thử:
```bash
# Chạy ablation theo nhóm feature
python experiments/run_ablation.py --feature_group without_ratios --device cuda
python experiments/run_ablation.py --feature_group source_only --device cuda

# Chạy script vẽ biểu đồ
python experiments/plot_ablation_results.py --input results/ablation/ablation_summary.json
```

#### 5. Sản phẩm bàn giao (Deliverables & DoD):
- [ ] Tệp `experiments/run_ablation.py` và script vẽ biểu đồ `experiments/plot_ablation_results.py`.
- [ ] Bảng số liệu tổng hợp `results/ablation/ablation_summary.json`.
- [ ] Biểu đồ trực quan chất lượng cao `results/figures/ablation_feature_contribution.png` để đưa vào báo cáo và slide.
```

---

### Issue #6: [GĐ 3 & 5] Phân tích Thống kê EDA, Kiểm định Giả thuyết p-value & Event Analysis (RQ3)

```markdown
## Tiêu đề Issue:
[Analysis] [Giai đoạn 3 & 5] Phân tích thống kê EDA, kiểm định p-value và phân tích 9 sự kiện (RQ3)

## Người phụ trách (Assignee):
@ThanhMai (23520910 - Nguyễn Thị Thanh Mai)

## Người phối hợp (Reviewer):
@BichPhuong (23521239 - Bùi Phạm Bích Phương), @ThienPhu (23521188 - Trần Thiên Phú)

## Nhãn (Labels):
`giai-doan-3`, `giai-doan-5`, `analysis`, `statistics`, `priority: high`

## Milestone:
Milestone 3 & 4: Analysis, RQ3 & Event Studies (Tuần 9 - 11)

---

### 📖 DESCRIPTION (MÔ TẢ CHI TIẾT KỸ THUẬT):

#### 1. Bối cảnh & Mục tiêu:
Một công trình nghiên cứu khoa học đạt điểm xuất sắc bắt buộc phải chứng minh:
1. Mức độ cải thiện của mô hình đề xuất có **ý nghĩa thống kê** ($p < 0.05$) chứ không phải do may rủi.
2. Trả lời trọn vẹn **Research Question 3 (RQ3)** và **Hypothesis H3**: Giải thích tại sao User Credibility giúp tăng mạnh hiệu năng ở sự kiện này nhưng lại ít ở sự kiện khác dựa trên bản chất phân bố người dùng của từng event.

#### 2. Dữ liệu & Công cụ sử dụng:
- Bảng thống kê sự kiện: `data/processed/events_summary.csv` và `data/processed/threads_summary.csv`.
- Kết quả 9 folds của Baseline 3 (`results/ttt_baseline_results.json`) và UC-TTT (`results/ucttt_results.json`).
- Thư viện: `scipy.stats`, `seaborn`, `matplotlib`, `pandas`.

#### 3. Nhiệm vụ cụ thể cần thực hiện:
1. **Phân tích EDA Đặc trưng Người dùng:**
   - Tạo notebook `notebooks/user_credibility_eda.ipynb`.
   - Phân tích phân bố của 12 đặc trưng (KDE plot / Boxplot): So sánh giữa tài khoản đăng Rumour vs Non-rumour (Tỷ lệ có tick xanh, tỷ lệ dùng avatar mặc định, phân bố follower log-scale, reputation ratio).
   - Xuất hình ảnh trực quan: `results/figures/eda_user_distributions.png`.
2. **Kiểm định Ý nghĩa Thống kê (Hypothesis Testing cho RQ1):**
   - Viết script `src/utils/statistical_tests.py`:
     - Thực hiện **Paired Student's t-test** trên 9 cặp kết quả Macro-F1 tương ứng của 9 folds LOEO giữa UC-TTT và Original TTT.
     - Thực hiện **Wilcoxon signed-rank test** (kiểm định phi tham số bổ trợ).
     - Báo cáo rõ giá trị t-statistic, W-statistic và kết luận ý nghĩa với mức tin cậy $95\%$ ($p < 0.05$).
3. **Phân tích Sự khác biệt Giữa các Sự kiện (Event-level Analysis cho RQ3):**
   - Lập bảng ma trận so sánh F1 tăng/giảm trên từng sự kiện trong 9 sự kiện PHEME.
   - Đối chiếu với đặc trưng sự kiện trong `events_summary.csv`: Ví dụ sự kiện `charliehebdo` hay `ferguson` có lượng tài khoản mới tạo và bot tham gia cao, khiến đặc trưng uy tín phát huy tác dụng rõ rệt nhất.
   - Soạn thảo đoạn văn phân tích lập luận chặt chẽ để đưa vào chương Discussion của báo cáo.

#### 4. Lệnh chạy thực thi & Kiểm thử:
```bash
python src/utils/statistical_tests.py --baseline results/ttt_baseline_results.json --proposed results/ucttt_results.json
```

#### 5. Sản phẩm bàn giao (Deliverables & DoD):
- [ ] Tệp `src/utils/statistical_tests.py` xuất ra bảng số liệu t-test và p-value rõ ràng.
- [ ] Notebook `notebooks/user_credibility_eda.ipynb` hoàn thiện kèm biểu đồ.
- [ ] Bản thảo văn bản trả lời trọn vẹn RQ3 sẵn sàng ghép vào Báo cáo cuối kỳ.
```

---

### Issue #7: [GĐ 4 & 5] Phân tích Lỗi sai (Error Analysis) & Tuyển chọn Case Studies

```markdown
## Tiêu đề Issue:
[Analysis] [Giai đoạn 5] Phân tích chi tiết các ca dự đoán sai (Error Analysis) và tuyển chọn Case Studies

## Người phụ trách (Assignee):
@BichPhuong (23521239 - Bùi Phạm Bích Phương)

## Người phối hợp (Reviewer):
@ThanhMai (23520910 - Nguyễn Thị Thanh Mai), @ThienPhu (23521188 - Trần Thiên Phú)

## Nhãn (Labels):
`giai-doan-5`, `analysis`, `case-study`, `priority: medium`

## Milestone:
Milestone 4: Event Studies & Final Artifacts (Tuần 11)

---

### 📖 DESCRIPTION (MÔ TẢ CHI TIẾT KỸ THUẬT):

#### 1. Bối cảnh & Mục tiêu:
Theo Mục 2 - Giai đoạn 5 của `Plan.md`, để bài báo cáo có chiều sâu học thuật và thuyết phục hội đồng giám khảo, nhóm cần thực hiện Error Analysis định tính nhằm mổ xẻ những giới hạn của mô hình và chứng minh bằng trực quan lý do tại sao thông tin người dùng lại giúp sửa sai cho các mô hình truyền thống.

#### 2. Dữ liệu đầu vào:
- Dự đoán chi tiết của 6,425 threads từ Baseline 3 TTT và Đề xuất UC-TTT (file nhãn thực tế $y$ và nhãn dự đoán $\hat{y}$).
- Thông tin ngữ cảnh từ `data/processed/threads_summary.csv` (text tweet gốc, số lượng reaction, thông số user).

#### 3. Nhiệm vụ cụ thể cần thực hiện:
1. Viết script trích xuất dữ liệu phân tích: `experiments/extract_error_cases.py`.
2. Phân loại 4 nhóm mẫu điển hình:
   - **Nhóm 1 (Success Cases of UC-TTT):** Các mẫu mà Baseline TTT gốc đoán SAI, nhưng UC-TTT đoán ĐÚNG nhờ thông tin uy tín người dùng. (Ví dụ: Một tin giật gân có nội dung hoang đường nhưng được đăng bởi cơ quan báo chí có tick xanh lâu năm -> UC-TTT nhận diện đúng là Non-rumour).
   - **Nhóm 2 (Deceptive Rumour Cases):** Các mẫu cả 2 mô hình cùng đoán SAI do tin đồn được lan truyền bởi các tài khoản uy tín cao hoặc bị hack.
   - **Nhóm 3 (False Positives):** Tin thật nhưng bị đoán nhầm là tin đồn do tài khoản đăng là tài khoản mới tạo ít follower.
   - **Nhóm 4 (False Negatives):** Tin đồn tinh vi không bị phát hiện.
3. Tuyển chọn **4 đến 6 Case Studies tiêu biểu nhất** (mỗi case có: Tweet ID, Nội dung văn bản, Profile user gốc, Cấu trúc cây phản hồi, Phân tích lý do mô hình đoán đúng/sai).
4. Định dạng các Case Studies thành các khung minh họa trực quan (có thể chụp ảnh minh họa tweet gốc) để đưa vào Báo cáo cuối kỳ và Slide thuyết trình.

#### 4. Lệnh chạy thực thi & Kiểm thử:
```bash
python experiments/extract_error_cases.py --baseline_preds results/ttt_baseline_results.json --proposed_preds results/ucttt_results.json --output results/case_studies.json
```

#### 5. Sản phẩm bàn giao (Deliverables & DoD):
- [ ] Tệp `experiments/extract_error_cases.py`.
- [ ] Báo cáo phân tích định tính `results/case_studies_report.md` chứa 4-6 ca mổ xẻ chi tiết.
- [ ] Slide minh họa Case Studies sẵn sàng cho buổi bảo vệ.
```

---

### Issue #8: [GĐ 6] Hoàn thiện Báo cáo Cuối kỳ chuẩn IMRaD & Slide Thuyết trình

```markdown
## Tiêu đề Issue:
[Documentation] [Giai đoạn 6] Biên soạn Báo cáo cuối kỳ chuẩn IMRaD, Slide thuyết trình & Đóng gói Repo

## Người phụ trách (Assignee):
@BichPhuong (23521239 - Bùi Phạm Bích Phương) & Toàn bộ thành viên

## Người duyệt (Reviewer):
@ThienPhu (23521188 - Trần Thiên Phú - Nhóm trưởng)

## Nhãn (Labels):
`giai-doan-6`, `documentation`, `report`, `presentation`, `priority: high`

## Milestone:
Milestone 5: Final Submission & Defense (Tuần 12)

---

### 📖 DESCRIPTION (MÔ TẢ CHI TIẾT KỸ THUẬT):

#### 1. Bối cảnh & Mục tiêu:
Tổng hợp toàn bộ thành quả nghiên cứu từ Tuần 1 đến Tuần 11 thành một bộ hồ sơ nghiệm thu hoàn chỉnh đạt chất lượng xuất sắc nộp cho Giảng viên hướng dẫn TS. Trần Hưng Nghiệp.

#### 2. Cấu trúc Báo cáo Cuối kỳ (Chuẩn IMRaD khoa học):
- **1. Introduction (Mở đầu):** Bối cảnh lan truyền tin đồn trên MXH, bài toán PHEME, hạn chế của TTT gốc, Research Gap, 3 Research Questions (RQ1-RQ3) và 3 Hypotheses (H1-H3). *(Phụ trách: Phú, Mai)*
- **2. Related Work (Tổng quan nghiên cứu):** Đánh giá TTT 2025, SARD-Net 2026, KG Rumour 2024 theo đúng đề xuất Mini Proposal. *(Phụ trách: Mai)*
- **3. Proposed Method (Phương pháp đề xuất):** Mô tả kiến trúc UC-TTT, công thức toán học Gated Attention Fusion, mã hóa 12 đặc trưng uy tín người dùng, Tree Transformer và Temporal GRU. *(Phụ trách: Phú)*
- **4. Experiments & Results (Thực nghiệm & Kết quả):** Giao thức 9-fold LOEO, thống kê dữ liệu PHEME, bảng so sánh đối chứng 3 Baselines vs UC-TTT (trả lời RQ1), kiểm định t-test ($p < 0.05$). *(Phụ trách: Phú, Đạt, Phương)*
- **5. Ablation Studies & Discussion (Bóc tách & Thảo luận):** Bảng số liệu Feature-level & Model-level ablation (trả lời RQ2), Event-level analysis (trả lời RQ3), Error Analysis & Case Studies. *(Phụ trách: Đạt, Mai, Phương)*
- **6. Conclusion & Future Work (Kết luận):** Tóm lược đóng góp và hướng phát triển. *(Phụ trách: Cả nhóm)*

#### 3. Bộ Slide Thuyết trình (Presentation):
- Thiết kế hiện đại, tinh gọn (khoảng 20-25 slides).
- Có sơ đồ kiến trúc UC-TTT chuyên nghiệp, biểu đồ cột so sánh F1 trực quan, bảng số liệu rõ ràng và 2 case studies thực tế.

#### 4. Đóng gói Mã nguồn GitHub (Reproducibility):
- Rà soát code sạch đẹp, chuẩn PEP8, có docstrings đầy đủ.
- Hoàn thiện `README.md` với hướng dẫn cài đặt và chạy thử nghiệm bằng 1 lệnh duy nhất.
- Đảm bảo seed cố định 42 cho tính tái lập 100%.

#### 5. Sản phẩm bàn giao (Deliverables & DoD):
- [ ] Báo cáo cuối kỳ định dạng PDF và file Word hoàn chỉnh.
- [ ] Bộ Slide thuyết trình (.pptx / PDF) sẵn sàng báo cáo.
- [ ] GitHub Repository sạch đẹp, commit lịch sử chỉn chu, tag release `v1.0-final`.
```

---

## 4. QUY CHUẨN THỰC HIỆN & ĐỊNH NGHĨA HOÀN THÀNH (DEFINITION OF DONE)

Mọi thành viên trong nhóm khi thực hiện và đóng (close) Issue cần tuân thủ nghiêm ngặt các quy chuẩn sau:

1. **Tuân thủ Giao thức Đánh giá (Evaluation Protocol):**
   - Toàn bộ thực nghiệm deep learning bắt buộc dùng thể thức **9-Fold Leave-One-Event-Out (LOEO)**. Tuyệt đối không xáo trộn ngẫu nhiên dữ liệu để tránh data leakage giữa các sự kiện.
2. **Đảm bảo tính tái lập 100% (Reproducibility):**
   - Mọi script phải khởi tạo random seed: `torch.manual_seed(42)`, `np.random.seed(42)`, `torch.cuda.manual_seed_all(42)`.
3. **Quy chuẩn Git Flow & Pull Request:**
   - Không commit trực tiếp vào `main`. Mỗi Issue phải tạo nhánh riêng theo format:
     - `feat/issue-1-text-embeddings`
     - `feat/issue-2-tree-dataloader`
     - `exp/issue-5-ablation-studies`
   - Mỗi Pull Request phải được ít nhất 1 thành viên khác review trước khi merge vào `main`.
4. **Bảo vệ tài nguyên lưu trữ:**
   - Kiểm tra kỹ `.gitignore` trước khi push. Không bao giờ commit các file trọng số `.pth`, file cache `.pkl`, `.pt` vượt quá 50MB lên GitHub commit history.

<a id="quy-trinh-releases"></a>
### 4.4. QUY TRÌNH CHIA SẺ & TẢI FILE NẶNG (>100MB) BẰNG GITHUB RELEASES

Vì GitHub chặn commit các file dữ liệu nặng (`pheme_processed.pkl` ~181MB, `text_embeddings.pt` ~170MB, các checkpoint `.pth` ~150MB), toàn bộ thành viên trong nhóm thống nhất sử dụng tính năng **GitHub Releases** (hỗ trợ đính kèm binary files lên đến 2GB/file) theo quy trình chuẩn sau:

#### A. Dành cho người tạo file (Người phụ trách task - ví dụ: @GiaHuy hoặc @ThienPhu):
1. **Truy cập GitHub Repo:**
   - Mở trình duyệt vào repo dự án: `https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection`.
2. **Tạo Release mới:**
   - Ở thanh bên phải giao diện chính, tìm mục **Releases** $\rightarrow$ bấm **Draft a new release** (hoặc truy cập trực tiếp `.../releases/new`).
3. **Điền thông tin Release:**
   - **Choose a tag:** Nhập tag mới, ví dụ:
     - Dành cho cache dữ liệu: `v0.1-data-cache`
     - Dành cho model checkpoints: `v0.2-model-checkpoints`
   - Bấm **Create new tag: v0.1-data-cache on publish**.
   - **Release title:** Nhập tiêu đề rõ ràng, ví dụ: `PHEME Processed Cache & Offline Text Embeddings`.
   - **Describe this release:** Ghi chú nội dung ngắn (ví dụ: `Gồm file text_embeddings.pt sinh bởi sentence-transformers/all-MiniLM-L6-v2 và pheme_processed.pkl`).
4. **Đính kèm file (Upload binaries):**
   - Kéo và thả file `data/processed/text_embeddings.pt` (hoặc nén `.zip` nếu muốn) vào khung **"Attach binaries by dropping them here or selecting them"**.
   - Chờ thanh upload chạy xong 100%.
5. **Xuất bản:**
   - Bấm nút xanh **Publish release**.

---

#### B. Dành cho các thành viên khác lấy file về máy:

##### Cách 1: Tải trực tiếp bằng giao diện Web (Dễ nhất)
1. Vào mục **Releases** của repo trên trình duyệt: `https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection/releases`.
2. Chọn bản release `v0.1-data-cache`.
3. Trong mục **Assets**, bấm chuột trái vào file `text_embeddings.pt` để tải về máy.
4. Di chuyển file vừa tải vào đúng thư mục `data/processed/` trong project của bạn.

##### Cách 2: Tải nhanh bằng dòng lệnh PowerShell / Command Prompt (Khuyên dùng)
Chạy lệnh sau ngay tại thư mục gốc của project:
```powershell
# PowerShell (Windows)
Invoke-WebRequest -Uri "https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection/releases/download/v0.1-data-cache/text_embeddings.pt" -OutFile "data/processed/text_embeddings.pt"
```

Hoặc nếu dùng `curl`:
```bash
curl -L -o data/processed/text_embeddings.pt https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection/releases/download/v0.1-data-cache/text_embeddings.pt
```

##### Cách 3: Tự sinh lại cục bộ (Dành cho thành viên có GPU khỏe hoặc khi nộp bài)
Vì code sinh embeddings đã cố định `seed=42`, bạn chỉ cần chạy lệnh sau để máy tự sinh ra file giống hệt 100%:
```bash
python experiments/extract_text_embeddings.py
```
