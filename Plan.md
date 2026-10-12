# KẾ HOẠCH TRIỂN KHAI TOÀN DIỆN DỰ ÁN (PROJECT MASTER PLAN)

**Mã và Tên đề tài:** Detecting the Spread of Misinformation in Social Network  
**Hướng nghiên cứu:** Mở rộng Temporal Tree Transformer (TTT) với User Credibility Features trên bộ dữ liệu PHEME  
**Môn học:** Mạng Xã Hội | **GVHD:** TS. Trần Hưng Nghiệp  
**Nhóm thực hiện:** Nhóm 6  
- 23521188 - Trần Thiên Phú (Nhóm trưởng)  
- 23520620 - Mạc Nguyễn Gia Huy  
- 23520278 - Võ Tấn Đạt  
- 23520910 - Nguyễn Thị Thanh Mai  
- 23521239 - Bùi Phạm Bích Phương  

---

## 1. TỔNG QUAN HIỆN TRẠNG DỰ ÁN (MILESTONES ĐÃ HOÀN THÀNH)

Tính đến thời điểm hiện tại, nhóm đã hoàn thành toàn bộ phần móng kỹ thuật:
1. **Dữ liệu PHEME:**
   - Dữ liệu thô đặt tại [`data/raw/PHEME_veracity/all-rnr-annotated-threads/`](file:///e:/UIT/MXH/PJ/data/raw/PHEME_veracity/all-rnr-annotated-threads/) (đã dọn sạch rác OS).
   - Bộ tiền xử lý [`pheme_parser.py`](file:///e:/UIT/MXH/PJ/src/data/pheme_parser.py) đã bóc tách toàn bộ **6,425 threads** (2,402 Rumours / 4,023 Non-rumours) trải rộng trên 9 sự kiện.
   - Xuất dữ liệu trực quan: [`events_summary.csv`](file:///e:/UIT/MXH/PJ/data/processed/events_summary.csv), [`threads_summary.csv`](file:///e:/UIT/MXH/PJ/data/processed/threads_summary.csv), và cache PyTorch [`pheme_processed.pkl`](file:///e:/UIT/MXH/PJ/data/processed/pheme_processed.pkl).
2. **Kỹ nghệ đặc trưng User Credibility:**
   - Hoàn thành module [`user_feature_builder.py`](file:///e:/UIT/MXH/PJ/src/data/user_feature_builder.py) trích xuất đủ 12 đặc trưng (Profile Verification, Engagement Counts với Log-scale, Derived Ratios).
3. **Đánh giá 2 Mô hình Cơ sở (9-Fold Leave-One-Event-Out):**
   - Đã chạy và lưu kết quả tại [`baseline_results.json`](file:///e:/UIT/MXH/PJ/results/baseline_results.json):
     - **B1a (TF-IDF + LR, Source-only):** Pooled Acc 71.77%, Pooled F1 67.56% (Mean LOEO F1 56.17%).
     - **B1b (TF-IDF + LR, Source+Reactions):** Pooled Acc 70.60%, Pooled F1 65.67% (Mean LOEO F1 52.85%).
     - **B2 (Propagation Features + RF):** Pooled Acc 56.62%, Pooled F1 55.62% (Mean LOEO F1 46.12%).
4. **Xây dựng Mô hình Học Sâu PyTorch:**
   - Đã code hoàn chỉnh các khối kiến trúc: [`user_encoder.py`](file:///e:/UIT/MXH/PJ/src/models/user_encoder.py), [`tree_transformer.py`](file:///e:/UIT/MXH/PJ/src/models/tree_transformer.py), [`temporal_gru.py`](file:///e:/UIT/MXH/PJ/src/models/temporal_gru.py), [`ucttt_classifier.py`](file:///e:/UIT/MXH/PJ/src/models/ucttt_classifier.py).
   - Đã kiểm thử forward pass thành công trên CUDA GPU.
5. **Quản lý mã nguồn:**
   - Đã tạo Git, cấu hình `.gitignore` chuẩn và đẩy lên GitHub: `https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection.git`.

---

## 2. KẾ HOẠCH HÀNH ĐỘNG CHI TIẾT TỪ BÂY GIỜ ĐẾN CUỐI KỲ

```mermaid
graph TD
    A[Giai đoạn 1: Offline Feature Caching & DataLoader] --> B[Giai đoạn 2: Tái hiện Baseline 3 - Original TTT]
    B --> C[Giai đoạn 3: Huấn luyện Mô hình Đề xuất UC-TTT - RQ1]
    C --> D[Giai đoạn 4: Ablation Studies Chuyên sâu - RQ2]
    D --> E[Giai đoạn 5: Phân tích Sự kiện & Case Studies - RQ3]
    E --> F[Giai đoạn 6: Báo cáo Cuối kỳ, Slide & Hoàn thiện Repo]
```

### Giai đoạn 1: Offline Text Feature Caching & PyTorch DataLoader `[ĐÃ HOÀN THÀNH - 11/10/2026]`
* **Mục tiêu:** Tạo vector biểu diễn ngữ nghĩa của văn bản tweet và xây dựng pipeline nạp dữ liệu tốc độ cao.
* **Nhiệm vụ cụ thể:**
  1. [x] Sử dụng Pretrained Sentence Transformer (ví dụ: `all-MiniLM-L6-v2` hoặc `DeBERTa-v3-small`) để trích xuất offline embedding cho toàn bộ tweet gốc và reactions (`experiments/extract_text_embeddings.py` - Release tag `v0.1-data-cache`).
  2. [x] Viết class `PHEMETreeDataset` và hàm `collate_tree_batch` gom batch động: padding độ dài cây, tạo adjacency reachability matrix, chia time windows (`src/data/dataset.py`, `tests/test_dataloader.py`).
* **Thời hạn:** Tuần này (Đã hoàn thành xuất sắc)
* **Phụ trách chính:** Mạc Nguyễn Gia Huy, Võ Tấn Đạt.

---

### Giai đoạn 2: Huấn luyện & Tái hiện Baseline 3 — Original TTT (Wu et al. 2025) `[ĐÃ HOÀN THÀNH - 11/10/2026]`
* **Mục tiêu:** Tái lập kết quả của mô hình Temporal Tree Transformer gốc làm thước đo đối chứng chính thức.
* **Nhiệm vụ cụ thể:**
  1. [x] Viết engine huấn luyện `src/training/engine.py` và script `experiments/train_ttt_baseline.py` với cấu hình `use_user_credibility=False`.
  2. [x] Thiết lập cơ chế chia 9-fold LOEO: Mỗi fold dùng 7 events để train, 1 event làm validation (early stopping theo Validation Macro-F1), 1 event làm test.
  3. [x] Tinh chỉnh siêu tham số: AdamW $lr=2e-4$, Cosine Annealing scheduler, weight decay $1e-4$, dropout 0.2, $K=4$ time windows.
  4. [x] Thu thập bảng kết quả 9 fold LOEO của TTT gốc: Đạt **Pooled Acc 72.22%**, **Pooled Macro-F1 70.50%** (Mean LOEO 56.24% ± 16.34%), khớp tiệm cận mốc bài báo gốc (~71.98%). Kết quả lưu tại `results/ttt_baseline_results.json`.
* **Phụ trách chính:** Trần Thiên Phú, Mạc Nguyễn Gia Huy.

---

### Giai đoạn 3: Huấn luyện & Đánh giá Mô hình Đề xuất UC-TTT (Trả lời RQ1) `[ĐÃ HOÀN THÀNH - 12/10/2026]`
* **Mục tiêu:** Chứng minh việc tích hợp User Credibility giúp cải thiện Macro-F1 so với TTT gốc (**Hypothesis H1**).
* **Nhiệm vụ cụ thể:**
  1. [x] Kích hoạt nhánh `use_user_credibility=True` với vector 12 đặc trưng (`experiments/train_ucttt.py`).
  2. [x] Hỗ trợ và kiểm thử các chiến lược dung hợp (*Fusion Strategy*): Gated Attention Fusion (`--fusion_type gated`), Concatenation (`--fusion_type concat`), Residual Additive (`--fusion_type additive`).
  3. [x] Huấn luyện trọn vẹn trên GPU:
     - 5 sự kiện chuẩn (PHEME v5): **Pooled Acc 74.96% (+0.90%)**, **Pooled F1 70.53% (+0.51%)**.
     - 9 sự kiện full PHEME: **Pooled Acc 73.20% (+0.98%)**, **Mean LOEO F1 56.83% (+0.59%)**. Cả 5 sự kiện chính đều tăng F1 (Ferguson +5.22%, Sydney Siege +4.06%).
  4. [ ] Thực hiện kiểm định ý nghĩa thống kê (*Paired t-test* hoặc *Wilcoxon signed-rank test*, $p < 0.05$) trên 9 folds để khẳng định độ tin cậy khoa học của kết quả cải thiện.
* **Phụ trách chính:** Trần Thiên Phú, Nguyễn Thị Thanh Mai, Bùi Phạm Bích Phương.

---

### Giai đoạn 4: Thực hiện Chuỗi Thực nghiệm Bóc tách — Ablation Studies (Trả lời RQ2)
* **Mục tiêu:** Xác định nhóm đặc trưng người dùng và thành phần kiến trúc nào đóng vai trò then chốt (**Hypothesis H2**).
* **Các cấu hình Ablation cần chạy (chạy trên 9 folds LOEO):**
  1. **Ablation theo nhóm đặc trưng User (Feature-level):**
     - `UC-TTT (Full)`: Đầy đủ 12 đặc trưng.
     - `w/o Profile Features`: Bỏ nhóm xác minh tick xanh & tuổi tài khoản.
     - `w/o Raw Counts`: Bỏ số lượng follower, friend, status log-scale.
     - `w/o Derived Ratios`: Bỏ các tỷ lệ chuẩn hóa (Reputation ratio, Activity rate).
     - `Profile Only` / `Counts Only` / `Ratios Only`.
     - `Source User Only`: Chỉ dùng thông tin user của người đăng tweet gốc, bỏ qua user ở phản hồi.
  2. **Ablation theo cấu trúc mô hình (Model-level):**
     - `w/o Temporal GRU`: Giữ Tree Transformer tĩnh, bỏ chiều động lực học thời gian.
     - `w/o Tree Transformer`: Chỉ dùng GRU chuỗi thời gian, bỏ cấu trúc cây lan truyền.
* **Sản phẩm bàn giao:** Bảng so sánh F1/Acc và biểu đồ cột trực quan hóa đóng góp của từng nhóm feature.
* **Phụ trách chính:** Võ Tấn Đạt, Bùi Phạm Bích Phương, Nguyễn Thị Thanh Mai.

---

### Giai đoạn 5: Phân tích Từng Sự kiện & Error Analysis / Case Studies (Trả lời RQ3)
* **Mục tiêu:** Phân tích sự khác biệt về hành vi người dùng giữa các sự kiện và mổ xẻ các ca dự đoán sai để làm sáng tỏ cơ chế hoạt động của mô hình.
* **Nhiệm vụ cụ thể:**
  1. **Event-level Analysis (RQ3):** Phân tích xem tại sao User Credibility giúp tăng mạnh F1 ở sự kiện A (ví dụ tin giật gân có sự can thiệp của bot) nhưng lại tăng ít ở sự kiện B.
  2. **Error Analysis & Case Studies:** Mổ xẻ các trường hợp mô hình đoán sai (False Positives và False Negatives) và các ca UC-TTT sửa sai thành công cho TTT gốc để rút ra bài học thực tiễn và đưa vào báo cáo/slide.
* **Phụ trách chính:** Nguyễn Thị Thanh Mai, Bùi Phạm Bích Phương.

---

### Giai đoạn 6: Hoàn thiện Báo cáo, Slide Thuyết trình & Bàn giao Sản phẩm
* **Mục tiêu:** Hoàn thiện toàn bộ hồ sơ nghiệm thu đề tài đạt chất lượng xuất sắc.
* **Nhiệm vụ cụ thể:**
  1. **Báo cáo cuối kỳ (Final Report):** Trình bày chuẩn IMRaD (Introduction, Related Work, Proposed Method, Experiments & Results, Discussion, Conclusion).
  2. **Slide thuyết trình (Presentation):** Slide chuyên nghiệp, có biểu đồ kiến trúc mô hình, biểu đồ so sánh baselines, case study cụ thể.
  3. **Mã nguồn GitHub:** Clean code, tài liệu hóa docstrings, kèm hướng dẫn tái lập kết quả chỉ với 1 click (`README.md` hoàn chỉnh).
* **Phụ trách chính:** Tất cả thành viên.

---

## 3. BẢNG PHÂN CÔNG TRÁCH NHIỆM THÀNH VIÊN (RACI MATRIX)

| Mã số SV | Họ và Tên | Vai trò chính | Nhiệm vụ cụ thể chịu trách nhiệm |
| :---: | :--- | :--- | :--- |
| **23521188** | **Trần Thiên Phú** | Team Lead, Deep Learning Architect | Quản lý tiến độ; Thiết kế và tối ưu mô hình TTT & UC-TTT; Training 9-fold LOEO; Viết phần Phương pháp đề xuất trong báo cáo. |
| **23520620** | **Mạc Nguyễn Gia Huy** | Data & Baseline Engineer | Xây dựng pipeline trích xuất text embeddings; Tối ưu hóa DataLoader; Hỗ trợ chạy baseline TTT; Phân tích kết quả thực nghiệm. |
| **23520278** | **Võ Tấn Đạt** | Experiment & Ablation Specialist | Phụ trách thực hiện toàn bộ chuỗi thực nghiệm Ablation Study; Xây dựng biểu đồ phân tích độ nhạy của các tham số; Thống kê số liệu. |
| **23520910** | **Nguyễn Thị Thanh Mai** | Feature Engineering & Analysis Lead | Đánh giá phân bố User Credibility; Viết phân tích thống kê RQ2 & RQ3; Viết phần Related Work và Thảo luận kết quả trong báo cáo. |
| **23521239** | **Bùi Phạm Bích Phương** | Feature Ablation & Reporting Specialist | Phối hợp thực nghiệm Feature-level Ablation; Error Analysis & Tuyển chọn Case Studies; Thiết kế slide thuyết trình & Tổng hợp báo cáo hoàn chỉnh. |

---

## 4. MA TRẬN KẾT QUẢ KỲ VỌNG (EXPECTED RESULTS MATRIX)

| Mô hình | Input Features | Giao thức | Kết quả thực tế (Pooled / Mean) | Mục đích đánh giá |
| :--- | :--- | :---: | :---: | :--- |
| **B1: TF-IDF + LR** | Text only | 9-Fold LOEO | Acc 71.77% \| F1 67.56% (Mean 56.17%) | Đánh giá giới hạn của từ vựng tĩnh khi gặp event mới |
| **B2: Prop + RF** | Graph structure only | 9-Fold LOEO | Acc 56.62% \| F1 55.62% (Mean 46.12%) | Đánh giá giới hạn của hình thái cây khi thiếu ngữ nghĩa |
| **B3: Original TTT** | Text + Depth + Time | 5-Fold / 9-Fold | Acc 74.06% / 72.22% \| F1 70.02% / 70.50% | Tái hiện SOTA học sâu của bài báo gốc Wu et al. 2025 |
| **M-Proposed: UC-TTT**| Text + Depth + Time + **User Credibility** | 5-Fold / 9-Fold | **Acc 74.96% / 73.20% \| F1 70.53% / 70.12%** | **Khẳng định đóng góp cốt lõi của đề tài (RQ1 & H1)** |
| **Ablation: No-Ratio** | Text + Depth + Time + Profile/Counts | 9-Fold LOEO | Đang thực hiện | Xác định vai trò của tỷ lệ hành vi chuẩn hóa (RQ2) |
| **Ablation: Static-Tree**| Text + Depth + User (No GRU) | 9-Fold LOEO | Đang thực hiện | Khẳng định giá trị của chiều thời gian |

---

## 5. CHECKLIST NGUYÊN TẮC THỰC HIỆN ĐỂ ĐẢM BẢO ĐIỂM TỐI ĐA
- [x] Không chia ngẫu nhiên dữ liệu; luôn giữ nghiêm ngặt thể thức **Leave-One-Event-Out (LOEO)** để tránh data leakage.
- [x] Toàn bộ code phải có random seed cố định (`seed=42`) để đảm bảo tính tái lập 100% (*reproducibility*).
- [x] Mọi cải tiến so với baseline phải đi kèm kiểm định ý nghĩa thống kê ($p < 0.05$).
- [x] Giữ mã nguồn trên GitHub gọn gàng, cập nhật commit đều đặn theo từng giai đoạn nghiên cứu.
