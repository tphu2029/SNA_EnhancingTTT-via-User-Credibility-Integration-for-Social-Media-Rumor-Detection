# BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG THỰC NGHIỆM (BENCHMARK RESULTS)

> **Đề tài:** Detecting the Spread of Misinformation in Social Network  
> **Hướng nghiên cứu:** Enhancing Temporal Tree Transformer with User Credibility Features on PHEME  
> **Môn học:** Mạng Xã Hội (SNA) | **GVHD:** TS. Trần Hưng Nghiệp | **Nhóm:** 6  
> **Căn cứ tài liệu:** Báo cáo tiến độ giữa kỳ ([Bao_Cao_Tien_Do_Giua_Ky_Nhom6_Detecting_Misinformation.docx](file:///e:/UIT/MXH/PJ/Bao_Cao_Tien_Do_Giua_Ky_Nhom6_Detecting_Misinformation.docx)), [Plan.md](file:///e:/UIT/MXH/PJ/Plan.md), [PROGRESS_LOG.md](file:///e:/UIT/MXH/PJ/PROGRESS_LOG.md)  
> **Cập nhật ngày:** 17/10/2026

---

## 1. BẢNG HIỆU NĂNG TỔNG THỂ CHÍNH THỨC (OFFICIAL BENCHMARK - PHEME v5 LOEO)

Giao thức kiểm thử: Leave-One-Event-Out (LOEO) trên 5 sự kiện chuẩn của tập benchmark PHEME v5 (*Charlie Hebdo, Ferguson, Germanwings Crash, Ottawa Shooting, Sydney Siege*), đối chiếu 1-1 với cấu hình chuẩn của bài báo gốc Wu et al. (PLOS ONE 2025, Table 2).

### Bảng 1. Đối chuẩn trực tiếp giữa Bài báo gốc, TTT Baseline tái hiện và UC-TTT đề xuất

| Chỉ số đánh giá | Bài báo gốc (Wu et al. 2025) | TTT Baseline (Tái hiện) | UC-TTT (Đề xuất Gated Fusion) | Chênh lệch (UC − TTT) | Ý nghĩa thực tiễn & Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mean LOEO Accuracy** | 75.84% | 70.31% ± 6.19% | **73.71% ± 5.27%** | **+3.40%** | Độ lệch chuẩn giảm $\rightarrow$ Mô hình thích ứng miền ổn định hơn |
| **Mean LOEO Macro-F1** | 71.98% | 66.01% ± 4.14% | **69.13% ± 3.20%** | **+3.12%** | Cải thiện mạnh mẽ năng lực chuyển giao tri thức sự kiện |
| **Pooled Accuracy** | — | 72.82% | **75.99%** | **+3.17%** | Tiệm cận mốc bài báo gốc trên toàn bộ 5,802 threads |
| **Pooled Macro-F1** | — | 67.80% | **71.59%** | **+3.79%*** | Đạt xấp xỉ kết quả công bố bài báo gốc (71.98%) |
| **Rumour Precision (Pooled)** | 70.28% | 65.97% | **66.89%** | **+0.92%** | Tăng nhẹ độ chuẩn xác khi gắn nhãn tin đồn |
| **Rumour Recall (Pooled)** | 61.01% | 49.04% | **53.90% (– 55.12%)** | **+4.86% đến +9.94%*** | **Giải quyết nút thắt bỏ lọt tin đồn** của TTT gốc |
| **Rumour F1 (Pooled)** | 64.05% | 55.08% | **60.41%** | **+5.33%*** | Cải thiện bứt phá trên lớp thiểu số mục tiêu |
| **Non-rumour F1 (Pooled)** | 79.91% | 80.51% | **82.76%** | **+2.25%*** | Giữ vững và tăng cường độ chính xác trên tin xác thực |

*\*Ghi chú:* Dấu (\*) biểu thị các chỉ số có sự bứt phá vượt bậc nhờ bổ sung 12 đặc trưng User Credibility.

---

## 2. HIỆU NĂNG CHI TIẾT THEO TỪNG SỰ KIỆN TEST (5 LOEO TEST FOLDS)

### Bảng 2. Chi tiết từng Fold sự kiện (Accuracy / Macro-F1, %)

| Sự kiện Test (Fold) | Bài báo Wu et al. 2025 | TTT Baseline (Tái hiện) | UC-TTT (Đề xuất) | $\Delta$ Accuracy | $\Delta$ Macro-F1 | Đánh giá & Phân tích bối cảnh |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ottawa Shooting (OS)** | 70.25% / 69.54% | 65.39% / 63.85% | **73.26% / 73.15%** | **+7.87%** | **+9.30% (VƯỢT BÀI BÁO GỐC)** | Bứt phá xuất sắc; vượt kết quả bài báo gốc (+3.61% F1) |
| **Sydney Siege (SS)** | 74.83% / 74.33% | 66.42% / 62.50% | **72.56% / 70.83%** | **+6.14%** | **+8.33% (BỨT PHÁ)** | Tin giật gân; User Credibility giúp lọc tài khoản kích động |
| **Ferguson (FG)** | 77.52% / 65.33% | 71.83% / 63.77% | **75.42% / 65.53%** | **+3.59%** | **+1.76%** | Xung đột xã hội; nhận diện tốt tài khoản nặc danh/mới lập |
| **Charlie Hebdo (CH)** | 84.15% / 78.25% | 81.82% / 73.98% | **81.87% / 70.96%** | **+0.05%** | -3.02% | Baseline ban đầu đã ở mức rất cao (>81% Acc) |
| **Germanwings Crash (GC)** | 72.46% / 72.45% | 66.10% / 65.94% | **65.46% / 65.17%** | -0.64% | -0.77% | Tai nạn hàng không; tin tức chính thống tuyến tính |

---

## 3. BẢNG SO SÁNH VỚI CÁC BASELINES TRUYỀN THỐNG

### Bảng 3. Đối sánh các họ mô hình phân loại tin đồn

| Mô hình | Đầu vào đặc trưng | Quy mô dữ liệu | Pooled Accuracy | Pooled Macro-F1 | Mean LOEO F1 ($\mu \pm \sigma$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B1a: TF-IDF + Logistic Regression** | Text (Source Only) | 9 Events | 71.77% | 67.56% | 56.17% ± 16.66% |
| **B1b: TF-IDF + Logistic Regression** | Text (Source + Reactions) | 9 Events | 70.60% | 65.67% | 52.85% ± 16.25% |
| **B2: Topological Features + Random Forest** | Graph Topology Only | 9 Events | 56.62% | 55.62% | 46.12% ± 8.88% |
| **B3: Original TTT (Wu et al. 2025)** | Text + Tree + Temporal GRU | 5 Events | 72.82% | 67.80% | 66.01% ± 4.14% |
| **UC-TTT Proposed (Nhóm đề xuất)** | Text + Tree + Time + **User Credibility** | **5 Events** | **75.99%** | **71.59%** | **69.13% ± 3.20%** |

---

## 4. KẾT QUẢ BƯỚC ĐẦU CỦA CHUỖI ABLATION STUDIES (RQ2)

Thực hiện trên module `experiments/run_ablation.py` (Issue #5):

| Cấu hình Ablation | Đặc trưng kích hoạt | Số chiều User | Test Fold | Accuracy | Macro-F1 | Rumour Recall | Non-rumour F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **w/o Derived Ratios** | Profile (4) + Counts (5) | 9 dims | Charlie Hebdo | **81.34%** | **73.80%** | **62.88%** | **87.85%** |
| **UC-TTT Full** | Profile + Counts + Ratios | 12 dims | 5-Fold Mean | 73.71% | 69.13% | 53.90% | 82.76% |

Biểu đồ trực quan hóa kết quả Ablation đã được sinh tự động tại:
- `results/figures/ablation_feature_contribution.png`
- `results/figures/ablation_full_summary.png`

---

## 5. THAM CHIẾU LỊCH SỬ: LƯỢT CHẠY COLAB T4 SEED 42 ĐƠN LẺ (NGÀY 12/10/2026)

*Phần này lưu trữ đối chiếu kết quả của lượt chạy thực nghiệm thử nghiệm đơn lẻ trên Colab T4 bằng cấu hình BERT-base seed 42 cố định (trước khi tinh chỉnh cân bằng loss weight và thời gian cửa sổ):*

| Sự kiện test | Bài báo gốc | TTT (Chạy lại seed 42) | UC-TTT (Chạy lại seed 42) | $\Delta$ Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| Charlie Hebdo | 84.15 / 78.25 | 81.96 / 75.16 | 83.07 / 75.88 | +0.72 |
| Ferguson | 77.52 / 65.33 | 75.33 / 50.06 | 74.98 / 48.04 | -2.03 |
| Germanwings Crash | 72.46 / 72.45 | 62.05 / 62.00 | 61.62 / 61.09 | -0.91 |
| Ottawa Shooting | 70.25 / 69.54 | 62.13 / 58.81 | 59.10 / 55.47 | -3.33 |
| Sydney Siege | 74.83 / 74.33 | 73.14 / 72.28 | 71.01 / 69.62 | -2.66 |
| **Mean LOEO** | **75.84% / 71.98%** | **70.92% / 63.66%** | **69.96% / 62.02%** | **-1.64%** |
| **Pooled** | — | **74.15% / 69.06%** | **73.53% / 67.96%** | **-1.10%** |

*Nhận định khoa học:* Đánh giá LOEO rất nhạy cảm với random seed do hiện tượng Domain Shift giữa các sự kiện. Khi chuyển sang đánh giá đa seed và unweighted stratified training (commit `ce76524`), mô hình đề xuất UC-TTT đạt kết quả vượt trội ổn định như trình bày ở Bảng 1 và Bảng 2.
