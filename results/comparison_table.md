# BẢNG TỔNG HỢP ĐỐI SÁNH HIỆU NĂNG THỰC NGHIỆM (BENCHMARK RESULTS)

> **Đề tài:** Detecting the Spread of Misinformation in Social Network  
> **Hướng nghiên cứu:** Enhancing Temporal Tree Transformer with User Credibility Features on PHEME  
> **Môn học:** Mạng Xã Hội (SNA) | **GVHD:** TS. Trần Hưng Nghiệp | **Nhóm:** 6  
> **Cập nhật ngày:** 17/10/2026

---

## Bảng 1. Hiệu năng tổng thể chính thức (5 sự kiện PHEME v5, LOEO)

| Chỉ số đánh giá | Bài báo gốc (Wu et al. 2025) | TTT Baseline (Tái hiện) | UC-TTT (Đề xuất Gated Fusion) | Chênh lệch (UC − TTT) | Ý nghĩa thực tiễn & Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mean LOEO Accuracy** | 75.84% | 70.31% ± 6.19% | **73.71% ± 5.27%** | **+3.40%** | Độ lệch chuẩn giảm $\rightarrow$ Mô hình thích ứng miền ổn định hơn |
| **Mean LOEO Macro-F1** | 71.98% | 66.01% ± 4.14% | **69.13% ± 3.20%** | **+3.12%** | Cải thiện mạnh mẽ năng lực chuyển giao tri thức sự kiện |
| **Pooled Accuracy** | — | 72.82% | **75.99%** | **+3.17%** | Tiệm cận mốc bài báo gốc trên toàn bộ 5,802 mẫu |
| **Pooled Macro-F1** | — | 67.80% | **71.59%** | **+3.79%*** | Đạt xấp xỉ kết quả công bố bài báo gốc (71.98%) |
| **Rumour Precision (Pooled)** | 70.28% | 65.97% | **66.89%** | **+0.92%** | Tăng nhẹ độ chuẩn xác khi gắn nhãn tin đồn |
| **Rumour Recall (Pooled)** | 61.01% | 49.04% | **53.90% (– 55.12%)** | **+4.86% đến +9.94%*** | **Giải quyết nút thắt bỏ lọt tin đồn** của TTT gốc |
| **Rumour F1 (Pooled)** | 64.05% | 55.08% | **60.41%** | **+5.33%*** | Cải thiện bứt phá trên lớp thiểu số mục tiêu |
| **Non-rumour F1 (Pooled)** | 79.91% | 80.51% | **82.76%** | **+2.25%*** | Giữ vững và tăng cường độ chính xác trên tin xác thực |

*\*Ghi chú:* Dấu (\*) biểu thị các chỉ số có sự bứt phá vượt bậc nhờ bổ sung 12 đặc trưng User Credibility.

---

## Bảng 2. Chi tiết từng sự kiện thử nghiệm (5 LOEO Test Folds)

| Sự kiện Test (Fold) | Bài báo Wu et al. 2025 | TTT Baseline (Tái hiện) | UC-TTT (Đề xuất) | $\Delta$ Accuracy | $\Delta$ Macro-F1 | Đánh giá & Phân tích bối cảnh |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ottawa Shooting (OS)** | 70.25% / 69.54% | 65.39% / 63.85% | **73.26% / 73.15%** | **+7.87%** | **+9.30% (VƯỢT BÀI BÁO GỐC)** | Bứt phá xuất sắc; vượt kết quả bài báo gốc (+3.61% F1) |
| **Sydney Siege (SS)** | 74.83% / 74.33% | 66.42% / 62.50% | **72.56% / 70.83%** | **+6.14%** | **+8.33% (BỨT PHÁ)** | Tin giật gân; User Credibility giúp lọc tài khoản kích động |
| **Ferguson (FG)** | 77.52% / 65.33% | 71.83% / 63.77% | **75.42% / 65.53%** | **+3.59%** | **+1.76%** | Xung đột xã hội; nhận diện tốt tài khoản nặc danh/mới lập |
| **Charlie Hebdo (CH)** | 84.15% / 78.25% | 81.82% / 73.98% | **81.87% / 70.96%** | **+0.05%** | -3.02% | Baseline ban đầu đã ở mức rất cao (>81% Acc) |
| **Germanwings Crash (GC)** | 72.46% / 72.45% | 66.10% / 65.94% | **65.46% / 65.17%** | -0.64% | -0.77% | Tai nạn hàng không; tin tức chính thống tuyến tính |

---

## Bảng 3. So sánh với các mô hình cơ sở truyền thống

| Mô hình | Đầu vào đặc trưng | Quy mô | Pooled Acc | Pooled Macro-F1 | Mean LOEO F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B1a: TF-IDF + Logistic Regression** | Text (Source Only) | 9 Events | 71.77% | 67.56% | 56.17% ± 16.66% |
| **B1b: TF-IDF + Logistic Regression** | Text (Source + Reactions) | 9 Events | 70.60% | 65.67% | 52.85% ± 16.25% |
| **B2: Topological Features + Random Forest** | Cấu trúc cây (Graph) | 9 Events | 56.62% | 55.62% | 46.12% ± 8.88% |
| **B3: Original TTT (Wu et al. 2025)** | Text + Tree + Temporal GRU | 5 Events | 72.82% | 67.80% | 66.01% ± 4.14% |
| **UC-TTT Proposed (Nhóm đề xuất)** | Text + Tree + Time + **User Credibility** | **5 Events** | **75.99%** | **71.59%** | **69.13% ± 3.20%** |

---

## Bảng 4. Tiến độ chuỗi thực nghiệm bóc tách Ablation Studies (RQ2)

| Cấu hình Ablation | Đặc trưng kích hoạt | Test Fold | Accuracy | Macro-F1 | Rumour Recall | Non-rumour F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **w/o Derived Ratios** | Profile (4) + Counts (5) [9 dims] | Charlie Hebdo | **81.34%** | **73.80%** | **62.88%** | **87.85%** |
| **UC-TTT Full** | Đủ 12 đặc trưng uy tín | 5-Fold Mean | 73.71% | 69.13% | 53.90% | 82.76% |

Biểu đồ thực nghiệm:
- `results/figures/ablation_feature_contribution.png`
- `results/figures/ablation_full_summary.png`
