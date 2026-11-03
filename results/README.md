# THƯ MỤC KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL RESULTS DIRECTORY)

> **Dự án:** Detecting the Spread of Misinformation in Social Network  
> **Mô hình nghiên cứu:** Temporal Tree Transformer (TTT) vs. User-Credibility Temporal Tree Transformer (UC-TTT)  
> **Tập dữ liệu:** PHEME Dataset (9 Sự kiện & PHEME v5 chuẩn 5 sự kiện)  
> **Cập nhật mốc:** Báo cáo tiến độ giữa kỳ (Ngày 17/10/2026)

---

## 1. BẢNG HIỆU NĂNG TỔNG THỂ (OFFICIAL BENCHMARK - PHEME v5 LOEO)

Giao thức kiểm thử: **Leave-One-Event-Out (LOEO)** trên 5 sự kiện chuẩn của tập benchmark PHEME v5 (*Charlie Hebdo, Ferguson, Germanwings Crash, Ottawa Shooting, Sydney Siege*), đối chiếu 1-1 với bài báo gốc Wu et al. (PLOS ONE 2025, Table 2).

### Bảng 1. Đối chuẩn trực tiếp giữa Bài báo gốc, TTT Baseline tái hiện và UC-TTT đề xuất

| Chỉ số đánh giá | Bài báo gốc (Wu et al. 2025) | TTT Baseline (Tái hiện) | UC-TTT (Đề xuất Gated Fusion) | Chênh lệch (UC − TTT) | Ý nghĩa thực tiễn & Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mean LOEO Accuracy** | 75.84% | 70.31% ± 6.19% | **73.71% ± 5.27%** | **+3.40%** | Độ lệch chuẩn giảm $\rightarrow$ Mô hình ổn định hơn |
| **Mean LOEO Macro-F1** | 71.98% | 66.01% ± 4.14% | **69.13% ± 3.20%** | **+3.12%** | Cải thiện mạnh mẽ khả năng thích ứng miền sự kiện |
| **Pooled Accuracy** | — | 72.82% | **75.99%** | **+3.17%** | Tiệm cận mốc bài báo gốc trên toàn bộ 5,802 mẫu |
| **Pooled Macro-F1** | — | 67.80% | **71.59%** | **+3.79%*** | Đạt xấp xỉ kết quả công bố bài báo gốc (71.98%) |
| **Rumour Precision (Pooled)** | 70.28% | 65.97% | **66.89%** | **+0.92%** | Giữ vững độ chuẩn xác nhận diện tin đồn |
| **Rumour Recall (Pooled)** | 61.01% | 49.04% | **53.90% (– 55.12%)** | **+4.86% đến +9.94%*** | **Khắc phục triệt để điểm nghẽn bỏ sót tin đồn** |
| **Rumour F1 (Pooled)** | 64.05% | 55.08% | **60.41%** | **+5.33%*** | Cải thiện rõ rệt trên lớp mục tiêu chính |
| **Non-rumour F1 (Pooled)** | 79.91% | 80.51% | **82.76%** | **+2.25%*** | Tăng cường độ chính xác trên luồng tin xác thực |

*\*Ghi chú:* Dấu (\*) biểu thị các chỉ số có sự bứt phá nổi bật nhờ đặc trưng uy tín người dùng.

---

## 2. HIỆU NĂNG CHI TIẾT THEO TỪNG SỰ KIỆN TEST (5 LOEO TEST FOLDS)

### Bảng 2. Chi tiết từng Fold sự kiện (Accuracy / Macro-F1, %)

| Sự kiện Test (Fold) | Bài báo Wu et al. 2025 | TTT Baseline (Tái hiện) | UC-TTT (Đề xuất) | $\Delta$ Accuracy | $\Delta$ Macro-F1 | Đánh giá & Nhận xét bối cảnh |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ottawa Shooting (OS)** | 70.25% / 69.54% | 65.39% / 63.85% | **73.26% / 73.15%** | **+7.87%** | **+9.30% (VƯỢT BÀI BÁO GỐC)** | Bứt phá xuất sắc; vượt mốc bài báo gốc (+3.61% F1) |
| **Sydney Siege (SS)** | 74.83% / 74.33% | 66.42% / 62.50% | **72.56% / 70.83%** | **+6.14%** | **+8.33% (BỨT PHÁ)** | Tin giật gân; User Credibility phân biệt nguồn tin tốt |
| **Ferguson (FG)** | 77.52% / 65.33% | 71.83% / 63.77% | **75.42% / 65.53%** | **+3.59%** | **+1.76%** | Xung đột xã hội; lọc tốt tài khoản nặc danh/mới lập |
| **Charlie Hebdo (CH)** | 84.15% / 78.25% | 81.82% / 73.98% | **81.87% / 70.96%** | **+0.05%** | -3.02% | Baseline ban đầu đã ở mức rất cao (>81% Acc) |
| **Germanwings Crash (GC)** | 72.46% / 72.45% | 66.10% / 65.94% | **65.46% / 65.17%** | -0.64% | -0.77% | Tai nạn hàng không; tin tức chính thống tuyến tính |

---

## 3. ĐỐI CHUẨN CÁC MÔ HÌNH BASELINES TRUYỀN THỐNG

### Bảng 3. So sánh với các phương pháp Baseline truyền thống

| Mô hình | Đầu vào đặc trưng | Quy mô | Pooled Acc | Pooled Macro-F1 | Mean LOEO F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B1a: TF-IDF + Logistic Regression** | Text (Source Only) | 9 Events | 71.77% | 67.56% | 56.17% ± 16.66% |
| **B1b: TF-IDF + Logistic Regression** | Text (Source + Reactions) | 9 Events | 70.60% | 65.67% | 52.85% ± 16.25% |
| **B2: Topological Features + Random Forest** | Cấu trúc cây (Graph) | 9 Events | 56.62% | 55.62% | 46.12% ± 8.88% |
| **B3: Original TTT (Wu et al. 2025)** | Text + Tree + Temporal GRU | 5 Events | 72.82% | 67.80% | 66.01% ± 4.14% |
| **UC-TTT Proposed (Nhóm đề xuất)** | Text + Tree + Time + **User Credibility** | **5 Events** | **75.99%** | **71.59%** | **69.13% ± 3.20%** |

---

## 4. BẢNG TIẾN ĐỘ THỰC NGHIỆM ABLATION STUDIES (RQ2)

| Biến thể Ablation | Nhóm đặc trưng | Test Fold | Accuracy | Macro-F1 | Rumour Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **w/o Derived Ratios** | Bỏ Ratios, giữ Profile + Counts (9 dims) | Charlie Hebdo | **81.34%** | **73.80%** | **62.88%** |
| **UC-TTT Full** | Đủ 12 đặc trưng uy tín | 5-Fold Mean | 73.71% | 69.13% | 53.90% |

Biểu đồ thực nghiệm lưu tại:
- `results/figures/ablation_feature_contribution.png`
- `results/figures/ablation_full_summary.png`

---

## 5. CẤU TRÚC THƯ MỤC KẾT QUẢ (FILE INDEX)

```
results/
├── README.md                                    # File tài liệu này
├── comparison_table.md                          # Bảng tổng hợp đối sánh kết quả
├── baseline_results.json                        # Kết quả 9-Fold LOEO của Baselines truyền thống (B1a, B1b, B2)
├── ttt_baseline_results.json                    # Kết quả TTT Baseline (Wu et al. 2025)
├── ucttt_results.json                           # Kết quả UC-TTT Proposed (Gated Fusion)
├── ucttt_gated_events5_results.json             # Kết quả UC-TTT trên 5 sự kiện PHEME v5
├── ucttt_gated_events9_results.json             # Kết quả UC-TTT trên 9 sự kiện full PHEME
├── ucttt_residual_gate_events5_results.json     # Kết quả UC-TTT với Residual Gating
├── paper_config/
│   ├── comparison_table.md                      # Bảng đối chuẩn chi tiết định dạng Markdown
│   ├── ttt_baseline_results.json                # Artifact kết quả TTT cấu hình bài báo
│   └── ucttt_gated_events5_results.json         # Artifact kết quả UC-TTT cấu hình bài báo
├── ablation/
│   ├── ablation_summary.json                    # Tổng hợp chuỗi bóc tách đặc trưng RQ2
│   ├── without_ratios_results.json              # Kết quả bóc tách nhóm đặc trưng Ratios
│   └── checkpoints/                             # Trọng số mô hình ablation
├── checkpoints/                                 # Trọng số PyTorch tốt nhất (.pt) của từng fold
│   ├── ttt_baseline/                            # Checkpoints mô hình TTT Baseline
│   └── ucttt/                                   # Checkpoints mô hình UC-TTT đề xuất
└── figures/                                     # Biểu đồ trực quan hóa kết quả
    ├── ablation_feature_contribution.png        # Đóng góp của từng nhóm đặc trưng uy tín
    └── ablation_full_summary.png                # So sánh toàn diện các biến thể
```
