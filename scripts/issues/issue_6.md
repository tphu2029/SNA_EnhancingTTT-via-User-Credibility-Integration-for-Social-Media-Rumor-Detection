> 👤 **Người phụ trách:** @ThanhMai (23520910 - Nguyễn Thị Thanh Mai)
> 👥 **Người phối hợp:** @BichPhuong (23521239 - Bùi Phạm Bích Phương), @ThienPhu (23521188 - Trần Thiên Phú)

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