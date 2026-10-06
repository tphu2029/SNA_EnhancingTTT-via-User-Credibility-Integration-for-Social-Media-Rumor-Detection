> 👤 **Người phụ trách:** @BichPhuong (23521239 - Bùi Phạm Bích Phương)
> 👥 **Người phối hợp:** @ThanhMai (23520910 - Nguyễn Thị Thanh Mai), @ThienPhu (23521188 - Trần Thiên Phú)

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