> 👤 **Người phụ trách:** @TanDat (23520278 - Võ Tấn Đạt)
> 👥 **Người phối hợp:** @BichPhuong (23521239 - Bùi Phạm Bích Phương), @ThanhMai (23520910 - Nguyễn Thị Thanh Mai)

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