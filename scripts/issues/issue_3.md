> 👤 **Người phụ trách:** @ThienPhu (23521188 - Trần Thiên Phú)
> 👥 **Người phối hợp:** @GiaHuy (23520620 - Mạc Nguyễn Gia Huy)

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