> 👤 **Người phụ trách:** @ThienPhu (23521188 - Trần Thiên Phú)
> 👥 **Người phối hợp:** @ThanhMai (23520910 - Nguyễn Thị Thanh Mai), @BichPhuong (23521239 - Bùi Phạm Bích Phương)

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