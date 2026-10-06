> 👤 **Người phụ trách:** @GiaHuy (23520620 - Mạc Nguyễn Gia Huy)
> 👥 **Người phối hợp:** @ThienPhu (23521188 - Trần Thiên Phú), @TanDat (23520278 - Võ Tấn Đạt)

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