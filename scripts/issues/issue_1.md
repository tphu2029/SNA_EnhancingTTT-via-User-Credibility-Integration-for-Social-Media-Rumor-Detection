> 👤 **Người phụ trách:** @GiaHuy (23520620 - Mạc Nguyễn Gia Huy)
> 👥 **Người phối hợp:** @TanDat (23520278 - Võ Tấn Đạt)

---

### 📖 DESCRIPTION (MÔ TẢ CHI TIẾT KỸ THUẬT):

#### 1. Bối cảnh & Mục tiêu:
Dữ liệu PHEME sau khi tiền xử lý tại `data/processed/pheme_processed.pkl` chứa **6,425 threads** với tổng cộng hơn **111,000 tweets** (gồm tweet gốc và reactions). Nếu để mô hình nạp raw text và fine-tune trực tiếp qua Transformer trong mỗi epoch của 9-fold LOEO, thời gian huấn luyện sẽ rất chậm và dễ gây tràn bộ nhớ GPU. Do đó, theo Mục 2 - Giai đoạn 1 của `Plan.md`, nhóm cần trích xuất offline sentence embeddings một lần duy nhất và lưu cache nhị phân ra đĩa.

#### 2. Dữ liệu đầu vào (Input):
- File cache: `data/processed/pheme_processed.pkl` (Chứa list các dict thread; mỗi thread có `source_tweet` và danh sách `reactions`, trong đó mỗi node có `tweet_id` và `text`).

#### 3. Nhiệm vụ cụ thể cần thực hiện:
1. Tạo tệp mã nguồn: `experiments/extract_text_embeddings.py`.
2. Sử dụng thư viện `sentence-transformers` hoặc HuggingFace `transformers` với pre-trained model:
   - Mô hình đề xuất chính: `sentence-transformers/all-MiniLM-L6-v2` (output 384 chiều, tốc độ trích xuất cực nhanh).
   - Thiết lập fallback dự phòng: `microsoft/deberta-v3-small`.
3. Viết vòng lặp thu gom toàn bộ văn bản của source tweets và reactions, loại bỏ trùng lặp tweet_id (nếu có), làm sạch cơ bản (loại bỏ URL, ký tự điều khiển rác).
4. Thực hiện batch encoding trên GPU CUDA (`batch_size=128` hoặc `256`) với `torch.no_grad()`.
5. Tạo dictionary ánh xạ: `{int(tweet_id): torch.FloatTensor(embedding_vector)}`.
6. Lưu tensor dictionary ra đĩa: `data/processed/text_embeddings.pt` bằng `torch.save()`.
7. Viết hàm kiểm tra (sanity check) đảm bảo:
   - Tổng số key trong dict khớp với tổng số tweet duy nhất trong `pheme_processed.pkl`.
8. **Upload file nặng lên GitHub Releases:** Đính kèm file `text_embeddings.pt` lên Release với tag `v0.1-data-cache` (xem [Hướng dẫn chi tiết tại Mục 4.4](#quy-trinh-releases) hoặc xem trên [GitHub](https://github.com/tphu2029/SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection/blob/main/TODO.md#quy-trinh-releases)):
   - **Bước 1:** Vào tab **Releases** trên repo $\rightarrow$ bấm **Draft a new release**.
   - **Bước 2:** Đặt tag `v0.1-data-cache`, tiêu đề `PHEME Processed Cache & Offline Text Embeddings`.
   - **Bước 3:** Kéo thả file `data/processed/text_embeddings.pt` vào ô binaries $\rightarrow$ bấm **Publish release**.

#### 4. Lệnh chạy thực thi & Kiểm thử:
```bash
python experiments/extract_text_embeddings.py --model_name all-MiniLM-L6-v2 --batch_size 256
```

#### 5. Sản phẩm bàn giao (Deliverables & DoD):
- [ ] Tệp `experiments/extract_text_embeddings.py` hoàn chỉnh, có log tiến độ `tqdm`.
- [ ] Tệp cache `data/processed/text_embeddings.pt` sẵn sàng trong thư mục dữ liệu cục bộ.
- [ ] Đã upload `text_embeddings.pt` lên **GitHub Releases (Tag `v0.1-data-cache`)**.
- [ ] Bổ sung đường dẫn `data/processed/*.pt` vào `.gitignore` để tránh đẩy file nặng lên Git commit.
```

---