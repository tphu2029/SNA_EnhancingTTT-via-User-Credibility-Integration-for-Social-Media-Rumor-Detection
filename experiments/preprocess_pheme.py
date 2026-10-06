import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.pheme_parser import parse_pheme_tar, parse_pheme_dir, cache_processed_dataset

def main():
    raw_dir = r"e:\UIT\MXH\PJ\data\raw\PHEME_veracity\all-rnr-annotated-threads"
    tar_path = r"e:\UIT\MXH\PHEME_veracity.tar.bz2"
    output_path = r"e:\UIT\MXH\PJ\data\processed\pheme_processed.pkl"
    
    print("==================================================")
    print("Bắt đầu tiền xử lý bộ dữ liệu PHEME...")
    
    start_time = time.time()
    if os.path.exists(raw_dir):
        print(f"Đang đọc từ thư mục: {raw_dir}")
        dataset = parse_pheme_dir(raw_dir)
    elif os.path.exists(tar_path):
        print(f"Đang đọc từ archive: {tar_path}")
        dataset = parse_pheme_tar(tar_path)
    else:
        print(f"Lỗi: Không tìm thấy dữ liệu tại {raw_dir} hoặc {tar_path}")
        return
    cache_processed_dataset(dataset, output_path)
    
    elapsed = time.time() - start_time
    print(f"Hoàn thành tiền xử lý trong {elapsed:.1f}s.")
    
    # In thống kê tổng quan
    events_count = {}
    rumour_count = {0: 0, 1: 0}
    for item in dataset:
        ev = item["event"]
        events_count[ev] = events_count.get(ev, 0) + 1
        rumour_count[item["label"]] += 1
        
    print("\n--- THỐNG KÊ DATASET ĐÃ TIỀN XỬ LÝ ---")
    print(f"Tổng số threads: {len(dataset)}")
    print(f"Non-rumours (0): {rumour_count[0]} | Rumours (1): {rumour_count[1]}")
    print("Phân bố theo từng sự kiện:")
    for ev, cnt in sorted(events_count.items()):
        print(f"  - {ev:25s}: {cnt:5d} threads")

if __name__ == "__main__":
    main()
