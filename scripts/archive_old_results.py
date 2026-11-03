# -*- coding: utf-8 -*-
"""
Script di chuyển toàn bộ kết quả thực nghiệm cũ (chạy theo tỷ lệ 85/15)
sang thư mục lưu trữ: "kết quả trước đó với test (85-15)".
"""

import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def main():
    src_dir = "results"
    dest_dir = "kết quả trước đó với test (85-15)"

    os.makedirs(dest_dir, exist_ok=True)
    print(f"📁 Đã tạo thư mục lưu trữ: '{dest_dir}'")

    if not os.path.exists(src_dir):
        print(f"Thư mục nguồn '{src_dir}' không tồn tại.")
        return

    items = os.listdir(src_dir)
    print(f"Tổng số mục cần di chuyển từ '{src_dir}': {len(items)}")

    moved_count = 0
    for item in items:
        # Bỏ qua nếu là chính thư mục đích hoặc file đặc biệt
        if item == dest_dir:
            continue
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)
        try:
            # Nếu đích đã tồn tại thì xóa trước hoặc ghi đè
            if os.path.exists(dest_path):
                if os.path.isdir(dest_path):
                    shutil.rmtree(dest_path)
                else:
                    os.remove(dest_path)
            shutil.move(src_path, dest_path)
            print(f"  -> Đã di chuyển: {item}")
            moved_count += 1
        except Exception as e:
            print(f"  [!] Lỗi khi di chuyển {item}: {e}")

    # Tạo lại các thư mục rỗng chuẩn trong results/ để sẵn sàng cho lượt chạy mới
    subdirs = ["ablation", "checkpoints", "figures", "paper_config", "ttt_baseline", "ucttt"]
    for sub in subdirs:
        os.makedirs(os.path.join(src_dir, sub), exist_ok=True)

    print(f"\n✅ Hoàn tất di chuyển {moved_count} mục vào '{dest_dir}'.")
    print("✨ Thư mục 'results/' hiện đã được làm mới, sẵn sàng đón nhận kết quả từ 4 bạn!")

    # Cập nhật .gitignore để không track thư mục lưu trữ lớn này
    gitignore_path = ".gitignore"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            gi_content = f.read()
        ignore_entry = "kết quả trước đó với test*/\n"
        if "kết quả trước đó với test" not in gi_content:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Lưu trữ kết quả cũ (85/15)\n{ignore_entry}")
            print(f"🔒 Đã thêm '{ignore_entry.strip()}' vào .gitignore.")

if __name__ == "__main__":
    main()
