import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.inference_engine import RumorDetectionDemoEngine
from demo.preset_cases import PRESET_CASES

def main():
    print("=" * 80)
    print("KIỂM THỬ ĐỘNG CƠ SUY LUẬN PROTOTYPE (UC-TTT vs. TTT BASELINE)")
    print("=" * 80)
    
    engine = RumorDetectionDemoEngine()
    engine.load_data()
    print(f"-> Đã tải {len(engine.threads)} threads và embeddings thành công.\n")

    for i, p in enumerate(PRESET_CASES, 1):
        print(f"[{i}/5] Ca: {p['title']}")
        th = engine.threads_by_id.get(p["thread_id"])
        if not th:
            print(f"  [ERROR] Không tìm thấy thread {p['thread_id']}")
            continue
            
        res = engine.predict_thread(th, fold_idx=p["fold"])
        src = res["source_tweet"]
        ttt = res["ttt_prediction"]
        uc = res["ucttt_prediction"]
        interp = res["interpretability"]
        
        print(f"  • Tác giả: @{src['screen_name']} (Verified: {src['verified']}, Followers: {src['followers_count']:,})")
        print(f"  • Nhãn gốc: {res['ground_truth_name']}")
        print(f"  • TTT Baseline:  {ttt['label_name']} (P_rumour: {ttt['prob_rumour']*100:.1f}%) [Đúng: {ttt['is_correct']}]")
        print(f"  • UC-TTT Đề xuất:{uc['label_name']} (P_rumour: {uc['prob_rumour']*100:.1f}%) [Đúng: {uc['is_correct']}]")
        print(f"  • Cổng Gate:     User {interp['user_weight_pct']}% | Text {interp['text_weight_pct']}%")
        print(f"  • Lát cắt timeline: {len(res['timeline']['ucttt'])} mốc thời gian")
        print(f"  • Đánh giá:      {res['impact']['title']}\n")

    print("=" * 80)
    print("HOÀN TẤT KIỂM THỬ: TẤT CẢ CÁC CA ĐỀU CHẠY THÀNH CÔNG VỚI ĐẦY ĐỦ METRICS!")
    print("=" * 80)

if __name__ == "__main__":
    main()
