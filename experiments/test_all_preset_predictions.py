import json
import pickle
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.inference_engine import RumorDetectionDemoEngine
from demo.preset_cases import PRESET_CASES

def main():
    engine = RumorDetectionDemoEngine()
    engine.load_data()
    
    print("=" * 80)
    print("KIỂM TRA CÁC CA PRESET TRONG DEMO BẰNG BỘ MÁY SUY LUẬN THẬT (PYTORCH)")
    print("=" * 80)
    
    for p in PRESET_CASES:
        th_id = p["thread_id"]
        th = engine.threads_by_id[th_id]
        res = engine.predict_thread(th, fold_idx=p["fold"])
        
        ttt = res["ttt_prediction"]
        uc = res["ucttt_prediction"]
        interp = res["interpretability"]
        
        print(f"\n[{p['title']}] - Thread ID: {th_id} (Fold {p['fold']}: {p['event']})")
        print(f"  Nhãn thực tế (Ground Truth): {res['ground_truth_name']} ({res['ground_truth']})")
        print(f"  Tác giả: @{res['source_tweet']['screen_name']} (Followers: {res['source_tweet']['followers_count']:,}, Verified: {res['source_tweet']['verified']})")
        print(f"  Nội dung: {res['source_tweet']['text'][:90]}...")
        print(f"  TTT Baseline : Dự đoán={ttt['label_name']} | P(Rumour)={ttt['prob_rumour']*100:.2f}% | P(Non)={ttt['prob_non_rumour']*100:.2f}% | Đúng={ttt['is_correct']}")
        print(f"  UC-TTT Đề Xuất: Dự đoán={uc['label_name']} | P(Rumour)={uc['prob_rumour']*100:.2f}% | P(Non)={uc['prob_non_rumour']*100:.2f}% | Đúng={uc['is_correct']}")
        print(f"  Cổng Gate α  : User Weight={interp['user_weight_pct']:.1f}% | Text Weight={interp['text_weight_pct']:.1f}%")
        print(f"  Trạng thái tác động: {res['impact']['status']} -> {res['impact']['title']}")

    # Check the error analysis threads from proposal
    print("\n" + "=" * 80)
    print("KIỂM TRA CÁC CA TỪ ERROR ANALYSIS REPORT (PROPOSAL)")
    print("=" * 80)
    
    error_thread_ids = [
        ("580882341880446977", 2, "BuzzFeed Germanwings"),
        ("544390130682068993", 4, "cnnbrk Sydney Siege"),
        ("580370249712697344", 2, "SkyNews Germanwings")
    ]
    
    for th_id, fold_idx, desc in error_thread_ids:
        if th_id not in engine.threads_by_id:
            print(f"Không tìm thấy thread {th_id}")
            continue
        th = engine.threads_by_id[th_id]
        res = engine.predict_thread(th, fold_idx=fold_idx)
        ttt = res["ttt_prediction"]
        uc = res["ucttt_prediction"]
        interp = res["interpretability"]
        print(f"\n[{desc}] - ID {th_id} (Event: {th['event']})")
        print(f"  Nhãn thực tế: {res['ground_truth_name']}")
        print(f"  Tác giả: @{res['source_tweet']['screen_name']}")
        print(f"  Nội dung: {res['source_tweet']['text'][:90]}...")
        print(f"  TTT Baseline : {ttt['label_name']} (P(Rumour)={ttt['prob_rumour']*100:.2f}%) | Đúng={ttt['is_correct']}")
        print(f"  UC-TTT Đề Xuất: {uc['label_name']} (P(Rumour)={uc['prob_rumour']*100:.2f}%) | Đúng={uc['is_correct']}")
        print(f"  Cổng Gate α  : User={interp['user_weight_pct']:.1f}% | Text={interp['text_weight_pct']:.1f}%")

if __name__ == "__main__":
    main()
