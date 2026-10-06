import sys
import pickle
import json
import os
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    pkl_path = r"e:\UIT\MXH\PJ\data\processed\pheme_processed.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 1. Events summary
    events_dict = {}
    for item in data:
        ev = item["event"]
        if ev not in events_dict:
            events_dict[ev] = {
                "event": ev,
                "total_threads": 0,
                "rumours": 0,
                "non_rumours": 0,
                "total_reactions": 0,
                "verified_users": 0,
                "max_depths": [],
                "tree_sizes": []
            }
        
        e_entry = events_dict[ev]
        e_entry["total_threads"] += 1
        if item["label"] == 1:
            e_entry["rumours"] += 1
        else:
            e_entry["non_rumours"] += 1
            
        num_reax = len(item.get("reactions", []))
        e_entry["total_reactions"] += num_reax
        
        source = item.get("source_tweet", {})
        if source.get("raw_user", {}).get("verified", False):
            e_entry["verified_users"] += 1
            
        metrics = item.get("propagation_metrics", {})
        e_entry["max_depths"].append(metrics.get("max_depth", 0))
        e_entry["tree_sizes"].append(metrics.get("tree_size", 1))

    summary_rows = []
    for ev, val in sorted(events_dict.items()):
        rumour_pct = (val["rumours"] / val["total_threads"]) * 100
        verified_pct = (val["verified_users"] / val["total_threads"]) * 100
        avg_reax = val["total_reactions"] / val["total_threads"]
        summary_rows.append({
            "Event": ev,
            "Total Threads": val["total_threads"],
            "Rumours": val["rumours"],
            "Non-Rumours": val["non_rumours"],
            "Rumour Ratio (%)": f"{rumour_pct:.1f}%",
            "Total Reactions": val["total_reactions"],
            "Avg Reactions/Thread": f"{avg_reax:.1f}",
            "Avg Tree Depth": f"{np.mean(val['max_depths']):.2f}",
            "Avg Tree Size": f"{np.mean(val['tree_sizes']):.2f}",
            "Source Verified Ratio (%)": f"{verified_pct:.1f}%"
        })

    df_events = pd.DataFrame(summary_rows)
    csv_events_path = r"e:\UIT\MXH\PJ\data\processed\events_summary.csv"
    df_events.to_csv(csv_events_path, index=False, encoding="utf-8-sig")
    print(f"Created events_summary.csv at: {csv_events_path}")

    # 2. Sample thread JSON
    sample = data[0]
    sample_serializable = {
        "thread_id": sample["thread_id"],
        "event": sample["event"],
        "label": sample["label"],
        "is_rumour": "Rumour" if sample["label"] == 1 else "Non-rumour",
        "source_tweet": {
            "id": sample["source_tweet"]["id"],
            "text": sample["source_tweet"]["text"],
            "created_at": sample["source_tweet"]["created_at"],
            "user_screen_name": sample["source_tweet"]["raw_user"].get("screen_name"),
            "user_verified": sample["source_tweet"]["raw_user"].get("verified"),
            "user_followers": sample["source_tweet"]["raw_user"].get("followers_count"),
            "user_friends": sample["source_tweet"]["raw_user"].get("friends_count"),
            "credibility_features_12d": sample["source_tweet"]["credibility_features"].tolist()
        },
        "num_reactions": len(sample["reactions"]),
        "propagation_metrics": sample["propagation_metrics"],
        "edges_sample": sample["edges"][:10]
    }
    sample_json_path = r"e:\UIT\MXH\PJ\data\processed\sample_thread.json"
    with open(sample_json_path, "w", encoding="utf-8") as f:
        json.dump(sample_serializable, f, indent=2, ensure_ascii=False)
    print(f"Created sample_thread.json at: {sample_json_path}")

if __name__ == "__main__":
    main()
