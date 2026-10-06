import os
import sys
import tarfile
import json
import pickle
from typing import Dict, List, Any, Optional
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.data.user_feature_builder import UserCredibilityExtractor
from src.data.tree_builder import extract_tree_edges, build_tree_propagation_metrics

def normalize_event_name(raw_event: str) -> str:
    """
    Chuẩn hóa tên sự kiện từ thư mục, ví dụ: 'charliehebdo-all-rnr-threads' -> 'charliehebdo'
    """
    return raw_event.replace("-all-rnr-threads", "").strip()

def parse_pheme_dir(root_dir: str, max_threads: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Đọc trực tiếp từ thư mục đã giải nén (ví dụ: data/raw/PHEME_veracity/all-rnr-annotated-threads).
    """
    cred_extractor = UserCredibilityExtractor()
    processed_threads: List[Dict[str, Any]] = []

    # root_dir chứa các thư mục sự kiện
    for event_folder in os.listdir(root_dir):
        event_path = os.path.join(root_dir, event_folder)
        if not os.path.isdir(event_path) or event_folder.startswith("."):
            continue
        
        event_name = normalize_event_name(event_folder)
        
        for label_folder in ["rumours", "non-rumours"]:
            label_path = os.path.join(event_path, label_folder)
            if not os.path.exists(label_path):
                continue
            
            label_val = 1 if label_folder == "rumours" else 0
            
            for thread_id in os.listdir(label_path):
                thread_dir = os.path.join(label_path, thread_id)
                if not os.path.isdir(thread_dir) or thread_id.startswith("."):
                    continue
                
                # 1. Source tweet
                source_dir = os.path.join(thread_dir, "source-tweets")
                source_tweet = None
                if os.path.exists(source_dir):
                    s_files = [f for f in os.listdir(source_dir) if f.endswith(".json") and not f.startswith(".")]
                    if s_files:
                        with open(os.path.join(source_dir, s_files[0]), "r", encoding="utf-8") as f:
                            t_data = json.load(f)
                            source_tweet = {
                                "id": str(t_data.get("id_str", t_data.get("id", thread_id))),
                                "text": t_data.get("text", ""),
                                "created_at": t_data.get("created_at"),
                                "credibility_features": cred_extractor.extract_from_tweet(t_data),
                                "raw_user": t_data.get("user", {}),
                            }
                if not source_tweet:
                    continue

                # 2. Reactions
                reactions = {}
                reax_dir = os.path.join(thread_dir, "reactions")
                if os.path.exists(reax_dir):
                    for r_file in os.listdir(reax_dir):
                        if r_file.endswith(".json") and not r_file.startswith("."):
                            r_id = r_file.replace(".json", "")
                            try:
                                with open(os.path.join(reax_dir, r_file), "r", encoding="utf-8") as f:
                                    r_data = json.load(f)
                                    reactions[r_id] = {
                                        "id": r_id,
                                        "text": r_data.get("text", ""),
                                        "created_at": r_data.get("created_at"),
                                        "credibility_features": cred_extractor.extract_from_tweet(r_data),
                                        "raw_user": r_data.get("user", {}),
                                    }
                            except Exception:
                                pass

                # 3. Structure
                raw_edges = []
                struct_path = os.path.join(thread_dir, "structure.json")
                if os.path.exists(struct_path):
                    try:
                        with open(struct_path, "r", encoding="utf-8") as f:
                            s_data = json.load(f)
                            raw_edges = extract_tree_edges(s_data)
                    except Exception:
                        pass

                all_node_ids = [thread_id] + list(reactions.keys())
                node_set = set(all_node_ids)
                valid_edges = [(p, c) for p, c in raw_edges if p in node_set and c in node_set]
                linked_children = {c for _, c in valid_edges}
                for r_id in reactions.keys():
                    if r_id not in linked_children:
                        valid_edges.append((thread_id, r_id))

                prop_metrics = build_tree_propagation_metrics(all_node_ids, valid_edges, root_id=thread_id)

                # 4. Annotation
                annot = None
                annot_path = os.path.join(thread_dir, "annotation.json")
                if os.path.exists(annot_path):
                    try:
                        with open(annot_path, "r", encoding="utf-8") as f:
                            annot = json.load(f)
                    except Exception:
                        pass

                processed_threads.append({
                    "thread_id": thread_id,
                    "event": event_name,
                    "label": label_val,
                    "source_tweet": source_tweet,
                    "reactions": list(reactions.values()),
                    "edges": valid_edges,
                    "num_nodes": len(all_node_ids),
                    "propagation_metrics": prop_metrics,
                    "annotation": annot,
                })

                if max_threads and len(processed_threads) >= max_threads:
                    return processed_threads

    return processed_threads

def parse_pheme_tar(tar_path: str, max_threads: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Đọc trực tiếp từ file nén tar.gz/tar.bz2 của PHEME và phân tích toàn bộ các threads.
    """
    cred_extractor = UserCredibilityExtractor()
    threads_map: Dict[str, Dict[str, Any]] = {}
    
    print(f"Bắt đầu đọc dữ liệu từ archive: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if "/._" in name or name.startswith("._") or ".DS_Store" in name:
                continue
            
            parts = name.split("/")
            # Ví dụ: ['all-rnr-annotated-threads', 'gurlitt-all-rnr-threads', 'rumours', '535438499432771584', 'source-tweets', '...']
            if len(parts) < 4:
                continue
            
            event_name = normalize_event_name(parts[1])
            label_str = parts[2]  # 'rumours' hoặc 'non-rumours'
            thread_id = parts[3]
            
            if thread_id not in threads_map:
                threads_map[thread_id] = {
                    "thread_id": thread_id,
                    "event": event_name,
                    "label": 1 if label_str == "rumours" else 0,
                    "source_tweet": None,
                    "reactions": {},
                    "structure_raw": None,
                    "annotation": None,
                }
            
            t_entry = threads_map[thread_id]
            
            if "source-tweets" in name and name.endswith(".json"):
                f = tar.extractfile(member)
                if f:
                    try:
                        tweet_data = json.load(f)
                        cred_vec = cred_extractor.extract_from_tweet(tweet_data)
                        t_entry["source_tweet"] = {
                            "id": str(tweet_data.get("id_str", tweet_data.get("id", thread_id))),
                            "text": tweet_data.get("text", ""),
                            "created_at": tweet_data.get("created_at"),
                            "credibility_features": cred_vec,
                            "raw_user": tweet_data.get("user", {}),
                        }
                    except Exception as e:
                        pass
            
            elif "reactions" in name and name.endswith(".json"):
                f = tar.extractfile(member)
                if f:
                    try:
                        tweet_data = json.load(f)
                        r_id = str(tweet_data.get("id_str", tweet_data.get("id", parts[-1].replace(".json", ""))))
                        cred_vec = cred_extractor.extract_from_tweet(tweet_data)
                        t_entry["reactions"][r_id] = {
                            "id": r_id,
                            "text": tweet_data.get("text", ""),
                            "created_at": tweet_data.get("created_at"),
                            "credibility_features": cred_vec,
                            "raw_user": tweet_data.get("user", {}),
                        }
                    except Exception as e:
                        pass
            
            elif name.endswith("structure.json"):
                f = tar.extractfile(member)
                if f:
                    try:
                        t_entry["structure_raw"] = json.load(f)
                    except Exception as e:
                        pass
            
            elif name.endswith("annotation.json"):
                f = tar.extractfile(member)
                if f:
                    try:
                        t_entry["annotation"] = json.load(f)
                    except Exception as e:
                        pass

    print(f"Tổng số threads tìm thấy: {len(threads_map)}")
    
    # Bước hoàn thiện cấu trúc cây cho từng thread
    processed_threads: List[Dict[str, Any]] = []
    for tid, t_entry in threads_map.items():
        if t_entry["source_tweet"] is None:
            continue
        
        # Trích xuất cạnh cây
        raw_edges = []
        if t_entry["structure_raw"]:
            raw_edges = extract_tree_edges(t_entry["structure_raw"])
        
        all_node_ids = [tid] + list(t_entry["reactions"].keys())
        node_set = set(all_node_ids)
        valid_edges = [(p, c) for p, c in raw_edges if p in node_set and c in node_set]
        
        # Nếu có reaction nhưng không có trong structure, nối vào root
        linked_children = {c for _, c in valid_edges}
        for r_id in t_entry["reactions"].keys():
            if r_id not in linked_children:
                valid_edges.append((tid, r_id))
        
        prop_metrics = build_tree_propagation_metrics(all_node_ids, valid_edges, root_id=tid)
        
        processed_threads.append({
            "thread_id": tid,
            "event": t_entry["event"],
            "label": t_entry["label"],
            "source_tweet": t_entry["source_tweet"],
            "reactions": list(t_entry["reactions"].values()),
            "edges": valid_edges,
            "num_nodes": len(all_node_ids),
            "propagation_metrics": prop_metrics,
            "annotation": t_entry["annotation"],
        })
        
        if max_threads and len(processed_threads) >= max_threads:
            break

    print(f"Số threads hợp lệ đã tiền xử lý: {len(processed_threads)}")
    return processed_threads

def cache_processed_dataset(dataset: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Đã lưu dataset đã tiền xử lý vào: {output_path}")

def load_cached_dataset(cache_path: str) -> List[Dict[str, Any]]:
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    return data
