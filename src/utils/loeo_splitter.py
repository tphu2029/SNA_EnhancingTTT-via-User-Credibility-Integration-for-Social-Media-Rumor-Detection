from typing import List, Dict, Tuple, Any

def get_loeo_folds(events: List[str]) -> List[Dict[str, Any]]:
    """
    Generates 9-fold Leave-One-Event-Out (LOEO) train/val/test event splits.
    For fold i:
      - test_event: events[i]
      - val_event: events[(i + 1) % len(events)]
      - train_events: all other events
    """
    unique_events = sorted(list(set(events)))
    folds = []
    
    for i, test_ev in enumerate(unique_events):
        val_ev = unique_events[(i + 1) % len(unique_events)]
        train_evs = [ev for ev in unique_events if ev != test_ev and ev != val_ev]
        folds.append({
            "fold_idx": i,
            "test_event": test_ev,
            "val_event": val_ev,
            "train_events": train_evs,
        })
    return folds

def split_dataset_by_loeo_fold(data_records: List[Dict[str, Any]], fold: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits records based on a LOEO fold specification.
    """
    test_data = [r for r in data_records if r.get("event") == fold["test_event"]]
    val_data = [r for r in data_records if r.get("event") == fold["val_event"]]
    train_data = [r for r in data_records if r.get("event") in fold["train_events"]]
    
    return train_data, val_data, test_data
