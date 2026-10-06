import sys
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Any, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.utils.metrics import compute_all_metrics
from src.utils.loeo_splitter import get_loeo_folds, split_dataset_by_loeo_fold
from src.data.user_feature_builder import parse_twitter_datetime

class PropagationRandomForestBaseline:
    """
    Baseline 2: Propagation & Diffusion features using Random Forest.
    Evaluated under Leave-One-Event-Out (LOEO) cross-validation.
    """
    def __init__(self, n_estimators: int = 150, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _extract_features(self, record: Dict[str, Any]) -> np.ndarray:
        metrics = record.get("propagation_metrics", {})
        tree_size = metrics.get("tree_size", 1.0)
        max_depth = metrics.get("max_depth", 0.0)
        max_breadth = metrics.get("max_breadth", 1.0)
        avg_branching = metrics.get("avg_branching", 0.0)
        leaf_ratio = metrics.get("leaf_ratio", 0.0)
        
        # Temporal diffusion span
        source_tweet = record.get("source_tweet", {})
        root_dt = parse_twitter_datetime(source_tweet.get("created_at"))
        
        reaction_timestamps = []
        for r in record.get("reactions", []):
            dt = parse_twitter_datetime(r.get("created_at"))
            if dt and root_dt:
                diff = max(0.0, (dt - root_dt).total_seconds())
                reaction_timestamps.append(diff)
                
        if reaction_timestamps:
            duration_hours = max(reaction_timestamps) / 3600.0
            avg_inter_arrival = np.mean(reaction_timestamps) / 3600.0
            median_inter_arrival = np.median(reaction_timestamps) / 3600.0
        else:
            duration_hours = 0.0
            avg_inter_arrival = 0.0
            median_inter_arrival = 0.0

        features = [
            np.log1p(tree_size),
            max_depth,
            np.log1p(max_breadth),
            avg_branching,
            leaf_ratio,
            np.log1p(duration_hours),
            np.log1p(avg_inter_arrival),
            np.log1p(median_inter_arrival),
        ]
        return np.array(features, dtype=np.float32)

    def evaluate_loeo(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        events = [r["event"] for r in dataset]
        folds = get_loeo_folds(events)
        
        fold_results = []
        all_true = []
        all_pred = []
        
        # Precompute features for speed
        X_all = np.array([self._extract_features(r) for r in dataset])
        y_all = np.array([r["label"] for r in dataset])
        events_all = np.array(events)
        
        for fold in folds:
            test_ev = fold["test_event"]
            train_mask = np.isin(events_all, fold["train_events"] + [fold["val_event"]])
            test_mask = events_all == test_ev
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test).tolist()
            fold_metrics = compute_all_metrics(y_test.tolist(), y_pred)
            fold_metrics["test_event"] = test_ev
            fold_metrics["num_samples"] = len(y_test)
            fold_results.append(fold_metrics)
            
            all_true.extend(y_test.tolist())
            all_pred.extend(y_pred)

        overall_metrics = compute_all_metrics(all_true, all_pred)
        macro_f1_scores = [f["macro_f1"] for f in fold_results]
        accuracies = [f["accuracy"] for f in fold_results]
        
        return {
            "model_name": "Propagation Features + Random Forest",
            "overall_pooled_metrics": overall_metrics,
            "mean_macro_f1": float(np.mean(macro_f1_scores)),
            "std_macro_f1": float(np.std(macro_f1_scores)),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "fold_results": fold_results,
        }
