# -*- coding: utf-8 -*-
"""
Baseline 2b: Tabular Features + XGBoost Classifier.
Kết hợp 36 đặc trưng dạng bảng:
  - 12 đặc trưng User Credibility của bài đăng gốc (Source tweet)
  - 12 đặc trưng User Credibility trung bình của các phản hồi (Reactions)
  - 8 đặc trưng topo hình thái lan truyền cây (Propagation topology)
  - 4 đặc trưng bề mặt văn bản bài gốc (Text surface statistics)
Đánh giá theo giao thức Leave-One-Event-Out (LOEO) trên PHEME.
"""

import sys
import numpy as np
from typing import List, Dict, Any
from xgboost import XGBClassifier

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.utils.metrics import compute_all_metrics
from src.utils.loeo_splitter import get_loeo_folds
from src.data.user_feature_builder import UserCredibilityExtractor, parse_twitter_datetime

class TabularXGBoostBaseline:
    """
    Baseline 2b: Tabular ML sử dụng XGBoost trên 36 đặc trưng kết hợp.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.extractor = UserCredibilityExtractor()

    def extract_features(self, record: Dict[str, Any]) -> np.ndarray:
        # 1. 12 đặc trưng User Credibility bài gốc (Source tweet user)
        src_feat = self.extractor.extract_from_tweet(record.get('source_tweet', {}))

        # 2. 12 đặc trưng User Credibility trung bình của phản hồi (Reactions users)
        reactions = record.get('reactions', [])
        if reactions:
            react_feats = np.array([self.extractor.extract_from_tweet(rx) for rx in reactions])
            react_mean = np.mean(react_feats, axis=0)
        else:
            react_mean = np.zeros(12, dtype=np.float32)

        # 3. 8 đặc trưng topo cây lan truyền (Propagation metrics)
        metrics = record.get('propagation_metrics', {})
        tree_size = metrics.get('tree_size', 1.0)
        max_depth = metrics.get('max_depth', 0.0)
        max_breadth = metrics.get('max_breadth', 1.0)
        avg_branching = metrics.get('avg_branching', 0.0)
        leaf_ratio = metrics.get('leaf_ratio', 0.0)

        source_tweet = record.get('source_tweet', {})
        root_dt = parse_twitter_datetime(source_tweet.get('created_at'))
        reaction_timestamps = []
        for rx in reactions:
            dt = parse_twitter_datetime(rx.get('created_at'))
            if dt and root_dt:
                reaction_timestamps.append(max(0.0, (dt - root_dt).total_seconds()))

        if reaction_timestamps:
            duration_hours = max(reaction_timestamps) / 3600.0
            avg_inter_arrival = np.mean(reaction_timestamps) / 3600.0
            median_inter_arrival = np.median(reaction_timestamps) / 3600.0
        else:
            duration_hours = avg_inter_arrival = median_inter_arrival = 0.0

        prop_feats = [
            np.log1p(tree_size),
            max_depth,
            np.log1p(max_breadth),
            avg_branching,
            leaf_ratio,
            np.log1p(duration_hours),
            np.log1p(avg_inter_arrival),
            np.log1p(median_inter_arrival)
        ]

        # 4. 4 đặc trưng bề mặt văn bản bài gốc (Text surface features)
        txt = source_tweet.get('text', '')
        txt_feats = [
            len(txt) / 100.0,
            len(txt.split()) / 20.0,
            float('?' in txt),
            float('!' in txt)
        ]

        return np.concatenate([src_feat, react_mean, prop_feats, txt_feats]).astype(np.float32)

    def evaluate_loeo(self, dataset: List[Dict[str, Any]], events_mode: str = "5") -> Dict[str, Any]:
        """
        Đánh giá Leave-One-Event-Out:
          - events_mode='5': Chỉ chạy trên 5 sự kiện chuẩn (PHEME v5).
          - events_mode='9': Chạy trên full 9 sự kiện.
        """
        PHEME_5_EVENTS = ["charliehebdo", "sydneysiege", "ferguson", "ottawashooting", "germanwings-crash"]
        if events_mode == "5":
            target_dataset = [r for r in dataset if r["event"] in PHEME_5_EVENTS]
        else:
            target_dataset = dataset

        events = [r["event"] for r in target_dataset]
        folds = get_loeo_folds(events)

        X_all = np.array([self.extract_features(r) for r in target_dataset])
        y_all = np.array([r["label"] for r in target_dataset])
        events_all = np.array(events)

        fold_results = []
        all_true = []
        all_pred = []

        for fold in folds:
            test_ev = fold["test_event"]
            train_mask = np.isin(events_all, fold["train_events"] + [fold["val_event"]])
            test_mask = events_all == test_ev

            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]

            clf = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric="logloss"
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
            "model_name": f"Tabular Features + XGBoost (PHEME {events_mode} Events)",
            "overall_pooled_metrics": overall_metrics,
            "mean_macro_f1": float(np.mean(macro_f1_scores)),
            "std_macro_f1": float(np.std(macro_f1_scores)),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "fold_results": fold_results,
        }
