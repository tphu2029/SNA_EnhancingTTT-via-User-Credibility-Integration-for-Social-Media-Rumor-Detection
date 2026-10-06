import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Any, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.utils.metrics import compute_all_metrics
from src.utils.loeo_splitter import get_loeo_folds, split_dataset_by_loeo_fold

class TfidfLogisticRegressionBaseline:
    """
    Baseline 1: Content-only rumour detection using TF-IDF and Logistic Regression.
    Evaluated under Leave-One-Event-Out (LOEO) cross-validation protocol.
    """
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2), include_reactions: bool = False):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.include_reactions = include_reactions

    def _prepare_text(self, record: Dict[str, Any]) -> str:
        source_text = record.get("source_tweet", {}).get("text", "")
        if not self.include_reactions:
            return source_text
        reaction_texts = [r.get("text", "") for r in record.get("reactions", [])]
        return source_text + " " + " ".join(reaction_texts)

    def evaluate_loeo(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        events = [r["event"] for r in dataset]
        folds = get_loeo_folds(events)
        
        fold_results = []
        all_true = []
        all_pred = []
        
        for fold in folds:
            test_ev = fold["test_event"]
            train_data, val_data, test_data = split_dataset_by_loeo_fold(dataset, fold)
            
            # Combine train and val for final fit on the fold
            train_full = train_data + val_data
            
            train_texts = [self._prepare_text(r) for r in train_full]
            y_train = [r["label"] for r in train_full]
            
            test_texts = [self._prepare_text(r) for r in test_data]
            y_test = [r["label"] for r in test_data]
            
            # TF-IDF Vectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                stop_words="english",
            )
            X_train = vectorizer.fit_transform(train_texts)
            X_test = vectorizer.transform(test_texts)
            
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test).tolist()
            fold_metrics = compute_all_metrics(y_test, y_pred)
            fold_metrics["test_event"] = test_ev
            fold_metrics["num_samples"] = len(y_test)
            fold_results.append(fold_metrics)
            
            all_true.extend(y_test)
            all_pred.extend(y_pred)

        overall_metrics = compute_all_metrics(all_true, all_pred)
        
        # Mean across folds
        macro_f1_scores = [f["macro_f1"] for f in fold_results]
        accuracies = [f["accuracy"] for f in fold_results]
        
        return {
            "model_name": f"TF-IDF + Logistic Regression ({'Source+Reactions' if self.include_reactions else 'Source-only'})",
            "overall_pooled_metrics": overall_metrics,
            "mean_macro_f1": float(np.mean(macro_f1_scores)),
            "std_macro_f1": float(np.std(macro_f1_scores)),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "fold_results": fold_results,
        }
