"""
Random Forest matcher for athlete peer matching.

Provides ensemble-based modelling with built-in feature importance,
out-of-bag validation and per-tree vote inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


@dataclass
class RandomForestReport:
    metrics: Dict[str, float]
    oob_score: Optional[float]
    feature_importance: pd.DataFrame
    tree_depths: np.ndarray


class RandomForestMatchingModel:
    """Random forest classifier with diagnostic utilities."""

    def __init__(
        self,
        n_trees: int = 200,
        random_state: int = 42,
        **kwargs,
    ):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            oob_score=True,
            n_jobs=-1,
            random_state=random_state,
            **kwargs,
        )
        self.feature_names: Optional[Iterable[str]] = None

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, object]:
        param_grid = {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_features": ["sqrt", "log2"],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        return search.best_params_

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Iterable[str]) -> None:
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_names is None:
            raise RuntimeError("Model not fitted")
        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        return importance.reset_index(drop=True)

    def predict_tree_votes(self, X: np.ndarray) -> np.ndarray:
        """Return per-tree predicted probabilities."""
        tree_votes = np.array([tree.predict_proba(X)[:, 1] for tree in self.model.estimators_])
        return tree_votes.T

    def explain_prediction(self, x: np.ndarray) -> Dict[str, object]:
        """Extract decision paths from the first tree for interpretability."""
        tree = self.model.estimators_[0]
        node_indicator = tree.decision_path(x.reshape(1, -1))
        feature_index = tree.tree_.feature

        path = []
        for node_id in node_indicator.indices:
            if feature_index[node_id] != -2:
                threshold = tree.tree_.threshold[node_id]
                feature_name = (
                    self.feature_names[feature_index[node_id]]
                    if self.feature_names is not None
                    else f"feature_{feature_index[node_id]}"
                )
                path.append(
                    {
                        "feature": feature_name,
                        "threshold": threshold,
                    }
                )
        return {
            "tree_id": 0,
            "decision_path": path,
        }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> RandomForestReport:
        proba = self.predict_proba(X_test)
        y_pred = (proba >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        feature_importance = self.get_feature_importance()
        tree_depths = np.array([estimator.tree_.max_depth for estimator in self.model.estimators_])

        return RandomForestReport(
            metrics=metrics,
            oob_score=self.model.oob_score_,
            feature_importance=feature_importance,
            tree_depths=tree_depths,
        )

