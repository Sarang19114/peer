"""
Weighted K-Nearest Neighbour matcher for athlete peer matching.

Distances are feature-weighted using logistic regression coefficients which
gives greater influence to informative features while remaining instance-based
and interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class WeightedKNNReport:
    metrics: Dict[str, float]
    best_k: int
    confidence_scores: np.ndarray
    feature_weights: pd.Series


class WeightedKNNMatcher:
    """Feature-weighted KNN classifier."""

    def __init__(
        self,
        k: Optional[int] = None,
        feature_weights: Optional[Iterable[float]] = None,
        random_state: int = 42,
    ):
        self.initial_k = k
        self.random_state = random_state
        self.feature_weights = None if feature_weights is None else np.asarray(feature_weights)

        self.scaler = StandardScaler()
        self.model: Optional[KNeighborsClassifier] = None
        self.training_matrix_: Optional[np.ndarray] = None
        self.training_labels_: Optional[np.ndarray] = None
        self.feature_names: Optional[Iterable[str]] = None
        self.best_k_: Optional[int] = None

    def _prepare_weights(self, n_features: int) -> np.ndarray:
        if self.feature_weights is None:
            return np.ones(n_features)

        weights = np.abs(np.asarray(self.feature_weights, dtype=float))
        if weights.shape[0] != n_features:
            raise ValueError("Feature weight length must match number of features")
        if np.allclose(weights, 0):
            return np.ones(n_features)
        return weights / weights.max()

    def _apply_weights(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return X * weights

    def find_optimal_k(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weights: np.ndarray,
        candidate_k: Tuple[int, ...] = (3, 5, 7, 9),
    ) -> int:
        X_weighted = self._apply_weights(X_train, weights)
        X_scaled = self.scaler.fit_transform(X_weighted)

        params = {"n_neighbors": candidate_k}
        knn = KNeighborsClassifier(weights="distance", metric="euclidean")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grid = GridSearchCV(knn, params, cv=cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_scaled, y_train)
        return int(grid.best_params_["n_neighbors"])

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Iterable[str],
        logistic_coefficients: Optional[Iterable[float]] = None,
    ) -> None:
        if logistic_coefficients is not None:
            self.feature_weights = np.asarray(logistic_coefficients)

        self.feature_names = feature_names
        weights = self._prepare_weights(X_train.shape[1])

        if self.initial_k is None:
            self.best_k_ = self.find_optimal_k(X_train, y_train, weights)
        else:
            self.best_k_ = self.initial_k

        X_weighted = self._apply_weights(X_train, weights)
        X_scaled = self.scaler.fit_transform(X_weighted)

        self.model = KNeighborsClassifier(
            n_neighbors=self.best_k_,
            weights="distance",
            metric="euclidean",
        )
        self.model.fit(X_scaled, y_train)

        self.training_matrix_ = X_weighted
        self.training_labels_ = y_train
        self.weight_vector_ = weights

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        weighted = self._apply_weights(X, self.weight_vector_)
        return self.scaler.transform(weighted)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        X_transformed = self._transform(X)
        return self.model.predict_proba(X_transformed)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> WeightedKNNReport:
        proba = self.predict_proba(X_test)
        y_pred = (proba >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        feature_weights = pd.Series(
            self.weight_vector_,
            index=self.feature_names,
            name="distance_weight",
        )

        confidence = self._confidence_scores(X_test)

        return WeightedKNNReport(
            metrics=metrics,
            best_k=self.best_k_,
            confidence_scores=confidence,
            feature_weights=feature_weights,
        )

    def _confidence_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        X_transformed = self._transform(X)
        distances, _ = self.model.kneighbors(X_transformed)
        return 1 / (1 + distances.mean(axis=1))

    def explain_prediction(self, x: np.ndarray, top_k: Optional[int] = None) -> pd.DataFrame:
        if self.model is None or self.training_matrix_ is None:
            raise RuntimeError("Model not fitted")

        k = top_k if top_k is not None else self.best_k_
        x_transformed = self._transform(x.reshape(1, -1))
        distances, indices = self.model.kneighbors(x_transformed, n_neighbors=k)
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]

        votes = self.training_labels_[neighbor_indices]
        vote_weights = 1 / (neighbor_distances + 1e-6)

        df = pd.DataFrame(
            {
                "neighbor_index": neighbor_indices,
                "distance": neighbor_distances,
                "vote_label": votes,
                "vote_weight": vote_weights,
            }
        )
        df["weighted_vote"] = df["vote_label"] * df["vote_weight"]
        return df.sort_values("vote_weight", ascending=False).reset_index(drop=True)

