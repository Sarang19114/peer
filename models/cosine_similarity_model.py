"""
Cosine similarity baseline for athlete peer matching.

The model projects each pairwise feature vector onto the positive-class
centroid (computed from the training data) and uses cosine similarity as
the matching score. A configurable threshold converts similarity into a
binary prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer


@dataclass
class CosineSimilarityReport:
    metrics: Dict[str, float]
    similarity_scores: np.ndarray
    threshold_sweep: pd.DataFrame


class CosineMatchingModel:
    """Threshold-based cosine similarity classifier."""

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.normalizer = Normalizer(norm="l2")
        self.positive_centroid: Optional[np.ndarray] = None
        self.negative_centroid: Optional[np.ndarray] = None
        self.feature_names: Optional[Iterable[str]] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Iterable[str]) -> None:
        self.feature_names = feature_names
        X_norm = self.normalizer.fit_transform(X_train)

        positives = X_norm[y_train == 1]
        negatives = X_norm[y_train == 0]

        if len(positives) == 0 or len(negatives) == 0:
            raise ValueError("Training data must contain both positive and negative samples")

        positive_mean = positives.mean(axis=0, keepdims=True)
        negative_mean = negatives.mean(axis=0, keepdims=True)

        self.positive_centroid = positive_mean / np.linalg.norm(positive_mean)
        self.negative_centroid = negative_mean / np.linalg.norm(negative_mean)

        # Automatically set decision threshold using training data (F1 optimisation)
        scores = self._compute_similarity(X_train, normalised=True)
        thresholds = np.linspace(0.3, 0.7, 17)
        best_f1 = -1.0
        best_threshold = self.threshold
        for th in thresholds:
            preds = (scores >= th).astype(int)
            tp = ((y_train == 1) & (preds == 1)).sum()
            fp = ((y_train == 0) & (preds == 1)).sum()
            fn = ((y_train == 1) & (preds == 0)).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th
        self.threshold = float(best_threshold)

    def _compute_similarity(self, X: np.ndarray, normalised: bool = False) -> np.ndarray:
        if self.positive_centroid is None:
            raise RuntimeError("Model not fitted")

        X_norm = X if normalised else self.normalizer.transform(X)
        pos_sim = cosine_similarity(X_norm, self.positive_centroid).ravel()
        neg_sim = cosine_similarity(X_norm, self.negative_centroid).ravel()
        score = (pos_sim - neg_sim + 1.0) / 2.0
        return np.clip(score, 0.0, 1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._compute_similarity(X)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        scores = self.predict_proba(X)
        return (scores >= th).astype(int)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Path,
        logistic_metrics: Optional[Dict[str, float]] = None,
        threshold_grid: Optional[np.ndarray] = None,
    ) -> CosineSimilarityReport:
        output_dir.mkdir(parents=True, exist_ok=True)

        scores = self.predict_proba(X_test)
        y_pred = (scores >= self.threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, scores),
        }

        self._plot_similarity_distribution(scores, output_dir)
        threshold_summary = self._plot_threshold_sweep(
            scores,
            y_test,
            output_dir,
            threshold_grid=threshold_grid,
        )
        self._plot_roc_curve(scores, y_test, output_dir)
        if logistic_metrics:
            self._create_comparison_table(metrics, logistic_metrics, output_dir)

        return CosineSimilarityReport(
            metrics=metrics,
            similarity_scores=scores,
            threshold_sweep=threshold_summary,
        )

    # ------------------------------------------------------------------ #
    # Plotting utilities
    # ------------------------------------------------------------------ #
    def _plot_similarity_distribution(self, scores: np.ndarray, output_dir: Path) -> None:
        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=30, color="#4C72B0", alpha=0.85)
        plt.axvline(self.threshold, color="#C44E52", linestyle="--", label=f"Threshold {self.threshold:.2f}")
        plt.title("Cosine Similarity Distribution")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "cosine_similarity_distribution.png", dpi=300)
        plt.close()

    def _plot_threshold_sweep(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        output_dir: Path,
        threshold_grid: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if threshold_grid is None:
            threshold_grid = np.linspace(0.3, 0.9, 13)

        summary_rows = []
        for th in threshold_grid:
            preds = (scores >= th).astype(int)
            summary_rows.append(
                {
                    "threshold": th,
                    "accuracy": accuracy_score(y_true, preds),
                    "precision": precision_score(y_true, preds, zero_division=0),
                    "recall": recall_score(y_true, preds, zero_division=0),
                    "f1": f1_score(y_true, preds, zero_division=0),
                }
            )

        summary_df = pd.DataFrame(summary_rows)

        plt.figure(figsize=(9, 6))
        for metric in ["accuracy", "precision", "recall", "f1"]:
            plt.plot(summary_df["threshold"], summary_df[metric], marker="o", label=metric.capitalize())
        plt.axvline(self.threshold, color="gray", linestyle="--", label="Default threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title("Threshold Sensitivity Analysis")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "cosine_threshold_sensitivity.png", dpi=300)
        plt.close()

        summary_df.to_csv(output_dir / "cosine_threshold_sweep.csv", index=False)
        return summary_df

    def _plot_roc_curve(self, scores: np.ndarray, y_true: np.ndarray, output_dir: Path) -> None:
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#55A868", label=f"Cosine Similarity (AUC={roc_auc_score(y_true, scores):.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Cosine Similarity ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "cosine_roc_curve.png", dpi=300)
        plt.close()

    def _create_comparison_table(
        self,
        cosine_metrics: Dict[str, float],
        logistic_metrics: Dict[str, float],
        output_dir: Path,
    ) -> None:
        table = pd.DataFrame(
            [
                {"model": "Logistic Regression", **logistic_metrics},
                {"model": "Cosine Similarity", **cosine_metrics},
            ]
        )
        table.to_csv(output_dir / "cosine_vs_logistic_metrics.csv", index=False)

