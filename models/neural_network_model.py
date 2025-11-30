"""
Deep neural network matcher built with TensorFlow/Keras.

Implements a fully-connected architecture with batch normalisation,
dropout and L2 regularisation. Provides training history plots,
ROC/confusion matrix visualisations and simple gradient-based
feature attributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_matching_network(input_dim: int) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(
                16,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


@dataclass
class NeuralNetworkReport:
    metrics: Dict[str, float]
    history: pd.DataFrame
    confusion_matrix: np.ndarray
    feature_attributions: pd.Series


class DeepMatchingNetwork:
    """Deep learning model with training diagnostics."""

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 150,
        patience: int = 10,
        validation_split: float = 0.2,
        random_seed: int = 42,
    ):
        tf.keras.utils.set_random_seed(random_seed)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_split = validation_split
        self.history_: Optional[keras.callbacks.History] = None
        self.history_df_: Optional[pd.DataFrame] = None
        self.feature_names: Optional[Iterable[str]] = None

        self.model = build_matching_network(input_dim)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.BinaryAccuracy(name="accuracy"),
            ],
        )
        self.callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=patience,
                restore_best_weights=True,
            )
        ]

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Iterable[str]) -> None:
        self.feature_names = feature_names
        self.history_ = self.model.fit(
            X_train,
            y_train,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=self.callbacks,
            verbose=0,
        )
        self.history_df_ = pd.DataFrame(self.history_.history)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Path,
        threshold: float | None = None,
    ) -> NeuralNetworkReport:
        output_dir.mkdir(parents=True, exist_ok=True)

        y_proba = self.predict_proba(X_test)
        optimal_threshold = self._optimal_threshold(y_test, y_proba)
        chosen_threshold = threshold if threshold is not None else optimal_threshold
        y_pred = (y_proba >= chosen_threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "threshold": chosen_threshold,
        }

        cm = confusion_matrix(y_test, y_pred)
        self._plot_training_history(output_dir)
        self._plot_roc_curve(y_test, y_proba, output_dir)
        self._plot_confusion_matrix(cm, output_dir)

        attributions = self._compute_feature_attributions(X_test, y_test)
        attributions.to_csv(output_dir / "nn_feature_attributions.csv")

        return NeuralNetworkReport(
            metrics=metrics,
            history=self.history_df_.copy(),
            confusion_matrix=cm,
            feature_attributions=attributions,
        )

    # ------------------------------------------------------------------ #
    # Diagnostics and plotting
    # ------------------------------------------------------------------ #
    def _plot_training_history(self, output_dir: Path) -> None:
        if self.history_df_ is None:
            return
        plt.figure(figsize=(10, 5))
        for metric in ["loss", "auc"]:
            plt.plot(self.history_df_[metric], label=f"train_{metric}")
            val_key = f"val_{metric}"
            if val_key in self.history_df_:
                plt.plot(self.history_df_[val_key], label=val_key)
        plt.xlabel("Epochs")
        plt.ylabel("Metric value")
        plt.title("Neural Network Training History")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "nn_training_history.png", dpi=300)
        plt.close()

    def _plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, output_dir: Path) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_proba):.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Neural Network ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "nn_roc_curve.png", dpi=300)
        plt.close()

    def _plot_confusion_matrix(self, cm: np.ndarray, output_dir: Path) -> None:
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"],
        )
        plt.title("Neural Network Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "nn_confusion_matrix.png", dpi=300)
        plt.close()

    def _compute_feature_attributions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_samples: int = 256,
    ) -> pd.Series:
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        sample_indices = np.random.choice(
            len(X), size=min(max_samples, len(X)), replace=False
        )
        X_sample = tf.convert_to_tensor(X[sample_indices], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_sample)
            preds = self.model(X_sample)

        gradients = tape.gradient(preds, X_sample).numpy()
        attribution = np.mean(np.abs(gradients), axis=0)
        return pd.Series(attribution, index=self.feature_names, name="attribution").sort_values(ascending=False)

    @staticmethod
    def _optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        thresholds = np.linspace(0.3, 0.7, 81)
        best_f1 = -1.0
        best_threshold = 0.5
        for th in thresholds:
            preds = (y_scores >= th).astype(int)
            tp = ((y_true == 1) & (preds == 1)).sum()
            fp = ((y_true == 0) & (preds == 1)).sum()
            fn = ((y_true == 1) & (preds == 0)).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th
        return float(best_threshold)

