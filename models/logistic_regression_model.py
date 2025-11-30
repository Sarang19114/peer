"""
Logistic Regression baseline model for athlete peer matching.

This wrapper mirrors the calibrated logistic regression configuration
from the main pipeline so it can be reused inside the model comparison
framework. It handles:
    - Train/test scaling
    - Optional SMOTE class balancing
    - Model calibration for reliable probabilities
    - Standard evaluation metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


@dataclass
class LogisticRegressionReport:
    metrics: Dict[str, float]
    coefficients: pd.Series
    confusion_matrix: np.ndarray
    roc_auc_cv_scores: np.ndarray


class LogisticMatchingModel:
    """Calibrated logistic regression with optional SMOTE balancing."""

    def __init__(
        self,
        use_smote: bool = True,
        random_state: int = 42,
        regularization: float = 0.1,
    ):
        self.use_smote = use_smote
        self.random_state = random_state
        self.regularization = regularization

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            C=regularization,
        )
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.coefficients_: Optional[pd.Series] = None
        self.roc_auc_cv_scores_: Optional[np.ndarray] = None

    def _balance(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_smote:
            return X, y

        smote = SMOTE(random_state=self.random_state)
        return smote.fit_resample(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        X_balanced, y_balanced = self._balance(X, y)

        X_scaled = self.scaler.fit_transform(X_balanced)

        self.model.fit(X_scaled, y_balanced)

        self.calibrator = CalibratedClassifierCV(
            estimator=LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                C=self.regularization,
            ),
            cv=5,
            method="sigmoid",
        )
        self.calibrator.fit(X_scaled, y_balanced)

        self.coefficients_ = pd.Series(
            self.model.coef_[0], index=feature_names, name="coefficient"
        )

        cv_pipeline = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            C=self.regularization,
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.roc_auc_cv_scores_ = cross_val_score(
            cv_pipeline,
            X_scaled,
            y_balanced,
            cv=skf,
            scoring="roc_auc",
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.calibrator.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5
    ) -> LogisticRegressionReport:
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
        }

        return LogisticRegressionReport(
            metrics=metrics,
            coefficients=self.coefficients_.copy() if self.coefficients_ is not None else pd.Series(dtype=float),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            roc_auc_cv_scores=self.roc_auc_cv_scores_.copy()
            if self.roc_auc_cv_scores_ is not None
            else np.array([]),
        )

