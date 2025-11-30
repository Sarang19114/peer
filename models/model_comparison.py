"""
Model comparison framework for athlete peer matching.

Trains and evaluates multiple algorithms on a shared train/test split,
aggregates metrics, benchmarking information, and generates comparative
visualisations and PDF-ready artefacts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.cosine_similarity_model import CosineMatchingModel
from models.logistic_regression_model import LogisticMatchingModel
from models.neural_network_model import DeepMatchingNetwork
from models.random_forest_model import RandomForestMatchingModel
from models.weighted_knn_model import WeightedKNNMatcher
from src.feature_engineering_improved import ImprovedFeatureEngineer
from src.complete_source_code import DataLoader


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    train_time: float
    predict_time: float
    confusion_matrix: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    extra: Dict[str, object]


class ModelComparator:
    """Coordinates training and evaluation of all requested models."""

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.3,
        output_dir: Path | str = "results",
    ):
        self.random_state = random_state
        self.test_size = test_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.results: Dict[str, ModelResult] = {}
        self.feature_names: List[str] = []
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        loader = DataLoader(data_dir="data/raw/")
        data = loader.load_all_datasets()

        engineer = ImprovedFeatureEngineer(random_state=self.random_state)
        feature_set = engineer.create_features(data["master"], n_pairs_per_athlete=12)
        self.feature_names = feature_set.feature_names

        X_train, X_test, y_train, y_test = train_test_split(
            feature_set.X,
            feature_set.y,
            test_size=self.test_size,
            stratify=feature_set.y,
            random_state=self.random_state,
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        np.save(self.output_dir / "X_train.npy", X_train)
        np.save(self.output_dir / "X_test.npy", X_test)
        np.save(self.output_dir / "y_train.npy", y_train)
        np.save(self.output_dir / "y_test.npy", y_test)

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------ #
    # Training & evaluation
    # ------------------------------------------------------------------ #
    def train_and_evaluate(self) -> Dict[str, ModelResult]:
        X_train, X_test, y_train, y_test = self.prepare_data()

        # 1. Logistic Regression (baseline)
        logistic = LogisticMatchingModel(random_state=self.random_state)
        start = time.perf_counter()
        logistic.fit(X_train, y_train, self.feature_names)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        lr_proba = logistic.predict_proba(X_test)
        lr_pred = (lr_proba >= 0.5).astype(int)
        predict_time = time.perf_counter() - start
        lr_report = logistic.evaluate(X_test, y_test)
        logistic_metrics = self._format_metrics(lr_report.metrics)
        self.results["Logistic Regression"] = ModelResult(
            name="Logistic Regression",
            metrics=logistic_metrics,
            train_time=train_time,
            predict_time=predict_time,
            confusion_matrix=lr_report.confusion_matrix,
            probabilities=lr_proba,
            predictions=lr_pred,
            extra={
                "coefficients": lr_report.coefficients,
                "confusion_matrix": lr_report.confusion_matrix,
                "cv_scores": lr_report.roc_auc_cv_scores,
            },
        )

        # 2. Cosine Similarity
        cosine = CosineMatchingModel(threshold=0.65)
        start = time.perf_counter()
        cosine.fit(X_train, y_train, self.feature_names)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        cosine_proba = cosine.predict_proba(X_test)
        cosine_pred = cosine.predict(X_test)
        predict_time = time.perf_counter() - start
        cosine_report = cosine.evaluate(
            X_test,
            y_test,
            output_dir=self.output_dir / "cosine_similarity",
            logistic_metrics=logistic_metrics,
        )
        self.results["Cosine Similarity"] = ModelResult(
            name="Cosine Similarity",
            metrics=self._format_metrics(cosine_report.metrics),
            train_time=train_time,
            predict_time=predict_time,
            confusion_matrix=self._confusion(y_test, cosine_pred),
            probabilities=cosine_proba,
            predictions=cosine_pred,
            extra={"threshold_sweep": cosine_report.threshold_sweep},
        )

        # 3. Weighted KNN
        knn = WeightedKNNMatcher(random_state=self.random_state)
        start = time.perf_counter()
        knn.fit(
            X_train,
            y_train,
            feature_names=self.feature_names,
            logistic_coefficients=self.results["Logistic Regression"].extra["coefficients"],
        )
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        knn_proba = knn.predict_proba(X_test)
        knn_pred = knn.predict(X_test)
        predict_time = time.perf_counter() - start
        knn_report = knn.evaluate(X_test, y_test)
        self.results["Weighted KNN"] = ModelResult(
            name="Weighted KNN",
            metrics=self._format_metrics(knn_report.metrics),
            train_time=train_time,
            predict_time=predict_time,
            confusion_matrix=self._confusion(y_test, knn_pred),
            probabilities=knn_proba,
            predictions=knn_pred,
            extra={
                "best_k": knn_report.best_k,
                "feature_weights": knn_report.feature_weights,
                "confidence_scores": knn_report.confidence_scores,
            },
        )

        # 4. Random Forest
        rf = RandomForestMatchingModel(n_trees=200, random_state=self.random_state)
        start = time.perf_counter()
        rf.fit(X_train, y_train, feature_names=self.feature_names)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        rf_proba = rf.predict_proba(X_test)
        rf_pred = rf.predict(X_test)
        predict_time = time.perf_counter() - start
        rf_report = rf.evaluate(X_test, y_test)
        self.results["Random Forest"] = ModelResult(
            name="Random Forest",
            metrics=self._format_metrics(rf_report.metrics),
            train_time=train_time,
            predict_time=predict_time,
            confusion_matrix=self._confusion(y_test, rf_pred),
            probabilities=rf_proba,
            predictions=rf_pred,
            extra={
                "oob_score": rf_report.oob_score,
                "feature_importance": rf_report.feature_importance,
                "tree_depths": rf_report.tree_depths,
            },
        )

        # 5. Neural Network (scale features first)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        nn = DeepMatchingNetwork(input_dim=X_train.shape[1])
        start = time.perf_counter()
        nn.train(X_train_scaled, y_train, feature_names=self.feature_names)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        nn_proba = nn.predict_proba(X_test_scaled)
        nn_pred = nn.predict(X_test_scaled)
        predict_time = time.perf_counter() - start
        nn_report = nn.evaluate(
            X_test_scaled,
            y_test,
            output_dir=self.output_dir / "neural_network",
        )
        self.results["Neural Network"] = ModelResult(
            name="Neural Network",
            metrics=self._format_metrics(nn_report.metrics),
            train_time=train_time,
            predict_time=predict_time,
            confusion_matrix=nn_report.confusion_matrix,
            probabilities=nn_proba,
            predictions=nn_pred,
            extra={
                "history": nn_report.history,
                "feature_attributions": nn_report.feature_attributions,
                "threshold": nn_report.metrics.get("threshold", 0.5),
            },
        )

        return self.results

    # ------------------------------------------------------------------ #
    # Reporting utilities
    # ------------------------------------------------------------------ #
    def build_metrics_table(self) -> pd.DataFrame:
        rows = []
        for name, res in self.results.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy": res.metrics["accuracy"],
                    "Precision": res.metrics["precision"],
                    "Recall": res.metrics["recall"],
                    "F1": res.metrics["f1"],
                    "ROC-AUC": res.metrics["roc_auc"],
                }
            )
        table = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
        table.to_csv(self.output_dir / "comparison_table.csv", index=False)
        return table

    def build_speed_table(self) -> pd.DataFrame:
        rows = []
        for name, res in self.results.items():
            rows.append(
                {
                    "Model": name,
                    "Train Time (s)": res.train_time,
                    "Predict Time per 1000 (s)": res.predict_time * 1000,
                    "Interpretability": self._interpretability_score(name),
                }
            )
        table = pd.DataFrame(rows)
        table.to_csv(self.output_dir / "speed_table.csv", index=False)
        return table

    def _interpretability_score(self, name: str) -> str:
        mapping = {
            "Logistic Regression": "★★★★★ Excellent",
            "Cosine Similarity": "★★★★★ Excellent",
            "Weighted KNN": "★★★★☆ Very Good",
            "Random Forest": "★★★★☆ Very Good",
            "Neural Network": "★★☆☆☆ Limited",
        }
        return mapping.get(name, "★★★☆☆ Moderate")

    def plot_roc_curves(self) -> Path:
        plt.figure(figsize=(8, 6))
        for name, res in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, res.probabilities)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(self.y_test, res.probabilities):.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend()
        plt.tight_layout()
        path = self.output_dir / "roc_curves_comparison.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_confusion_matrices(self) -> Path:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        for ax, (name, res) in zip(axes, self.results.items()):
            ConfusionMatrixDisplay(res.confusion_matrix).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
            ax.set_title(name)
        fig.suptitle("Confusion Matrices", fontsize=16)
        plt.tight_layout()
        path = self.output_dir / "confusion_matrices_grid.png"
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_performance_radar(self) -> Path:
        metrics = ["accuracy", "precision", "recall", "roc_auc"]
        data = []
        for res in self.results.values():
            data.append([res.metrics[m] for m in metrics])
        df = pd.DataFrame(data, index=self.results.keys(), columns=metrics)

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        df = pd.concat([df, df.iloc[:, 0:1]], axis=1)
        angles = np.append(angles, angles[0])

        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, polar=True)
        for idx, (name, row) in enumerate(df.iterrows()):
            ax.plot(angles, row.values, label=name)
            ax.fill(angles, row.values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_yticks(np.linspace(0.5, 1.0, 6))
        ax.set_title("Performance Radar Chart")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        path = self.output_dir / "performance_radar.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def _format_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        return {k: float(v) for k, v in metrics.items()}

    def _confusion(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)

    def generate_pdf_reports(self) -> Path:
        pdf_path = self.output_dir / "Model_Comparison_Summary.pdf"
        with PdfPages(pdf_path) as pdf:
            # Slide 1: Metric table
            table = self.build_metrics_table()
            table_display = table.copy()
            numeric_cols = table_display.select_dtypes(include=[np.number]).columns
            table_display.loc[:, numeric_cols] = table_display.loc[:, numeric_cols].applymap(lambda x: round(float(x), 4))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis("off")
            ax.set_title("Model Performance Summary", fontsize=14, weight="bold")
            ax.table(
                cellText=table_display.values,
                colLabels=table_display.columns,
                loc="center",
                cellLoc="center",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Slide 2: ROC curves
            roc_path = self.plot_roc_curves()
            fig = plt.figure(figsize=(8, 6))
            img = plt.imread(roc_path)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Slide 3: Confusion matrices
            cm_path = self.plot_confusion_matrices()
            fig = plt.figure(figsize=(10, 7))
            img = plt.imread(cm_path)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Slide 4: Radar chart
            radar_path = self.plot_performance_radar()
            fig = plt.figure(figsize=(8, 6))
            img = plt.imread(radar_path)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        return pdf_path

# noinspection PyAttributeOutsideInit
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def main(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare multiple matching models.")
    parser.add_argument("--output_dir", default="results", type=str, help="Directory for outputs.")
    ns = parser.parse_args(args) if args is not None else parser.parse_args()

    comparator = ModelComparator(output_dir=ns.output_dir)
    results = comparator.train_and_evaluate()

    metrics_table = comparator.build_metrics_table()
    comparator.build_speed_table()
    comparator.plot_roc_curves()
    comparator.plot_confusion_matrices()
    comparator.plot_performance_radar()
    comparator.generate_pdf_reports()

    # Save raw metrics json
    metrics_payload = {
        name: {
            "metrics": res.metrics,
            "train_time": res.train_time,
            "predict_time": res.predict_time,
        }
        for name, res in results.items()
    }
    with open(comparator.output_dir / "performance_metrics.json", "w") as fh:
        json.dump(metrics_payload, fh, indent=2)

    print("✅ Model comparison complete.")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()

