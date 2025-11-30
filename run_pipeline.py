"""
Improved Athlete Peer Matching Pipeline
======================================

This script implements the fixes outlined in the diagnostic report:
    - Feature engineering with measurable variance
    - Stratified train/test split + SMOTE for imbalance
    - Stronger regularisation and cross-validation
    - Probability calibration for realistic match scores
    - Richer diagnostics (feature stats, CV scores, sample predictions)
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from src.complete_source_code import DataLoader  # type: ignore  # noqa: E402
from src.feature_engineering_improved import (  # type: ignore  # noqa: E402
    FeatureSet,
    ImprovedFeatureEngineer,
)


def print_header(text: str) -> None:
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def ensure_dirs(paths: List[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_feature_outputs(feature_set: FeatureSet) -> None:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    ensure_dirs([processed_dir])

    feature_set.df_pairs.to_csv(
        processed_dir / "athlete_pairs_features.csv",
        index=False,
    )

    results_dir = PROJECT_ROOT / "results"
    ensure_dirs([results_dir])
    feature_set.feature_stats.to_csv(
        results_dir / "feature_variance.csv",
        index=True,
    )


def describe_feature_variance(feature_set: FeatureSet) -> None:
    print_header("FEATURE VARIANCE DIAGNOSTICS")
    print("Feature | mean | std | min | max | unique values")
    print("-" * 80)
    for feature, row in feature_set.feature_stats.iterrows():
        status = "OK" if row["std"] > 0.05 else "LOW VAR"
        print(
            f"{feature:30s} | mean={row['mean']:.3f} | std={row['std']:.4f} | "
            f"[{row['min']:.2f}, {row['max']:.2f}] | unique={int(row['unique_values'])} | {status}"
        )


def cross_validate_model(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int
) -> Dict[str, object]:
    cv_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=0.1,
                ),
            ),
        ]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        cv_pipeline, X_train, y_train, cv=skf, scoring="roc_auc"
    )
    return {
        "roc_auc_cv_mean": float(scores.mean()),
        "roc_auc_cv_std": float(scores.std()),
        "roc_auc_cv_folds": scores,
    }


def train_and_calibrate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> Dict[str, object]:
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_balanced_scaled = scaler.fit_transform(X_balanced)

    base_model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        C=0.1,
    )
    base_model.fit(X_balanced_scaled, y_balanced)

    calibrated_model = CalibratedClassifierCV(
        estimator=LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            C=0.1,
        ),
        cv=5,
        method="sigmoid",
    )
    calibrated_model.fit(X_balanced_scaled, y_balanced)

    return {
        "scaler": scaler,
        "base_model": base_model,
        "calibrated_model": calibrated_model,
        "X_balanced": X_balanced,
        "y_balanced": y_balanced,
    }


def evaluate_model(
    calibrated_model: CalibratedClassifierCV,
    base_model: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Dict[str, object]:
    X_test_scaled = scaler.transform(X_test)
    y_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "y_test": y_test,
    }

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": base_model.coef_[0],
        }
    )
    feature_importance["abs_coefficient"] = feature_importance["coefficient"].abs()
    feature_importance.sort_values("abs_coefficient", ascending=False, inplace=True)

    return {
        "metrics": metrics,
        "feature_importance": feature_importance,
    }


def save_evaluation_outputs(
    evaluation: Dict[str, object],
    feature_set: FeatureSet,
    calibrated_model: CalibratedClassifierCV,
    base_model: LogisticRegression,
    scaler: StandardScaler,
    roc_auc_cv: Dict[str, object],
) -> None:
    results_dir = PROJECT_ROOT / "results"
    ensure_dirs([results_dir])

    metrics = evaluation["metrics"]
    perf_payload = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "roc_auc_cv_mean": roc_auc_cv["roc_auc_cv_mean"],
        "roc_auc_cv_std": roc_auc_cv["roc_auc_cv_std"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    pd.DataFrame([perf_payload]).to_csv(
        results_dir / "performance_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        metrics["confusion_matrix"],
        columns=["Predicted_0", "Predicted_1"],
        index=["Actual_0", "Actual_1"],
    ).to_csv(results_dir / "confusion_matrix.csv")

    evaluation["feature_importance"].to_csv(
        results_dir / "feature_importance.csv",
        index=False,
    )

    X_all_scaled = scaler.transform(feature_set.X)
    proba_all = calibrated_model.predict_proba(X_all_scaled)[:, 1]
    pred_all = (proba_all >= 0.5).astype(int)

    predictions_df = feature_set.df_pairs[
        ["athlete1_id", "athlete2_id", "label", "true_match_score"]
    ].copy()
    predictions_df.rename(
        columns={
            "label": "actual_label",
            "true_match_score": "heuristic_match_score",
        },
        inplace=True,
    )
    predictions_df["predicted_probability"] = proba_all
    predictions_df["predicted_label"] = pred_all
    predictions_df.to_csv(results_dir / "predictions.csv", index=False)

    models_dir = PROJECT_ROOT / "models"
    ensure_dirs([models_dir])
    joblib.dump(
        {
            "scaler": scaler,
            "base_model": base_model,
            "calibrated_model": calibrated_model,
            "feature_names": feature_set.feature_names,
        },
        models_dir / "logistic_model.pkl",
    )


def generate_visualisations(
    evaluation: Dict[str, object],
) -> None:
    visual_dir = PROJECT_ROOT / "visualizations"
    ensure_dirs([visual_dir])

    sns.set_style("whitegrid")
    metrics = evaluation["metrics"]

    # ROC Curve
    fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_proba"])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Calibrated ROC (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Improved Peer Matching")
    plt.legend()
    plt.tight_layout()
    plt.savefig(visual_dir / "roc_curve.png", dpi=300)
    plt.close()

    # Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=evaluation["feature_importance"].head(10),
        x="coefficient",
        y="feature",
        palette="viridis",
    )
    plt.title("Top 10 Feature Importance (Logistic Coefficients)")
    plt.xlabel("Coefficient")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(visual_dir / "feature_importance.png", dpi=300)
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(visual_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # Match probability distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(metrics["y_proba"], bins=20, kde=True, color="teal")
    plt.title("Distribution of Calibrated Match Probabilities (Test Set)")
    plt.xlabel("Match Probability")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(visual_dir / "match_distribution.png", dpi=300)
    plt.close()


def sample_predictions(y_test: np.ndarray, y_proba: np.ndarray) -> None:
    print_header("SAMPLE CALIBRATED PREDICTIONS")
    limit = min(5, len(y_test))
    for idx in range(limit):
        actual = "MATCH" if y_test[idx] == 1 else "NO MATCH"
        probability = y_proba[idx]
        predicted = "MATCH" if probability >= 0.5 else "NO MATCH"
        print(
            f"{idx+1}. Actual: {actual:8s} | Predicted: {predicted:8s} | "
            f"Probability: {probability:.2%}"
        )


def main() -> int:
    random_state = 42

    print_header("IMPROVED ATHLETE PEER MATCHING MODEL")
    print(f"Execution started: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # STEP 1: DATA LOADING
    print_header("STEP 1: LOAD DATA")
    loader = DataLoader(data_dir="data/raw/")
    data = loader.load_all_datasets()

    print("[OK] Datasets loaded:")
    for name, df in data.items():
        print(f"  - {name.capitalize():<12s}: {len(df):5d} rows")

    # STEP 2: IMPROVED FEATURE ENGINEERING
    print_header("STEP 2: IMPROVED FEATURE ENGINEERING")
    engineer = ImprovedFeatureEngineer(random_state=random_state)
    feature_set = engineer.create_features(data["master"], n_pairs_per_athlete=12)

    print(
        f"[OK] Engineered {len(feature_set.X)} athlete pairs with "
        f"{len(feature_set.feature_names)} features"
    )
    class_pos = int(feature_set.y.sum())
    class_neg = int(len(feature_set.y) - class_pos)
    print(
        f"  - Positive matches: {class_pos} ({class_pos/len(feature_set.y):.1%})\n"
        f"  - Negative matches: {class_neg} ({class_neg/len(feature_set.y):.1%})"
    )

    describe_feature_variance(feature_set)
    save_feature_outputs(feature_set)

    # STEP 3: MODEL TRAINING (WITH DIAGNOSTICS)
    print_header("STEP 3: MODEL TRAINING & CROSS-VALIDATION")
    X_train, X_test, y_train, y_test = train_test_split(
        feature_set.X,
        feature_set.y,
        test_size=0.3,
        stratify=feature_set.y,
        random_state=random_state,
    )

    print(
        f"[OK] Stratified split - train: {len(X_train)} | test: {len(X_test)} "
        f"(class balance preserved)"
    )
    roc_auc_cv = cross_validate_model(X_train, y_train, random_state)
    print(
        f"  - 5-fold ROC-AUC: {roc_auc_cv['roc_auc_cv_mean']:.3f} "
        f"(+/- {roc_auc_cv['roc_auc_cv_std']:.3f})"
    )

    artefacts = train_and_calibrate(X_train, y_train, random_state)

    print(
        f"  - SMOTE up-sampled training set from {len(X_train)} to "
        f"{len(artefacts['X_balanced'])} samples"
    )

    # STEP 4: MODEL EVALUATION
    print_header("STEP 4: MODEL EVALUATION")
    evaluation = evaluate_model(
        artefacts["calibrated_model"],
        artefacts["base_model"],
        artefacts["scaler"],
        X_test,
        y_test,
        feature_set.feature_names,
    )

    metrics = evaluation["metrics"]
    print("[OK] Test set performance:")
    print(f"  - Accuracy : {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  - Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.2f}%)")
    print(f"  - Recall   : {metrics['recall']:.3f} ({metrics['recall']*100:.2f}%)")
    print(f"  - F1-Score : {metrics['f1']:.3f}")
    print(f"  - ROC-AUC  : {metrics['roc_auc']:.3f}")

    print("\nConfusion matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  - True Negatives : {cm[0,0]:3d}")
    print(f"  - False Positives: {cm[0,1]:3d}")
    print(f"  - False Negatives: {cm[1,0]:3d}")
    print(f"  - True Positives : {cm[1,1]:3d}")

    print("\nTop feature contributions:")
    for _, row in evaluation["feature_importance"].head(10).iterrows():
        direction = "increase" if row["coefficient"] > 0 else "decrease"
        print(
            f"  - {row['feature']:30s} | {row['coefficient']:+.3f} "
            f"(tends to {direction} match odds)"
        )

    sample_predictions(y_test, metrics["y_proba"])

    # STEP 5: SAVE OUTPUTS & VISUALS
    print_header("STEP 5: SAVE RESULTS & VISUALISATIONS")
    save_evaluation_outputs(
        evaluation,
        feature_set,
        artefacts["calibrated_model"],
        artefacts["base_model"],
        artefacts["scaler"],
        roc_auc_cv,
    )
    generate_visualisations(evaluation)
    print("[OK] Outputs saved to `data/processed`, `results`, `models`, and `visualizations`.")

    print_header("PIPELINE COMPLETE")
    print(f"Execution finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Generated artefacts:")
    print("  - data/processed/athlete_pairs_features.csv")
    print("  - results/performance_metrics.csv")
    print("  - results/feature_variance.csv")
    print("  - results/feature_importance.csv")
    print("  - results/predictions.csv")
    print("  - results/confusion_matrix.csv")
    print("  - models/logistic_model.pkl")
    print("  - visualizations/[roc_curve|feature_importance|confusion_matrix|match_distribution].png")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

