"""
Retrain Existing Models with New 1,000-Athlete Dataset
=======================================================

Retrains all existing models from the models/ folder using the new large-scale dataset.

Existing models:
- Logistic Regression
- Random Forest
- Neural Network
- Cosine Similarity
- Weighted KNN
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from train_models import UnifiedDataLoader, FeatureEngineer, generate_pairwise_dataset

# Import existing models
from models.logistic_regression_model import LogisticMatchingModel
from models.random_forest_model import RandomForestMatchingModel
from models.neural_network_model import DeepMatchingNetwork
from models.cosine_similarity_model import CosineMatchingModel
from models.weighted_knn_model import WeightedKNNMatcher

warnings.filterwarnings("ignore")


def load_split_data():
    """Load train/val/test splits using saved manifest."""
    print("Loading unified dataset and splits...")
    
    # Load split manifest
    with open('models/split_manifest.json', 'r') as f:
        split_manifest = json.load(f)
    
    # Load unified data
    loader = UnifiedDataLoader()
    datasets = loader.load_all_datasets()
    unified_df = loader.merge_datasets(datasets)
    
    # Split based on manifest
    train_df = unified_df[unified_df['athlete_id'].isin(split_manifest['train_athlete_ids'])]
    val_df = unified_df[unified_df['athlete_id'].isin(split_manifest['val_athlete_ids'])]
    test_df = unified_df[unified_df['athlete_id'].isin(split_manifest['test_athlete_ids'])]
    
    print(f"✓ Loaded data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    return train_df, val_df, test_df


def prepare_features():
    """Prepare features using saved feature pipeline."""
    print("\nPreparing features...")
    
    train_df, val_df, test_df = load_split_data()
    
    # Load feature pipeline
    feature_engineer = joblib.load('models/feature_pipeline.pkl')
    
    X_train = feature_engineer.transform(train_df)
    X_val = feature_engineer.transform(val_df)
    X_test = feature_engineer.transform(test_df)
    
    # Generate pairwise datasets for binary classification
    print("\nGenerating pairwise datasets...")
    X_pairs_train, y_compat_train, _ = generate_pairwise_dataset(train_df, n_pairs_per_athlete=50)
    X_pairs_val, y_compat_val, _ = generate_pairwise_dataset(val_df, n_pairs_per_athlete=50)
    X_pairs_test, y_compat_test, _ = generate_pairwise_dataset(test_df, n_pairs_per_athlete=50)
    
    # Convert to binary labels (threshold at 0.55)
    y_train_binary = (y_compat_train > 0.55).astype(int)
    y_val_binary = (y_compat_val > 0.55).astype(int)
    y_test_binary = (y_compat_test > 0.55).astype(int)
    
    print(f"✓ Pairwise datasets:")
    print(f"  - Train: {len(X_pairs_train)} pairs, {y_train_binary.sum()} positive")
    print(f"  - Val: {len(X_pairs_val)} pairs, {y_val_binary.sum()} positive")
    print(f"  - Test: {len(X_pairs_test)} pairs, {y_test_binary.sum()} positive")
    
    return {
        'X_pairs_train': X_pairs_train,
        'y_train': y_train_binary,
        'X_pairs_val': X_pairs_val,
        'y_val': y_val_binary,
        'X_pairs_test': X_pairs_test,
        'y_test': y_test_binary,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test
    }


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model."""
    print(f"\nEvaluating {model_name}...")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC: {auc:.4f}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def main():
    """Main training script."""
    print("=" * 80)
    print("RETRAINING EXISTING MODELS WITH 1,000-ATHLETE DATASET")
    print("=" * 80)
    
    # Prepare data
    data = prepare_features()
    
    # Results storage
    all_results = {}
    
    # ========================================================================
    # MODEL 1: Logistic Regression
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 80)
    
    # Generate feature names for pairwise features
    pair_feature_names = [f'feature_{i}' for i in range(data['X_pairs_train'].shape[1])]
    
    logistic_model = LogisticMatchingModel()
    logistic_model.fit(data['X_pairs_train'], data['y_train'], pair_feature_names)
    
    # Evaluate
    logistic_results = evaluate_model(
        logistic_model, 
        data['X_pairs_test'], 
        data['y_test'],
        "Logistic Regression"
    )
    all_results['logistic_regression'] = logistic_results
    
    # Save
    joblib.dump(logistic_model, 'models/logistic_model.pkl')
    print("✓ Saved logistic_model.pkl")
    
    # ========================================================================
    # MODEL 2: Random Forest
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 80)
    
    rf_model = RandomForestMatchingModel()
    rf_model.fit(data['X_pairs_train'], data['y_train'], pair_feature_names)
    
    # Evaluate
    rf_results = evaluate_model(
        rf_model,
        data['X_pairs_test'],
        data['y_test'],
        "Random Forest"
    )
    all_results['random_forest'] = rf_results
    
    # Save
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print("✓ Saved random_forest_model.pkl")
    
    # ========================================================================
    # MODEL 3: Neural Network
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING NEURAL NETWORK MODEL")
    print("=" * 80)
    
    nn_model = DeepMatchingNetwork(
        input_dim=data['X_pairs_train'].shape[1],
        max_epochs=50,
        batch_size=128
    )
    nn_model.fit(
        data['X_pairs_train'], 
        data['y_train'],
        feature_names=pair_feature_names
    )
    
    # Evaluate
    nn_results = evaluate_model(
        nn_model,
        data['X_pairs_test'],
        data['y_test'],
        "Neural Network"
    )
    all_results['neural_network'] = nn_results
    
    # Save
    joblib.dump(nn_model, 'models/neural_network_model.pkl')
    print("✓ Saved neural_network_model.pkl")
    
    # ========================================================================
    # MODEL 4: Cosine Similarity
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COSINE SIMILARITY MODEL")
    print("=" * 80)
    
    cosine_model = CosineMatchingModel()
    # Cosine model doesn't need training, just needs to fit scaler
    cosine_model.fit(data['X_pairs_train'], data['y_train'])
    
    # Evaluate
    cosine_results = evaluate_model(
        cosine_model,
        data['X_pairs_test'],
        data['y_test'],
        "Cosine Similarity"
    )
    all_results['cosine_similarity'] = cosine_results
    
    # Save
    joblib.dump(cosine_model, 'models/cosine_similarity_model.pkl')
    print("✓ Saved cosine_similarity_model.pkl")
    
    # ========================================================================
    # MODEL 5: Weighted KNN
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING WEIGHTED KNN MODEL")
    print("=" * 80)
    
    knn_model = WeightedKNNMatcher(n_neighbors=15)
    knn_model.fit(data['X_pairs_train'], data['y_train'])
    
    # Evaluate
    knn_results = evaluate_model(
        knn_model,
        data['X_pairs_test'],
        data['y_test'],
        "Weighted KNN"
    )
    all_results['weighted_knn'] = knn_results
    
    # Save
    joblib.dump(knn_model, 'models/weighted_knn_model.pkl')
    print("✓ Saved weighted_knn_model.pkl")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    # Create comparison table
    comparison_df = pd.DataFrame(all_results).T
    print("\n")
    print(comparison_df.to_string())
    
    # Save results
    with open('models/existing_models_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    comparison_df.to_csv('models/existing_models_comparison.csv')
    
    print("\n✓ Saved existing_models_results.json")
    print("✓ Saved existing_models_comparison.csv")
    
    # Find best model
    best_model = comparison_df['f1'].idxmax()
    best_f1 = comparison_df['f1'].max()
    
    print(f"\n🏆 Best performing model: {best_model} (F1: {best_f1:.4f})")
    
    print("\n" + "=" * 80)
    print("✅ ALL EXISTING MODELS RETRAINED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
