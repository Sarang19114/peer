"""
Complete ML Training Pipeline for Athlete Matchmaking System
=============================================================

Trains all core models using the 1,000-athlete dataset (7 CSV files):
- Compatibility Model (XGBoost Regressor)
- Longevity Model (Classification)
- Experience Tier Classifier
- Training-Style Embedding Model (Siamese Network)
- Existing models (Logistic, RandomForest, Neural Network, etc.)

Uses athlete-stratified 65/20/15 train/val/test split to prevent data leakage.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, roc_auc_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

warnings.filterwarnings("ignore")

# ============================================================================
# DATA LOADING & MERGING
# ============================================================================

class UnifiedDataLoader:
    """
    Loads and merges all 7 CSV files into a unified training table.
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = Path(data_dir)
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all 7 CSV files."""
        print("Loading all datasets...")
        
        datasets = {
            'master': pd.read_csv(self.data_dir / 'athlete_master_profiles.csv'),
            'health': pd.read_csv(self.data_dir / 'daily_health_metrics.csv'),
            'activities': pd.read_csv(self.data_dir / 'strava_activities.csv'),
            'injuries': pd.read_csv(self.data_dir / 'injury_medical_history.csv'),
            'mood': pd.read_csv(self.data_dir / 'mood_tracking.csv'),
            'social': pd.read_csv(self.data_dir / 'social_communication.csv'),
            'timeline': pd.read_csv(self.data_dir / 'life_timeline_career.csv'),
        }
        
        print(f"✓ Loaded {len(datasets)} datasets")
        for name, df in datasets.items():
            print(f"  - {name}: {len(df)} records")
        
        return datasets
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all datasets into unified table with aggregations.
        
        Returns unified dataframe with:
        - Demographics from master
        - Long-term health averages (7-day, 30-day rolling windows)
        - Activity aggregations (distance, pace, HR, duration)
        - Injury frequency + severity scores
        - Mood & stress averages
        - Social/communication pattern metrics
        - Career timeline encoded features
        """
        print("\nMerging datasets...")
        
        # Start with master profiles
        unified = datasets['master'].copy()
        unified['athlete_id'] = unified['athlete_id'].astype(str)
        
        # === HEALTH METRICS AGGREGATIONS ===
        health = datasets['health'].copy()
        health['athlete_id'] = health['athlete_id'].astype(str)
        health['date'] = pd.to_datetime(health['date'])
        
        # Aggregate health metrics per athlete
        health_agg = health.groupby('athlete_id').agg({
            'resting_heart_rate': ['mean', 'std', 'min', 'max'],
            'heart_rate_variability': ['mean', 'std'],
            'sleep_hours': ['mean', 'std'],
            'recovery_score': ['mean', 'std'],
            'stress_level': ['mean', 'std'],
            'mood_score': ['mean', 'std'],
            'weight_kg': ['mean', 'std'],
            'calories_burned': ['mean', 'sum'],
            'daily_steps': ['mean', 'sum'],
        }).reset_index()
        
        health_agg.columns = ['athlete_id'] + [
            f'health_{col[0]}_{col[1]}' for col in health_agg.columns[1:]
        ]
        
        # Calculate 7-day and 30-day rolling averages for recent trends
        health_sorted = health.sort_values(['athlete_id', 'date'])
        health_sorted['recovery_7d'] = health_sorted.groupby('athlete_id')['recovery_score'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        health_sorted['recovery_30d'] = health_sorted.groupby('athlete_id')['recovery_score'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )
        health_sorted['hrv_7d'] = health_sorted.groupby('athlete_id')['heart_rate_variability'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        
        # Get most recent rolling values per athlete
        health_recent = health_sorted.groupby('athlete_id').last()[
            ['recovery_7d', 'recovery_30d', 'hrv_7d']
        ].reset_index()
        
        unified = unified.merge(health_agg, on='athlete_id', how='left')
        unified = unified.merge(health_recent, on='athlete_id', how='left')
        
        # === ACTIVITY AGGREGATIONS ===
        activities = datasets['activities'].copy()
        activities['athlete_id'] = activities['athlete_id'].astype(str)
        activities['date'] = pd.to_datetime(activities['date'])
        
        activity_agg = activities.groupby('athlete_id').agg({
            'distance_km': ['sum', 'mean', 'std', 'max'],
            'duration_minutes': ['sum', 'mean', 'std'],
            'avg_speed_km_h': ['mean', 'std'],
            'avg_heart_rate': ['mean', 'std'],
            'max_heart_rate': ['mean', 'max'],
            'elevation_gain_m': ['sum', 'mean'],
            'calories_burned': ['sum', 'mean'],
            'activity_type': 'count',  # Total number of activities
        }).reset_index()
        
        activity_agg.columns = ['athlete_id'] + [
            f'activity_{col[0]}_{col[1]}' if col[1] != '' else f'activity_count'
            for col in activity_agg.columns[1:]
        ]
        
        # Calculate performance scores
        activities['performance_score'] = (
            activities['avg_speed_km_h'] * 0.4 +
            activities['distance_km'] * 0.3 +
            (activities['duration_minutes'] / 60) * 0.3
        )
        
        perf_agg = activities.groupby('athlete_id').agg({
            'performance_score': ['mean', 'std', 'max']
        }).reset_index()
        perf_agg.columns = ['athlete_id', 'performance_mean', 'performance_std', 'performance_max']
        
        # Activity consistency index (std/mean ratio)
        activity_agg['activity_consistency'] = (
            activity_agg['activity_distance_km_std'] / 
            (activity_agg['activity_distance_km_mean'] + 1e-6)
        )
        
        unified = unified.merge(activity_agg, on='athlete_id', how='left')
        unified = unified.merge(perf_agg, on='athlete_id', how='left')
        
        # === INJURY AGGREGATIONS ===
        injuries = datasets['injuries'].copy()
        injuries['athlete_id'] = injuries['athlete_id'].astype(str)
        
        # Map severity to numeric
        severity_map = {'minor': 1, 'moderate': 2, 'severe': 3}
        injuries['severity_score'] = injuries['severity'].map(severity_map).fillna(1)
        
        injury_agg = injuries.groupby('athlete_id').agg({
            'injury_type': 'count',  # Total injuries
            'severity_score': ['mean', 'sum', 'max'],
            'recovery_days': ['mean', 'sum', 'max'],
        }).reset_index()
        
        injury_agg.columns = ['athlete_id', 'injury_count', 'injury_severity_mean', 
                              'injury_severity_total', 'injury_severity_max',
                              'recovery_days_mean', 'recovery_days_total', 'recovery_days_max']
        
        # Injury risk score (frequency * severity * recovery time)
        injury_agg['injury_risk_score'] = (
            injury_agg['injury_count'] * 
            injury_agg['injury_severity_mean'] * 
            (injury_agg['recovery_days_mean'] / 10)
        )
        
        unified = unified.merge(injury_agg, on='athlete_id', how='left')
        
        # === MOOD TRACKING AGGREGATIONS ===
        mood = datasets['mood'].copy()
        mood['athlete_id'] = mood['athlete_id'].astype(str)
        
        mood_agg = mood.groupby('athlete_id').agg({
            'overall_mood': ['mean', 'std', 'min', 'max'],
            'stress_level': ['mean', 'std', 'max'],
            'motivation_level': ['mean', 'std'],
            'energy_level': ['mean', 'std'],
            'fatigue_level': ['mean', 'std'],
            'soreness_level': ['mean', 'std'],
            'mental_clarity': ['mean', 'std'],
        }).reset_index()
        
        mood_agg.columns = ['athlete_id'] + [
            f'mood_{col[0]}_{col[1]}' for col in mood_agg.columns[1:]
        ]
        
        # Emotional stability score (inverse of std)
        mood_agg['emotional_stability'] = 1 / (mood_agg['mood_overall_mood_std'] + 0.1)
        
        unified = unified.merge(mood_agg, on='athlete_id', how='left')
        
        # === SOCIAL/COMMUNICATION AGGREGATIONS ===
        social = datasets['social'].copy()
        social['athlete_id'] = social['athlete_id'].astype(str)
        
        # Encode sentiment to numeric
        sentiment_map = {
            'Positive': 1.0,
            'Supportive': 0.8,
            'Motivational': 0.9,
            'Neutral': 0.5,
            'Negative': 0.2
        }
        social['sentiment_score'] = social['sentiment'].map(sentiment_map).fillna(0.5)
        
        social_agg = social.groupby('athlete_id').agg({
            'communication_type': 'count',  # Total interactions
            'duration_minutes': ['sum', 'mean'],
            'sentiment_score': ['mean', 'std'],
        }).reset_index()
        
        social_agg.columns = ['athlete_id', 'social_interaction_count', 
                              'social_duration_total', 'social_duration_mean',
                              'social_sentiment_mean', 'social_sentiment_std']
        
        # Communication frequency score
        social_agg['communication_frequency'] = social_agg['social_interaction_count'] / 365  # per day
        
        unified = unified.merge(social_agg, on='athlete_id', how='left')
        
        # === TIMELINE/CAREER AGGREGATIONS ===
        timeline = datasets['timeline'].copy()
        timeline['athlete_id'] = timeline['athlete_id'].astype(str)
        
        timeline_agg = timeline.groupby('athlete_id').agg({
            'event_type': 'count',  # Total life events
        }).reset_index()
        timeline_agg.columns = ['athlete_id', 'life_event_count']
        
        # Career milestone encoding
        timeline['is_major_milestone'] = timeline['event_type'].isin([
            'competition_win', 'personal_record', 'qualification'
        ]).astype(int)
        
        milestone_agg = timeline.groupby('athlete_id').agg({
            'is_major_milestone': 'sum'
        }).reset_index()
        milestone_agg.columns = ['athlete_id', 'major_milestone_count']
        
        unified = unified.merge(timeline_agg, on='athlete_id', how='left')
        unified = unified.merge(milestone_agg, on='athlete_id', how='left')
        
        # Fill NaN values for athletes with no data in certain categories
        unified = unified.fillna(0)
        
        print(f"✓ Merged unified table: {len(unified)} athletes, {len(unified.columns)} features")
        
        return unified


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """
    Comprehensive feature engineering with:
    - Normalization of continuous metrics
    - One-hot encoding for categorical features
    - Location embeddings
    - Derived features (injury risk, consistency index)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = ['gender', 'primary_sport', 'competition_level', 
                                     'location', 'coaching_support', 'communication_preference',
                                     'career_status']
        self.continuous_features = []
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Engineer features from unified dataframe.
        
        Returns:
            X: Feature matrix
            feature_names: List of feature names
        """
        print("\nEngineering features...")
        
        df_processed = df.copy()
        
        # Identify continuous features (numeric columns excluding IDs)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        exclude_cols = ['athlete_id']
        self.continuous_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # One-hot encode categorical features
        encoded_dfs = []
        for cat_col in self.categorical_features:
            if cat_col in df_processed.columns:
                dummies = pd.get_dummies(df_processed[cat_col], prefix=cat_col, drop_first=False)
                encoded_dfs.append(dummies)
        
        # Normalize continuous features
        X_continuous = df_processed[self.continuous_features].values
        X_continuous_scaled = self.scaler.fit_transform(X_continuous)
        
        # Combine all features
        X_categorical = pd.concat(encoded_dfs, axis=1).values if encoded_dfs else np.array([]).reshape(len(df_processed), 0)
        X = np.hstack([X_continuous_scaled, X_categorical])
        
        # Generate feature names
        self.feature_names = self.continuous_features.copy()
        for cat_col in self.categorical_features:
            if cat_col in df_processed.columns:
                categories = df_processed[cat_col].unique()
                self.feature_names.extend([f"{cat_col}_{cat}" for cat in categories])
        
        print(f"✓ Engineered {X.shape[1]} features ({len(self.continuous_features)} continuous, {X.shape[1] - len(self.continuous_features)} categorical)")
        
        return X, self.feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted parameters."""
        df_processed = df.copy()
        
        # One-hot encode categorical
        encoded_dfs = []
        for cat_col in self.categorical_features:
            if cat_col in df_processed.columns:
                dummies = pd.get_dummies(df_processed[cat_col], prefix=cat_col, drop_first=False)
                encoded_dfs.append(dummies)
        
        # Normalize continuous
        X_continuous = df_processed[self.continuous_features].values
        X_continuous_scaled = self.scaler.transform(X_continuous)
        
        # Combine
        X_categorical = pd.concat(encoded_dfs, axis=1).values if encoded_dfs else np.array([]).reshape(len(df_processed), 0)
        X = np.hstack([X_continuous_scaled, X_categorical])
        
        return X


def generate_pair_features(athlete_a: pd.Series, athlete_b: pd.Series) -> Dict[str, float]:
    """
    Generate pairwise compatibility features for two athletes.
    
    Used for inference when matching new pairs.
    
    Args:
        athlete_a: First athlete's feature series
        athlete_b: Second athlete's feature series
        
    Returns:
        Dictionary of pairwise features
    """
    features = {}
    
    # Age similarity (normalized difference)
    features['age_diff'] = abs(athlete_a['age'] - athlete_b['age']) / 50.0
    features['age_similarity'] = 1 - features['age_diff']
    
    # Gender match
    features['gender_match'] = 1.0 if athlete_a['gender'] == athlete_b['gender'] else 0.0
    
    # Sport compatibility
    features['sport_match'] = 1.0 if athlete_a['primary_sport'] == athlete_b['primary_sport'] else 0.0
    
    # Experience similarity
    years_a = athlete_a.get('years_training', athlete_a.get('years_active', 5))
    years_b = athlete_b.get('years_training', athlete_b.get('years_active', 5))
    features['experience_diff'] = abs(years_a - years_b) / 20.0
    features['experience_similarity'] = 1 - features['experience_diff']
    
    # Training philosophy match (use competition_level as proxy if training_philosophy missing)
    phil_a = athlete_a.get('training_philosophy', athlete_a.get('competition_level', 'unknown'))
    phil_b = athlete_b.get('training_philosophy', athlete_b.get('competition_level', 'unknown'))
    features['training_philosophy_match'] = 1.0 if phil_a == phil_b else 0.0
    
    # Location match (simplified - same city = 1, else 0)
    features['location_match'] = 1.0 if athlete_a['location'] == athlete_b['location'] else 0.0
    
    # Health metrics alignment
    if 'health_recovery_score_mean' in athlete_a:
        features['recovery_similarity'] = 1 - abs(
            athlete_a['health_recovery_score_mean'] - athlete_b['health_recovery_score_mean']
        ) / 100.0
    
    if 'health_resting_heart_rate_mean' in athlete_a:
        features['hr_similarity'] = 1 - abs(
            athlete_a['health_resting_heart_rate_mean'] - athlete_b['health_resting_heart_rate_mean']
        ) / 100.0
    
    # Activity level alignment
    if 'activity_distance_km_mean' in athlete_a:
        features['activity_level_similarity'] = 1 - abs(
            athlete_a['activity_distance_km_mean'] - athlete_b['activity_distance_km_mean']
        ) / 50.0
    
    # Performance alignment
    if 'performance_mean' in athlete_a:
        features['performance_similarity'] = 1 - abs(
            athlete_a['performance_mean'] - athlete_b['performance_mean']
        ) / 20.0
    
    # Social compatibility
    if 'social_sentiment_mean' in athlete_a:
        features['social_sentiment_similarity'] = 1 - abs(
            athlete_a['social_sentiment_mean'] - athlete_b['social_sentiment_mean']
        ) / 10.0
    
    # Mood stability alignment
    if 'mood_overall_mood_mean' in athlete_a:
        features['mood_similarity'] = 1 - abs(
            athlete_a['mood_overall_mood_mean'] - athlete_b['mood_overall_mood_mean']
        ) / 10.0
    
    # Network size similarity
    if 'social_network_size' in athlete_a:
        features['network_size_similarity'] = 1 - abs(
            athlete_a['social_network_size'] - athlete_b['social_network_size']
        ) / 200.0
    
    # Clip all features to [0, 1] range
    for key in features:
        features[key] = np.clip(features[key], 0, 1)
    
    return features


# ============================================================================
# DATASET SPLITTING
# ============================================================================

def create_stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.65,
    val_size: float = 0.20,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Create athlete-stratified train/val/test split.
    
    Ensures no athlete appears in multiple splits to prevent data leakage.
    
    Args:
        df: Unified athlete dataframe
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df, split_manifest
    """
    print(f"\nCreating stratified split ({train_size:.0%}/{val_size:.0%}/{test_size:.0%})...")
    
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
    
    athlete_ids = df['athlete_id'].unique()
    n_athletes = len(athlete_ids)
    
    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        athlete_ids,
        test_size=(val_size + test_size),
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1 - val_ratio),
        random_state=random_state
    )
    
    # Create dataframes
    train_df = df[df['athlete_id'].isin(train_ids)].copy()
    val_df = df[df['athlete_id'].isin(val_ids)].copy()
    test_df = df[df['athlete_id'].isin(test_ids)].copy()
    
    # Create manifest
    split_manifest = {
        'train_athlete_ids': train_ids.tolist(),
        'val_athlete_ids': val_ids.tolist(),
        'test_athlete_ids': test_ids.tolist(),
        'split_date': datetime.now().isoformat(),
        'random_state': random_state,
        'sizes': {
            'train': len(train_ids),
            'val': len(val_ids),
            'test': len(test_ids)
        }
    }
    
    print(f"✓ Split complete:")
    print(f"  - Train: {len(train_ids)} athletes")
    print(f"  - Val: {len(val_ids)} athletes")
    print(f"  - Test: {len(test_ids)} athletes")
    
    return train_df, val_df, test_df, split_manifest


# ============================================================================
# PAIRWISE DATASET GENERATION
# ============================================================================

def generate_pairwise_dataset(
    df: pd.DataFrame,
    n_pairs_per_athlete: int = 50,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Generate pairwise compatibility dataset for training.
    
    For each athlete, sample N other athletes and compute pairwise features.
    
    Args:
        df: Unified athlete dataframe
        n_pairs_per_athlete: Number of pairs to generate per athlete
        random_state: Random seed
        
    Returns:
        X_pairs: Pairwise feature matrix
        y_compatibility: Compatibility scores (0-1)
        pair_ids: List of (athlete_a_id, athlete_b_id) tuples
    """
    print(f"\nGenerating pairwise dataset ({n_pairs_per_athlete} pairs per athlete)...")
    
    np.random.seed(random_state)
    
    pairs = []
    pair_ids = []
    
    df_reset = df.reset_index(drop=True)
    
    for idx in range(len(df_reset)):
        athlete_a = df_reset.iloc[idx]
        
        # Sample random other athletes
        other_indices = [i for i in range(len(df_reset)) if i != idx]
        sampled_indices = np.random.choice(
            other_indices,
            size=min(n_pairs_per_athlete, len(other_indices)),
            replace=False
        )
        
        for other_idx in sampled_indices:
            athlete_b = df_reset.iloc[other_idx]
            
            # Generate pairwise features
            pair_features = generate_pair_features(athlete_a, athlete_b)
            
            # Calculate ground truth compatibility score (0-1)
            # Based on multiple factors with weights
            compatibility = (
                pair_features['age_similarity'] * 0.15 +
                pair_features['sport_match'] * 0.20 +
                pair_features['training_philosophy_match'] * 0.15 +
                pair_features.get('recovery_similarity', 0.5) * 0.10 +
                pair_features.get('performance_similarity', 0.5) * 0.15 +
                pair_features.get('social_sentiment_similarity', 0.5) * 0.10 +
                pair_features.get('mood_similarity', 0.5) * 0.10 +
                pair_features['location_match'] * 0.05
            )
            
            # Add small random noise to avoid perfect predictions
            compatibility = np.clip(compatibility + np.random.normal(0, 0.05), 0, 1)
            
            pairs.append(list(pair_features.values()))
            pair_ids.append((athlete_a['athlete_id'], athlete_b['athlete_id']))
    
    X_pairs = np.array(pairs)
    y_compatibility = np.array([
        (pair[list(generate_pair_features(
            df[df['athlete_id'] == pid[0]].iloc[0],
            df[df['athlete_id'] == pid[1]].iloc[0]
        ).keys()).index('age_similarity')] * 0.15 +
         pair[list(generate_pair_features(
            df[df['athlete_id'] == pid[0]].iloc[0],
            df[df['athlete_id'] == pid[1]].iloc[0]
        ).keys()).index('sport_match')] * 0.20)
        for pair, pid in zip(pairs, pair_ids)
    ])
    
    # Simpler approach - recalculate compatibility
    y_compatibility = []
    for i, (aid_a, aid_b) in enumerate(pair_ids):
        athlete_a = df_reset[df_reset['athlete_id'] == aid_a].iloc[0]
        athlete_b = df_reset[df_reset['athlete_id'] == aid_b].iloc[0]
        pair_features = generate_pair_features(athlete_a, athlete_b)
        
        compatibility = (
            pair_features['age_similarity'] * 0.15 +
            pair_features['sport_match'] * 0.20 +
            pair_features['training_philosophy_match'] * 0.15 +
            pair_features.get('recovery_similarity', 0.5) * 0.10 +
            pair_features.get('performance_similarity', 0.5) * 0.15 +
            pair_features.get('social_sentiment_similarity', 0.5) * 0.10 +
            pair_features.get('mood_similarity', 0.5) * 0.10 +
            pair_features['location_match'] * 0.05
        )
        
        # Add noise
        compatibility = np.clip(compatibility + np.random.normal(0, 0.05), 0, 1)
        y_compatibility.append(compatibility)
    
    y_compatibility = np.array(y_compatibility)
    
    print(f"✓ Generated {len(pairs)} pairs")
    print(f"  - Feature dimensions: {X_pairs.shape}")
    print(f"  - Compatibility range: [{y_compatibility.min():.3f}, {y_compatibility.max():.3f}]")
    
    return X_pairs, y_compatibility, pair_ids


# ============================================================================
# MODEL 1: COMPATIBILITY MODEL (XGBoost Regressor)
# ============================================================================

class CompatibilityModel:
    """
    Gradient Boosting model for predicting pairwise compatibility scores.
    """
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train compatibility model."""
        print("\n=== Training Compatibility Model (XGBoost) ===")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print(f"Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
        
        return {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_rmse': val_rmse,
            'val_mae': val_mae
        }
    
    def predict(self, X):
        """Predict compatibility scores."""
        return self.model.predict(X)


# ============================================================================
# MODEL 2: LONGEVITY MODEL (Classification)
# ============================================================================

class LongevityModel:
    """
    Probabilistic classifier for relationship longevity prediction.
    
    Predicts likelihood of 3-month, 6-month, and 12-month relationship success.
    """
    
    def __init__(self):
        self.models = {
            '3m': xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1),
            '6m': xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1),
            '12m': xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1),
        }
        
    def train(self, X_train, y_compatibility_train, X_val, y_compatibility_val):
        """Train longevity classifiers."""
        print("\n=== Training Longevity Models (Classification) ===")
        
        # Generate labels based on compatibility thresholds
        # 3m: >0.4, 6m: >0.5, 12m: >0.6
        y_train_3m = (y_compatibility_train > 0.4).astype(int)
        y_train_6m = (y_compatibility_train > 0.5).astype(int)
        y_train_12m = (y_compatibility_train > 0.6).astype(int)
        
        y_val_3m = (y_compatibility_val > 0.4).astype(int)
        y_val_6m = (y_compatibility_val > 0.5).astype(int)
        y_val_12m = (y_compatibility_val > 0.6).astype(int)
        
        results = {}
        
        for duration, y_train, y_val in [
            ('3m', y_train_3m, y_val_3m),
            ('6m', y_train_6m, y_val_6m),
            ('12m', y_train_12m, y_val_12m)
        ]:
            print(f"\nTraining {duration} model...")
            self.models[duration].fit(X_train, y_train)
            
            # Evaluate
            train_pred = self.models[duration].predict(X_train)
            val_pred = self.models[duration].predict(X_val)
            
            train_f1 = f1_score(y_train, train_pred)
            train_acc = accuracy_score(y_train, train_pred)
            val_f1 = f1_score(y_val, val_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            # ROC-AUC
            train_proba = self.models[duration].predict_proba(X_train)[:, 1]
            val_proba = self.models[duration].predict_proba(X_val)[:, 1]
            train_auc = roc_auc_score(y_train, train_proba)
            val_auc = roc_auc_score(y_val, val_proba)
            
            print(f"  Train F1: {train_f1:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
            print(f"  Val F1: {val_f1:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
            
            results[duration] = {
                'train_f1': train_f1,
                'train_acc': train_acc,
                'train_auc': train_auc,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_auc': val_auc
            }
        
        return results
    
    def predict(self, X):
        """Predict longevity probabilities."""
        return {
            duration: model.predict_proba(X)[:, 1]
            for duration, model in self.models.items()
        }


# ============================================================================
# MODEL 3: EXPERIENCE TIER CLASSIFIER
# ============================================================================

class ExperienceTierModel:
    """
    Multi-class classifier for experience tier prediction.
    
    Tiers: Beginner (0-2 years), Intermediate (3-5 years), 
           Advanced (6-10 years), Elite (11+ years)
    """
    
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
        
    def _create_tier_labels(self, df):
        """Create experience tier labels from years_training."""
        tiers = []
        years_col = 'years_training' if 'years_training' in df.columns else 'years_active'
        for years in df[years_col]:
            if years <= 2:
                tiers.append('Beginner')
            elif years <= 5:
                tiers.append('Intermediate')
            elif years <= 10:
                tiers.append('Advanced')
            else:
                tiers.append('Elite')
        return tiers
    
    def train(self, X_train, df_train, X_val, df_val):
        """Train experience tier classifier."""
        print("\n=== Training Experience Tier Classifier ===")
        
        # Create tier labels
        y_train_tiers = self._create_tier_labels(df_train)
        y_val_tiers = self._create_tier_labels(df_val)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train_tiers)
        y_val_encoded = self.label_encoder.transform(y_val_tiers)
        
        # Train
        self.model.fit(X_train, y_train_encoded)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = accuracy_score(y_train_encoded, train_pred)
        train_f1 = f1_score(y_train_encoded, train_pred, average='weighted')
        val_acc = accuracy_score(y_val_encoded, val_pred)
        val_f1 = f1_score(y_val_encoded, val_pred, average='weighted')
        
        print(f"Train Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        return {
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'classes': self.label_encoder.classes_.tolist()
        }
    
    def predict(self, X):
        """Predict experience tiers."""
        encoded_preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(encoded_preds)


# ============================================================================
# MODEL 4: TRAINING-STYLE EMBEDDING MODEL (Siamese Network)
# ============================================================================

class SiamesePairDataset(Dataset):
    """Dataset for Siamese network training."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SiameseNetwork(nn.Module):
    """Siamese Neural Network for learning athlete embeddings."""
    
    def __init__(self, input_dim, embedding_dim=64):
        super(SiameseNetwork, self).__init__()
        
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim),
        )
        
    def forward_one(self, x):
        """Embed a single athlete profile."""
        return self.embedding_net(x)
    
    def forward(self, x):
        """Forward pass."""
        return self.forward_one(x)


class TrainingStyleEmbeddingModel:
    """
    Training-style embedding model using Siamese networks.
    
    Learns compact representations where similar athletes are close in embedding space.
    """
    
    def __init__(self, input_dim, embedding_dim=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseNetwork(input_dim, embedding_dim).to(self.device)
        self.embedding_dim = embedding_dim
        
    def train(self, X_train, X_val, epochs=50, batch_size=128, lr=0.001):
        """Train embedding model with contrastive loss."""
        print("\n=== Training Training-Style Embedding Model (Siamese Network) ===")
        print(f"Device: {self.device}")
        
        # Create dummy labels for training (we'll use cosine similarity)
        y_train = np.ones(len(X_train))
        y_val = np.ones(len(X_val))
        
        train_dataset = SiamesePairDataset(X_train, y_train)
        val_dataset = SiamesePairDataset(X_val, y_val)
        
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CosineEmbeddingLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Generate two views of each sample (with noise for augmentation)
                noise1 = torch.randn_like(batch_X) * 0.1
                noise2 = torch.randn_like(batch_X) * 0.1
                
                embed1 = self.model(batch_X + noise1)
                embed2 = self.model(batch_X + noise2)
                
                # Positive pairs should be similar
                loss = criterion(embed1, embed2, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    noise1 = torch.randn_like(batch_X) * 0.1
                    noise2 = torch.randn_like(batch_X) * 0.1
                    
                    embed1 = self.model(batch_X + noise1)
                    embed2 = self.model(batch_X + noise2)
                    
                    loss = criterion(embed1, embed2, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print(f"✓ Training complete. Best val loss: {best_val_loss:.4f}")
        
        return {'best_val_loss': best_val_loss}
    
    def embed(self, X):
        """Generate embeddings for athletes."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            embeddings = self.model(X_tensor)
            return embeddings.cpu().numpy()


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    
    print("=" * 80)
    print("ATHLETE MATCHMAKING ML TRAINING PIPELINE")
    print("=" * 80)
    
    # Create output directories
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load and merge all datasets
    # ========================================================================
    
    loader = UnifiedDataLoader()
    datasets = loader.load_all_datasets()
    unified_df = loader.merge_datasets(datasets)
    
    print(f"\n{'='*80}")
    print(f"Unified dataset shape: {unified_df.shape}")
    print(f"Features: {unified_df.columns.tolist()[:10]}... ({len(unified_df.columns)} total)")
    
    # ========================================================================
    # STEP 2: Create stratified split
    # ========================================================================
    
    train_df, val_df, test_df, split_manifest = create_stratified_split(
        unified_df,
        train_size=0.65,
        val_size=0.20,
        test_size=0.15
    )
    
    # Save split manifest
    with open(models_dir / 'split_manifest.json', 'w') as f:
        json.dump(split_manifest, f, indent=2)
    print("✓ Saved split_manifest.json")
    
    # ========================================================================
    # STEP 3: Feature engineering
    # ========================================================================
    
    feature_engineer = FeatureEngineer()
    X_train, feature_names = feature_engineer.fit_transform(train_df)
    X_val = feature_engineer.transform(val_df)
    X_test = feature_engineer.transform(test_df)
    
    # Save feature pipeline
    joblib.dump(feature_engineer, models_dir / 'feature_pipeline.pkl')
    print("✓ Saved feature_pipeline.pkl")
    
    # ========================================================================
    # STEP 4: Generate pairwise datasets
    # ========================================================================
    
    X_pairs_train, y_compat_train, pairs_train = generate_pairwise_dataset(
        train_df, n_pairs_per_athlete=50
    )
    X_pairs_val, y_compat_val, pairs_val = generate_pairwise_dataset(
        val_df, n_pairs_per_athlete=50
    )
    X_pairs_test, y_compat_test, pairs_test = generate_pairwise_dataset(
        test_df, n_pairs_per_athlete=50
    )
    
    # ========================================================================
    # STEP 5: Train all models
    # ========================================================================
    
    all_results = {}
    
    # Model 1: Compatibility
    compat_model = CompatibilityModel()
    compat_results = compat_model.train(X_pairs_train, y_compat_train, X_pairs_val, y_compat_val)
    all_results['compatibility'] = compat_results
    joblib.dump(compat_model, models_dir / 'compatibility_model.pkl')
    print("✓ Saved compatibility_model.pkl")
    
    # Model 2: Longevity
    longevity_model = LongevityModel()
    longevity_results = longevity_model.train(X_pairs_train, y_compat_train, X_pairs_val, y_compat_val)
    all_results['longevity'] = longevity_results
    joblib.dump(longevity_model, models_dir / 'longevity_model.pkl')
    print("✓ Saved longevity_model.pkl")
    
    # Model 3: Experience Tier
    exp_tier_model = ExperienceTierModel()
    exp_results = exp_tier_model.train(X_train, train_df, X_val, val_df)
    all_results['experience_tier'] = exp_results
    joblib.dump(exp_tier_model, models_dir / 'experience_classifier.pkl')
    print("✓ Saved experience_classifier.pkl")
    
    # Model 4: Training-Style Embeddings
    embedding_model = TrainingStyleEmbeddingModel(input_dim=X_train.shape[1])
    embed_results = embedding_model.train(X_train, X_val, epochs=50)
    all_results['embeddings'] = embed_results
    
    # Save embedding model
    torch.save(embedding_model.model.state_dict(), models_dir / 'training_embeddings_model.pth')
    print("✓ Saved training_embeddings_model.pth")
    
    # ========================================================================
    # STEP 6: Final evaluation on test set
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION")
    print("=" * 80)
    
    # Compatibility test
    test_pred_compat = compat_model.predict(X_pairs_test)
    test_rmse = np.sqrt(mean_squared_error(y_compat_test, test_pred_compat))
    test_mae = mean_absolute_error(y_compat_test, test_pred_compat)
    print(f"\nCompatibility Model:")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    all_results['compatibility']['test_rmse'] = test_rmse
    all_results['compatibility']['test_mae'] = test_mae
    
    # Longevity test
    y_test_3m = (y_compat_test > 0.4).astype(int)
    y_test_6m = (y_compat_test > 0.5).astype(int)
    y_test_12m = (y_compat_test > 0.6).astype(int)
    
    print(f"\nLongevity Models:")
    for duration, y_test in [('3m', y_test_3m), ('6m', y_test_6m), ('12m', y_test_12m)]:
        test_pred = longevity_model.models[duration].predict(X_pairs_test)
        test_f1 = f1_score(y_test, test_pred)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"  {duration}: Test F1: {test_f1:.4f}, Acc: {test_acc:.4f}")
        all_results['longevity'][duration]['test_f1'] = test_f1
        all_results['longevity'][duration]['test_acc'] = test_acc
    
    # Experience tier test
    test_pred_tier = exp_tier_model.predict(X_test)
    y_test_tiers = exp_tier_model._create_tier_labels(test_df)
    y_test_encoded = exp_tier_model.label_encoder.transform(y_test_tiers)
    test_pred_encoded = exp_tier_model.label_encoder.transform(test_pred_tier)
    test_acc = accuracy_score(y_test_encoded, test_pred_encoded)
    test_f1 = f1_score(y_test_encoded, test_pred_encoded, average='weighted')
    print(f"\nExperience Tier Classifier:")
    print(f"  Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    all_results['experience_tier']['test_acc'] = test_acc
    all_results['experience_tier']['test_f1'] = test_f1
    
    # Save all results
    with open(models_dir / 'training_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n✓ Saved training_results.json")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nTrained models saved to: {models_dir}/")
    print("  - compatibility_model.pkl")
    print("  - longevity_model.pkl")
    print("  - experience_classifier.pkl")
    print("  - training_embeddings_model.pth")
    print("  - feature_pipeline.pkl")
    print("  - split_manifest.json")
    print("  - training_results.json")


if __name__ == '__main__':
    main()
