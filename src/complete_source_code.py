"""
Complete Source Code Package for Athlete Peer Matching
========================================================
This file contains all the core Python modules needed for the project.
Split this into separate files for production use.

Author: Your Name
Date: November 12, 2025
"""

# ============================================================================
# FILE: src/data_loader.py
# ============================================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataLoader:
    """
    Loads all athlete datasets from CSV files
    
    Attributes:
        data_dir (str): Directory containing raw data files
    """
    
    def __init__(self, data_dir='data/raw/'):
        """Initialize data loader with data directory"""
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_all_datasets(self):
        """
        Load all athlete datasets
        
        Returns:
            dict: Dictionary of dataframes with keys:
                  'master', 'activities', 'daily', 'injuries', 
                  'mood', 'life', 'social'
        """
        datasets = {}
        
        # Load master athlete profiles
        datasets['master'] = pd.read_csv(
            self.data_dir / 'detailed_athlete_master_profiles.csv'
        )
        
        # Load activity data
        datasets['activities'] = pd.read_csv(
            self.data_dir / 'detailed_strava_activities.csv'
        )
        
        # Load daily health metrics
        datasets['daily'] = pd.read_csv(
            self.data_dir / 'detailed_daily_health_metrics.csv'
        )
        
        # Load injury history
        datasets['injuries'] = pd.read_csv(
            self.data_dir / 'detailed_injury_medical_history.csv'
        )
        
        # Load mood tracking
        datasets['mood'] = pd.read_csv(
            self.data_dir / 'detailed_mood_tracking.csv'
        )
        
        # Load life timeline
        datasets['life'] = pd.read_csv(
            self.data_dir / 'detailed_life_timeline_career.csv'
        )
        
        # Load social profiles
        datasets['social'] = pd.read_csv(
            self.data_dir / 'detailed_social_communication.csv'
        )
        
        return datasets


# ============================================================================
# FILE: src/feature_engineering.py
# ============================================================================

from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    """
    Creates pairwise features for athlete matching
    
    Attributes:
        scaler (StandardScaler): Scikit-learn scaler for normalization
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.scaler = StandardScaler()
    
    def create_features(self, df_athletes, n_pairs_per_athlete=8):
        """
        Create pairwise features from athlete profiles
        
        Args:
            df_athletes (pd.DataFrame): Athlete master profiles
            n_pairs_per_athlete (int): Number of comparison pairs per athlete
            
        Returns:
            tuple: (X, y, feature_names, pair_ids)
                X: Feature matrix (numpy array)
                y: Labels (numpy array)
                feature_names: List of feature names
                pair_ids: List of (athlete1_id, athlete2_id) tuples
        """
        pairs = []
        
        for idx, athlete1 in df_athletes.iterrows():
            # Sample random athletes to compare
            comparison_indices = np.random.choice(
                len(df_athletes), 
                size=n_pairs_per_athlete, 
                replace=False
            )
            
            for comp_idx in comparison_indices:
                athlete2 = df_athletes.iloc[comp_idx]
                
                # Calculate features
                features = self._compute_pairwise_features(athlete1, athlete2)
                
                # Calculate label (match or not)
                match_score = self._calculate_match_score(athlete1, athlete2)
                label = 1 if match_score > 0.65 else 0
                
                pairs.append({
                    'athlete1_id': athlete1['athlete_id'],
                    'athlete2_id': athlete2['athlete_id'],
                    **features,
                    'label': label
                })
        
        df_pairs = pd.DataFrame(pairs)
        
        # Feature names
        feature_names = [
            'age_diff', 'sport_match', 'level_match', 'training_vol_diff',
            'recovery_diff', 'mental_mood_diff', 'engagement_diff',
            'injury_history_diff', 'family_status_match', 
            'communication_style_match', 'social_support_diff',
            'life_balance_diff', 'match_probability_raw'
        ]
        
        X = df_pairs[feature_names].values
        y = df_pairs['label'].values
        pair_ids = list(zip(df_pairs['athlete1_id'], df_pairs['athlete2_id']))
        
        return X, y, feature_names, pair_ids
    
    def _compute_pairwise_features(self, athlete1, athlete2):
        """Compute features between two athletes"""
        features = {}
        
        # Age difference
        features['age_diff'] = abs(athlete1['age'] - athlete2['age'])
        
        # Sport match (if both are triathletes/cyclists)
        features['sport_match'] = 1.0  # Simplified
        
        # Level match
        features['level_match'] = 0.0  # Simplified
        
        # Training volume difference
        features['training_vol_diff'] = abs(
            athlete1['total_distance_km'] - athlete2['total_distance_km']
        )
        
        # Recovery score difference
        features['recovery_diff'] = abs(
            athlete1['avg_daily_recovery_score'] - athlete2['avg_daily_recovery_score']
        )
        
        # Mental mood difference (simplified)
        features['mental_mood_diff'] = 0.5
        
        # Engagement difference
        features['engagement_diff'] = abs(
            athlete1['social_engagement_score'] - athlete2['social_engagement_score']
        )
        
        # Injury history difference
        features['injury_history_diff'] = abs(
            athlete1['injury_count'] - athlete2['injury_count']
        )
        
        # Family status match (simplified)
        features['family_status_match'] = 1.0
        
        # Communication style match (simplified)
        features['communication_style_match'] = 0.5
        
        # Social support difference
        features['social_support_diff'] = abs(
            athlete1['social_network_size'] - athlete2['social_network_size']
        ) / 100.0
        
        # Life balance difference (simplified)
        features['life_balance_diff'] = 1.0
        
        # Raw match probability
        features['match_probability_raw'] = self._calculate_match_score(
            athlete1, athlete2
        )
        
        return features
    
    def _calculate_match_score(self, athlete1, athlete2):
        """Calculate compatibility score (0-1)"""
        # Age similarity
        age_diff = abs(athlete1['age'] - athlete2['age'])
        age_score = max(0, 1 - (age_diff / 15))
        
        # Training volume similarity
        vol_diff = abs(athlete1['total_distance_km'] - athlete2['total_distance_km'])
        vol_score = max(0, 1 - (vol_diff / 2000))
        
        # Recovery similarity
        recovery_diff = abs(
            athlete1['avg_daily_recovery_score'] - athlete2['avg_daily_recovery_score']
        )
        recovery_score = max(0, 1 - (recovery_diff / 50))
        
        # Weighted average
        match_score = (
            0.30 * age_score +
            0.25 * vol_score +
            0.25 * recovery_score +
            0.20 * 0.8  # Other factors
        )
        
        return match_score


# ============================================================================
# FILE: src/model.py
# ============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

class LogisticMatchingModel:
    """
    Logistic Regression model for athlete peer matching
    
    Attributes:
        model (LogisticRegression): Scikit-learn logistic regression
        scaler (StandardScaler): Feature scaler
        X_train, X_test, y_train, y_test: Train/test splits
    """
    
    def __init__(self, random_state=42):
        """Initialize model"""
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs'
        )
        self.scaler = StandardScaler()
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, X, y, test_size=0.3, random_state=None):
        """Split and scale data"""
        if random_state is None:
            random_state = self.random_state
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def train(self):
        """Train the model"""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, X):
        """Predict match labels"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict match probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, filepath):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        return instance
    
    def find_top_matches(self, target_athlete, all_athletes, feature_engineer, top_k=5):
        """
        Find top K matches for a target athlete
        
        Args:
            target_athlete: Athlete profile (Series or dict)
            all_athletes: DataFrame of all athletes
            feature_engineer: FeatureEngineer instance
            top_k: Number of top matches to return
            
        Returns:
            list: Top K matches with (athlete_id, score, profile) tuples
        """
        matches = []
        
        for idx, candidate in all_athletes.iterrows():
            if candidate['athlete_id'] == target_athlete['athlete_id']:
                continue
            
            # Compute features
            features = feature_engineer._compute_pairwise_features(
                target_athlete, candidate
            )
            
            # Predict match probability
            X = np.array([list(features.values())])
            match_prob = self.predict_proba(X)[0]
            
            matches.append((
                candidate['athlete_id'],
                match_prob,
                candidate
            ))
        
        # Sort by score and return top K
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]


# ============================================================================
# FILE: src/evaluation.py
# ============================================================================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

class ModelEvaluator:
    """
    Evaluates model performance
    
    Attributes:
        model (LogisticMatchingModel): Trained model
    """
    
    def __init__(self, model):
        """Initialize evaluator"""
        self.model = model
        self.metrics = {}
    
    def evaluate(self):
        """
        Evaluate model on test set
        
        Returns:
            dict: Performance metrics
        """
        # Predictions
        y_pred = self.model.predict(self.model.X_test)
        y_pred_proba = self.model.predict_proba(self.model.X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(self.model.y_test, y_pred),
            'precision': precision_score(self.model.y_test, y_pred),
            'recall': recall_score(self.model.y_test, y_pred),
            'f1_score': f1_score(self.model.y_test, y_pred),
            'roc_auc': roc_auc_score(self.model.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.model.y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return self.metrics
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from coefficients"""
        coefficients = self.model.model.coef_[0]
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance
    
    def save_results(self, output_dir):
        """Save evaluation results to CSV files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        pd.DataFrame([self.metrics]).to_csv(
            output_dir / 'performance_metrics.csv', index=False
        )
        
        # Save confusion matrix
        pd.DataFrame(
            self.metrics['confusion_matrix'],
            columns=['Predicted_0', 'Predicted_1'],
            index=['Actual_0', 'Actual_1']
        ).to_csv(output_dir / 'confusion_matrix.csv')


# ============================================================================
# FILE: src/visualization.py
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationGenerator:
    """
    Generates visualizations for model analysis
    
    Attributes:
        model (LogisticMatchingModel): Trained model
        evaluator (ModelEvaluator): Model evaluator
    """
    
    def __init__(self, model, evaluator):
        """Initialize visualization generator"""
        self.model = model
        self.evaluator = evaluator
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(
            self.model.y_test, 
            self.evaluator.metrics['y_pred_proba']
        )
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Athlete Peer Matching')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_names, save_path=None):
        """Plot feature importance"""
        importance = self.evaluator.get_feature_importance(feature_names)
        
        plt.figure(figsize=(10, 8))
        colors = ['green' if c > 0 else 'red' for c in importance['coefficient']]
        plt.barh(range(len(importance)), importance['coefficient'], color=colors)
        plt.yticks(range(len(importance)), importance['feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance - Logistic Regression Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix heatmap"""
        cm = self.evaluator.metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Match', 'Match'],
                   yticklabels=['No Match', 'Match'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix - Test Set')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_match_distribution(self, save_path=None):
        """Plot match probability distribution"""
        y_pred_proba = self.evaluator.metrics['y_pred_proba']
        y_test = self.model.y_test
        
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, 
                label='Matches', color='blue')
        plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6,
                label='Non-Matches', color='red')
        plt.xlabel('Predicted Match Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Match Probabilities')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


print("All source code modules defined")
print("\nTo use in production:")
print("1. Split this file into separate files in src/ directory")
print("2. Each class goes in its own file (data_loader.py, model.py, etc.)")
print("3. Add __init__.py to src/ directory")
print("4. Import modules as: from src.model import LogisticMatchingModel")