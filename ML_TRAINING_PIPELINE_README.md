# ML Training Pipeline - Complete Documentation

## Overview

This document describes the complete ML training pipeline for the Athlete Matchmaking System using the 1,000-athlete dataset (548,523 total records across 7 CSV files).

## Architecture

### Data Pipeline
```
7 CSV Files → Unified Loader → Feature Engineering → Stratified Split → Model Training
```

### Models Trained
1. **Compatibility Model** (XGBoost Regressor) - Predicts pairwise compatibility scores (0-1)
2. **Longevity Model** (XGBoost Classifier) - Predicts 3m/6m/12m relationship success
3. **Experience Tier Classifier** (XGBoost Multi-class) - Classifies athletes into experience tiers
4. **Training-Style Embedding Model** (Siamese Neural Network) - Learns athlete embeddings
5. **Existing Models** (Logistic, RandomForest, NN, Cosine, KNN) - Retrained on new dataset

---

## Data Sources

### Input Files (data/raw/)
| File | Records | Description |
|------|---------|-------------|
| `athlete_master_profiles.csv` | 1,000 | Demographics, sports, experience |
| `daily_health_metrics.csv` | 120,018 | Resting HR, HRV, recovery, VO2max |
| `strava_activities.csv` | 246,052 | Distance, pace, heart rate, duration |
| `injury_medical_history.csv` | 4,899 | Injury types, severity, recovery days |
| `mood_tracking.csv` | 90,348 | Mood scores, stress, motivation, energy |
| `social_communication.csv` | 72,060 | Interactions, duration, sentiment |
| `life_timeline_career.csv` | 14,146 | Career milestones, major events |

**Total Dataset Size:** 548,523 records, 29.96 MB

---

## Feature Engineering

### Unified Table Features

#### 1. Health Aggregations
- Mean, std, min, max for: resting HR, HRV, sleep, recovery, fatigue, VO2max, weight
- Rolling windows: 7-day and 30-day averages for recovery and HRV
- **Total:** ~24 features

#### 2. Activity Aggregations
- Distance: sum, mean, std, max
- Duration: sum, mean, std
- Speed & HR: mean, std
- Elevation & calories: sum, mean
- Activity count, performance scores
- Consistency index: std/mean ratio
- **Total:** ~18 features

#### 3. Injury Features
- Injury count, severity (mean, total, max)
- Recovery days (mean, total, max)
- **Injury risk score:** count × severity × (recovery_days / 10)
- **Total:** ~8 features

#### 4. Mood Features
- Mood score, stress, motivation, energy: mean, std, min, max
- **Emotional stability:** 1 / (mood_std + 0.1)
- **Total:** ~13 features

#### 5. Social Features
- Interaction count, duration (total, mean)
- Sentiment: mean, std
- **Communication frequency:** interactions per day
- **Total:** ~6 features

#### 6. Career Features
- Life event count
- Major milestone count (competitions, PRs, qualifications)
- **Total:** ~2 features

#### 7. Demographics (One-Hot Encoded)
- Gender, primary sport, training philosophy
- Location (80 cities), coaching status, partner preference
- **Total:** ~100+ categorical features

**Total Engineered Features:** ~170+ features per athlete

---

## Pairwise Feature Generation

The `generate_pair_features(athlete_a, athlete_b)` function creates compatibility features:

### Pairwise Features
1. **Age similarity:** 1 - (|age_diff| / 50)
2. **Gender match:** 1.0 if same, else 0.0
3. **Sport match:** 1.0 if same, else 0.0
4. **Experience similarity:** 1 - (|years_diff| / 20)
5. **Training philosophy match:** 1.0 if same, else 0.0
6. **Location match:** 1.0 if same city, else 0.0
7. **Recovery similarity:** 1 - (|recovery_diff| / 100)
8. **HR similarity:** 1 - (|hr_diff| / 100)
9. **Activity level similarity:** 1 - (|distance_diff| / 50)
10. **Performance similarity:** 1 - (|perf_diff| / 20)
11. **Social sentiment similarity:** 1 - (|sentiment_diff| / 10)
12. **Mood similarity:** 1 - (|mood_diff| / 10)
13. **Network size similarity:** 1 - (|network_diff| / 200)

All features are clipped to [0, 1] range.

### Ground Truth Compatibility Score
```python
compatibility = (
    age_similarity * 0.15 +
    sport_match * 0.20 +
    training_philosophy_match * 0.15 +
    recovery_similarity * 0.10 +
    performance_similarity * 0.15 +
    social_sentiment_similarity * 0.10 +
    mood_similarity * 0.10 +
    location_match * 0.05
) + random_noise(0, 0.05)
```

---

## Dataset Splitting Strategy

### Athlete-Stratified Split (65/20/15)

```python
Total Athletes: 1,000
├── Train: 650 athletes (65%)
├── Validation: 200 athletes (20%)
└── Test: 150 athletes (15%)
```

**Key Properties:**
- ✅ **No data leakage:** Each athlete appears in only ONE split
- ✅ **No mixing:** Health/activity timelines never cross splits
- ✅ **Reproducible:** Random seed = 42
- ✅ **Manifest saved:** `split_manifest.json` contains all athlete IDs per split

### Pairwise Dataset Generation
- **50 pairs per athlete** (train/val/test)
- Train: ~32,500 pairs
- Val: ~10,000 pairs
- Test: ~7,500 pairs

---

## Model Details

### 1. Compatibility Model (XGBoost Regressor)

**Purpose:** Predict pairwise compatibility scores (0-1)

**Architecture:**
```python
XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Inputs:** Pairwise features (13 dimensions)
**Output:** Compatibility score (continuous, 0-1)

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

**Saved As:** `models/compatibility_model.pkl`

---

### 2. Longevity Model (XGBoost Classifier)

**Purpose:** Predict relationship longevity at 3 time horizons

**Architecture:** 3 separate classifiers
- **3-month model:** Threshold = 0.4 compatibility
- **6-month model:** Threshold = 0.5 compatibility
- **12-month model:** Threshold = 0.6 compatibility

```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)
```

**Inputs:** Pairwise features (13 dimensions)
**Outputs:** Probability of success at each time horizon

**Evaluation Metrics:**
- F1 Score
- Accuracy
- ROC-AUC

**Saved As:** `models/longevity_model.pkl`

---

### 3. Experience Tier Classifier

**Purpose:** Classify athletes into experience tiers

**Tiers:**
- **Beginner:** 0-2 years active
- **Intermediate:** 3-5 years active
- **Advanced:** 6-10 years active
- **Elite:** 11+ years active

**Architecture:**
```python
XGBClassifier(
    n_estimators=150,
    max_depth=5,
    random_state=42
)
```

**Inputs:** Athlete features (~170 dimensions)
**Output:** Experience tier (4 classes)

**Evaluation Metrics:**
- Accuracy
- Weighted F1 Score

**Saved As:** `models/experience_classifier.pkl`

---

### 4. Training-Style Embedding Model (Siamese Network)

**Purpose:** Learn compact athlete representations for similarity matching

**Architecture:**
```python
SiameseNetwork(
    embedding_dim=64,
    layers=[
        Linear(input_dim, 256) + ReLU + Dropout(0.3),
        Linear(256, 128) + ReLU + Dropout(0.3),
        Linear(128, 64)
    ]
)
```

**Loss Function:** Cosine Embedding Loss
- Positive pairs (same athlete + noise) should be similar
- Training uses data augmentation with Gaussian noise (σ=0.1)

**Training:**
- Epochs: 50
- Batch size: 128
- Optimizer: Adam (lr=0.001)
- Device: GPU if available, else CPU

**Inputs:** Athlete features (~170 dimensions)
**Output:** 64-dimensional embedding vector

**Evaluation:** Cosine similarity between embeddings

**Saved As:** `models/training_embeddings_model.pth` (PyTorch state dict)

---

### 5. Existing Models (Retrained)

#### Logistic Regression
- Binary classifier for match/no-match
- L2 regularization
- Saved as: `models/logistic_model.pkl`

#### Random Forest
- Ensemble of decision trees
- 100 estimators, max_depth=10
- Saved as: `models/random_forest_model.pkl`

#### Neural Network
- Multi-layer perceptron
- Hidden layers: [128, 64, 32]
- Dropout: 0.3
- Saved as: `models/neural_network_model.pkl`

#### Cosine Similarity
- Non-parametric similarity model
- Measures cosine distance between feature vectors
- Saved as: `models/cosine_similarity_model.pkl`

#### Weighted KNN
- K=15 nearest neighbors
- Distance-weighted voting
- Saved as: `models/weighted_knn_model.pkl`

---

## Training Scripts

### train_models.py

**Main training pipeline for all new models.**

**Usage:**
```bash
python train_models.py
```

**What it does:**
1. Loads all 7 CSV files
2. Merges into unified table with aggregations
3. Engineers 170+ features
4. Creates 65/20/15 stratified split
5. Generates pairwise datasets
6. Trains 4 core models
7. Evaluates on test set
8. Saves all artifacts

**Runtime:** ~10-15 minutes (CPU), ~5-8 minutes (GPU for embedding model)

**Outputs:**
- `models/compatibility_model.pkl`
- `models/longevity_model.pkl`
- `models/experience_classifier.pkl`
- `models/training_embeddings_model.pth`
- `models/feature_pipeline.pkl`
- `models/split_manifest.json`
- `models/training_results.json`

---

### retrain_existing_models.py

**Retrains all 5 existing models with new dataset.**

**Usage:**
```bash
python retrain_existing_models.py
```

**What it does:**
1. Loads split manifest and unified data
2. Uses same train/val/test split
3. Trains: Logistic, RandomForest, NN, Cosine, KNN
4. Evaluates all models
5. Creates comparison table

**Runtime:** ~5-8 minutes

**Outputs:**
- `models/logistic_model.pkl`
- `models/random_forest_model.pkl`
- `models/neural_network_model.pkl`
- `models/cosine_similarity_model.pkl`
- `models/weighted_knn_model.pkl`
- `models/existing_models_results.json`
- `models/existing_models_comparison.csv`

---

## Inference

### predict.py

**Production inference script with multiple modes.**

#### Mode 1: Single Pair Prediction

```bash
python predict.py --athlete_a A001 --athlete_b A002
```

**Output:**
```
📊 Compatibility Score: 0.782
📅 Longevity Probabilities:
  - 3 months: 0.891
  - 6 months: 0.743
  - 12 months: 0.612
🎯 Experience Tiers:
  - Athlete A: Advanced
  - Athlete B: Advanced
  - Tier Match: Yes
🔗 Embedding Similarity: 0.854
💡 Recommendations:
  ✅ High compatibility - Excellent match potential
  📅 Strong long-term potential (12+ months)
  🎯 Experience tier match: Both Advanced
```

Saves detailed JSON to `prediction_detail.json`

#### Mode 2: Batch Prediction

```bash
python predict.py --batch_predict pairs.csv --output results.csv
```

**Input CSV format:**
```csv
athlete_a_id,athlete_b_id
A001,A002
A001,A003
A002,A005
```

**Output:** CSV with all predictions

#### Mode 3: Find Top Matches

```bash
python predict.py --find_matches A001 --top_k 10
```

Finds top 10 most compatible matches for athlete A001.

**Output:** Sorted CSV with compatibility scores

---

## Model Artifacts

### Directory Structure
```
models/
├── compatibility_model.pkl          # XGBoost compatibility regressor
├── longevity_model.pkl               # 3 XGBoost classifiers (3m/6m/12m)
├── experience_classifier.pkl         # Multi-class XGBoost
├── training_embeddings_model.pth     # PyTorch Siamese network
├── feature_pipeline.pkl              # Scaler + encoders
├── split_manifest.json               # Train/val/test athlete IDs
├── training_results.json             # All training metrics
├── logistic_model.pkl                # Existing model (retrained)
├── random_forest_model.pkl           # Existing model (retrained)
├── neural_network_model.pkl          # Existing model (retrained)
├── cosine_similarity_model.pkl       # Existing model (retrained)
├── weighted_knn_model.pkl            # Existing model (retrained)
├── existing_models_results.json      # Existing model metrics
└── existing_models_comparison.csv    # Comparison table
```

---

## Evaluation Metrics

### Compatibility Model (Regression)
- **RMSE:** Root Mean Squared Error (lower is better)
- **MAE:** Mean Absolute Error (lower is better)
- **Target:** < 0.10 RMSE, < 0.08 MAE

### Longevity Models (Classification)
- **F1 Score:** Harmonic mean of precision/recall (0-1, higher is better)
- **Accuracy:** Correct predictions / total predictions
- **ROC-AUC:** Area under ROC curve (0.5-1.0, higher is better)
- **Target:** > 0.75 F1, > 0.80 AUC

### Experience Tier Classifier
- **Accuracy:** Multi-class accuracy
- **Weighted F1:** Accounts for class imbalance
- **Target:** > 0.85 accuracy

### Embedding Model
- **Cosine Similarity Loss:** Measures embedding quality (lower is better)
- **Target:** < 0.15 validation loss

### Existing Models (Binary Classification)
- **F1 Score:** Primary metric
- **Precision:** True positives / predicted positives
- **Recall:** True positives / actual positives
- **ROC-AUC:** Overall discrimination ability
- **Target:** > 0.70 F1

---

## Dependencies

### Required Packages
```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch scipy joblib matplotlib seaborn
```

### Versions Tested
- Python: 3.11+
- pandas: 2.0+
- numpy: 1.24+
- scikit-learn: 1.3+
- xgboost: 2.0+
- torch: 2.0+

---

## Performance Benchmarks

### Training Time (1,000 athletes, ~50,000 pairs)
| Model | Training Time | Prediction Time (1000 pairs) |
|-------|---------------|------------------------------|
| Compatibility (XGBoost) | ~2 min | ~0.5 sec |
| Longevity (3 models) | ~3 min | ~1 sec |
| Experience Tier | ~1 min | ~0.2 sec |
| Embeddings (Siamese) | ~5 min (GPU) / ~15 min (CPU) | ~1 sec |
| Logistic Regression | ~30 sec | ~0.3 sec |
| Random Forest | ~2 min | ~0.5 sec |
| Neural Network | ~3 min | ~0.4 sec |
| Cosine Similarity | instant | ~0.2 sec |
| Weighted KNN | instant | ~2 sec |

**Total Training Time:** ~20 minutes (CPU) / ~12 minutes (GPU)

---

## Usage Examples

### Example 1: Train All Models
```bash
# Train new models
python train_models.py

# Retrain existing models
python retrain_existing_models.py
```

### Example 2: Make Predictions
```python
from predict import MatchmakingPredictor

predictor = MatchmakingPredictor()

# Single pair
result = predictor.predict_pair('A001', 'A002')
print(f"Compatibility: {result['compatibility_score']:.3f}")

# Find top matches
top_matches = predictor.find_top_matches('A001', top_k=5)
print(top_matches)
```

### Example 3: Use generate_pair_features
```python
import pandas as pd
from train_models import UnifiedDataLoader, generate_pair_features

# Load data
loader = UnifiedDataLoader()
datasets = loader.load_all_datasets()
unified_df = loader.merge_datasets(datasets)

# Get two athletes
athlete_a = unified_df[unified_df['athlete_id'] == 'A001'].iloc[0]
athlete_b = unified_df[unified_df['athlete_id'] == 'A002'].iloc[0]

# Generate pairwise features
features = generate_pair_features(athlete_a, athlete_b)
print(features)
# Output: {'age_similarity': 0.92, 'sport_match': 1.0, ...}
```

---

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce `n_pairs_per_athlete` in `generate_pairwise_dataset()`

### Issue: Slow Training
**Solution:** 
- Use GPU for embedding model: `torch.cuda.is_available()`
- Reduce `epochs` for Siamese network
- Reduce `n_estimators` for XGBoost models

### Issue: Import Errors
**Solution:** Ensure all dependencies installed: `pip install -r requirements.txt`

### Issue: FileNotFoundError
**Solution:** Run `train_models.py` first to generate all artifacts

---

## Best Practices

1. **Always use the same random seed (42)** for reproducibility
2. **Never modify split_manifest.json** - ensures consistent evaluation
3. **Check for data leakage** - athlete IDs should never overlap between splits
4. **Monitor validation metrics** - stop training if overfitting
5. **Use batch prediction** for large-scale inference (more efficient)
6. **Save predictions to CSV** for downstream analysis

---

## Future Improvements

1. **Hyperparameter tuning** - Grid search for optimal parameters
2. **Feature selection** - Remove redundant/low-importance features
3. **Ensemble models** - Combine multiple models for better predictions
4. **Online learning** - Update models with new data incrementally
5. **A/B testing** - Compare model versions in production
6. **Explainability** - SHAP values for feature importance

---

## Contact & Support

For questions or issues, please refer to:
- Main README: `README.md`
- API Documentation: `README_API.md`
- Consistency Improvements: `CONSISTENCY_IMPROVEMENTS.md`
- Data Generation: `DATA_GENERATION_SUMMARY.md`

---

**Last Updated:** November 25, 2025
**Version:** 1.0
**Dataset:** 1,000 athletes, 548,523 records
