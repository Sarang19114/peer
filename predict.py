"""
Inference Script for Athlete Matchmaking Models
================================================

Loads trained models and generates predictions for new athlete pairs.

Usage:
    python predict.py --athlete_a A001 --athlete_b A002
    python predict.py --batch_predict pairs.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from train_models import (
    UnifiedDataLoader,
    FeatureEngineer,
    generate_pair_features,
    SiameseNetwork
)


class MatchmakingPredictor:
    """
    Unified predictor for all matchmaking models.
    """
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        
        print("Loading trained models...")
        
        # Load models
        self.compatibility_model = joblib.load(self.models_dir / 'compatibility_model.pkl')
        self.longevity_model = joblib.load(self.models_dir / 'longevity_model.pkl')
        self.experience_classifier = joblib.load(self.models_dir / 'experience_classifier.pkl')
        self.feature_pipeline = joblib.load(self.models_dir / 'feature_pipeline.pkl')
        
        # Load embedding model
        with open(self.models_dir / 'training_results.json', 'r') as f:
            results = json.load(f)
        
        # Load PyTorch embedding model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # We need to know input dimension - load from a sample
        sample_data = pd.read_csv('data/raw/athlete_master_profiles.csv').head(1)
        loader = UnifiedDataLoader()
        datasets = loader.load_all_datasets()
        unified = loader.merge_datasets(datasets)
        sample_features, _ = self.feature_pipeline.fit_transform(unified.head(1))
        input_dim = sample_features.shape[1]
        
        self.embedding_model = SiameseNetwork(input_dim, embedding_dim=64).to(self.device)
        self.embedding_model.load_state_dict(
            torch.load(self.models_dir / 'training_embeddings_model.pth', map_location=self.device)
        )
        self.embedding_model.eval()
        
        # Load split manifest
        with open(self.models_dir / 'split_manifest.json', 'r') as f:
            self.split_manifest = json.load(f)
        
        print("✓ All models loaded successfully")
        
    def load_athlete_data(self):
        """Load and merge all athlete data."""
        loader = UnifiedDataLoader()
        datasets = loader.load_all_datasets()
        unified_df = loader.merge_datasets(datasets)
        return unified_df
    
    def predict_pair(self, athlete_a_id: str, athlete_b_id: str) -> Dict[str, any]:
        """
        Predict all metrics for a single athlete pair.
        
        Args:
            athlete_a_id: ID of first athlete
            athlete_b_id: ID of second athlete
            
        Returns:
            Dictionary containing all predictions
        """
        # Load athlete data
        df = self.load_athlete_data()
        
        athlete_a = df[df['athlete_id'] == athlete_a_id]
        athlete_b = df[df['athlete_id'] == athlete_b_id]
        
        if len(athlete_a) == 0:
            raise ValueError(f"Athlete {athlete_a_id} not found")
        if len(athlete_b) == 0:
            raise ValueError(f"Athlete {athlete_b_id} not found")
        
        athlete_a = athlete_a.iloc[0]
        athlete_b = athlete_b.iloc[0]
        
        # Generate pairwise features
        pair_features = generate_pair_features(athlete_a, athlete_b)
        X_pair = np.array([list(pair_features.values())])
        
        # Predict compatibility
        compatibility_score = self.compatibility_model.predict(X_pair)[0]
        
        # Predict longevity
        longevity_probs = {
            duration: model.predict_proba(X_pair)[0, 1]
            for duration, model in self.longevity_model.models.items()
        }
        
        # Generate embeddings
        X_a = self.feature_pipeline.transform(athlete_a.to_frame().T)
        X_b = self.feature_pipeline.transform(athlete_b.to_frame().T)
        
        with torch.no_grad():
            embed_a = self.embedding_model(torch.FloatTensor(X_a).to(self.device))
            embed_b = self.embedding_model(torch.FloatTensor(X_b).to(self.device))
            
            # Cosine similarity
            cosine_sim = torch.nn.functional.cosine_similarity(embed_a, embed_b).item()
        
        # Predict experience tiers
        tier_a = self.experience_classifier.predict(X_a)[0]
        tier_b = self.experience_classifier.predict(X_b)[0]
        
        return {
            'athlete_a_id': athlete_a_id,
            'athlete_b_id': athlete_b_id,
            'compatibility_score': float(compatibility_score),
            'longevity_probabilities': {
                '3_month': float(longevity_probs['3m']),
                '6_month': float(longevity_probs['6m']),
                '12_month': float(longevity_probs['12m']),
            },
            'embedding_similarity': float(cosine_sim),
            'experience_tiers': {
                'athlete_a': tier_a,
                'athlete_b': tier_b,
                'tier_match': tier_a == tier_b
            },
            'pairwise_features': pair_features,
            'recommendations': self._generate_recommendations(
                compatibility_score, longevity_probs, tier_a, tier_b
            )
        }
    
    def _generate_recommendations(
        self, 
        compat_score: float,
        longevity_probs: Dict[str, float],
        tier_a: str,
        tier_b: str
    ) -> List[str]:
        """Generate human-readable recommendations."""
        recommendations = []
        
        if compat_score > 0.7:
            recommendations.append("✅ High compatibility - Excellent match potential")
        elif compat_score > 0.5:
            recommendations.append("⚠️ Moderate compatibility - Could work with effort")
        else:
            recommendations.append("❌ Low compatibility - May face challenges")
        
        if longevity_probs['12m'] > 0.7:
            recommendations.append("📅 Strong long-term potential (12+ months)")
        elif longevity_probs['6m'] > 0.6:
            recommendations.append("📅 Good medium-term potential (6+ months)")
        elif longevity_probs['3m'] > 0.5:
            recommendations.append("📅 Suitable for short-term partnership (3+ months)")
        else:
            recommendations.append("📅 Limited longevity prediction")
        
        if tier_a == tier_b:
            recommendations.append(f"🎯 Experience tier match: Both {tier_a}")
        else:
            recommendations.append(f"⚖️ Experience gap: {tier_a} ↔ {tier_b}")
        
        return recommendations
    
    def batch_predict(self, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Predict for multiple athlete pairs.
        
        Args:
            pairs: List of (athlete_a_id, athlete_b_id) tuples
            
        Returns:
            DataFrame with predictions for all pairs
        """
        results = []
        
        print(f"Predicting for {len(pairs)} pairs...")
        
        for i, (a_id, b_id) in enumerate(pairs):
            try:
                prediction = self.predict_pair(a_id, b_id)
                results.append({
                    'athlete_a_id': a_id,
                    'athlete_b_id': b_id,
                    'compatibility_score': prediction['compatibility_score'],
                    'longevity_3m': prediction['longevity_probabilities']['3_month'],
                    'longevity_6m': prediction['longevity_probabilities']['6_month'],
                    'longevity_12m': prediction['longevity_probabilities']['12_month'],
                    'embedding_similarity': prediction['embedding_similarity'],
                    'tier_a': prediction['experience_tiers']['athlete_a'],
                    'tier_b': prediction['experience_tiers']['athlete_b'],
                    'tier_match': prediction['experience_tiers']['tier_match'],
                })
            except Exception as e:
                print(f"Error predicting pair ({a_id}, {b_id}): {e}")
                continue
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(pairs)} pairs...")
        
        return pd.DataFrame(results)
    
    def find_top_matches(
        self, 
        athlete_id: str, 
        top_k: int = 10,
        min_compatibility: float = 0.5
    ) -> pd.DataFrame:
        """
        Find top K matches for a given athlete.
        
        Args:
            athlete_id: ID of athlete to find matches for
            top_k: Number of top matches to return
            min_compatibility: Minimum compatibility threshold
            
        Returns:
            DataFrame with top matches sorted by compatibility
        """
        df = self.load_athlete_data()
        
        # Get all other athletes
        other_athletes = df[df['athlete_id'] != athlete_id]['athlete_id'].tolist()
        
        # Generate pairs
        pairs = [(athlete_id, other_id) for other_id in other_athletes]
        
        # Batch predict
        results = self.batch_predict(pairs)
        
        # Filter and sort
        results = results[results['compatibility_score'] >= min_compatibility]
        results = results.sort_values('compatibility_score', ascending=False)
        
        return results.head(top_k)


def main():
    """CLI interface for inference."""
    parser = argparse.ArgumentParser(description='Athlete Matchmaking Inference')
    parser.add_argument('--athlete_a', type=str, help='First athlete ID')
    parser.add_argument('--athlete_b', type=str, help='Second athlete ID')
    parser.add_argument('--batch_predict', type=str, help='Path to CSV with athlete pairs')
    parser.add_argument('--find_matches', type=str, help='Find top matches for athlete ID')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top matches to return')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file for batch predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MatchmakingPredictor()
    
    if args.athlete_a and args.athlete_b:
        # Single pair prediction
        print(f"\nPredicting match between {args.athlete_a} and {args.athlete_b}...")
        result = predictor.predict_pair(args.athlete_a, args.athlete_b)
        
        print("\n" + "=" * 80)
        print("MATCHMAKING PREDICTION")
        print("=" * 80)
        print(f"\nAthletes: {result['athlete_a_id']} ↔ {result['athlete_b_id']}")
        print(f"\n📊 Compatibility Score: {result['compatibility_score']:.3f}")
        print(f"\n📅 Longevity Probabilities:")
        print(f"  - 3 months: {result['longevity_probabilities']['3_month']:.3f}")
        print(f"  - 6 months: {result['longevity_probabilities']['6_month']:.3f}")
        print(f"  - 12 months: {result['longevity_probabilities']['12_month']:.3f}")
        print(f"\n🎯 Experience Tiers:")
        print(f"  - Athlete A: {result['experience_tiers']['athlete_a']}")
        print(f"  - Athlete B: {result['experience_tiers']['athlete_b']}")
        print(f"  - Tier Match: {'Yes' if result['experience_tiers']['tier_match'] else 'No'}")
        print(f"\n🔗 Embedding Similarity: {result['embedding_similarity']:.3f}")
        print(f"\n💡 Recommendations:")
        for rec in result['recommendations']:
            print(f"  {rec}")
        
        # Save detailed results
        with open('prediction_detail.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Detailed results saved to prediction_detail.json")
    
    elif args.batch_predict:
        # Batch prediction
        print(f"\nLoading pairs from {args.batch_predict}...")
        pairs_df = pd.read_csv(args.batch_predict)
        pairs = list(zip(pairs_df['athlete_a_id'], pairs_df['athlete_b_id']))
        
        results = predictor.batch_predict(pairs)
        results.to_csv(args.output, index=False)
        
        print(f"\n✓ Predictions saved to {args.output}")
        print(f"\nSummary:")
        print(f"  - Total pairs: {len(results)}")
        print(f"  - Avg compatibility: {results['compatibility_score'].mean():.3f}")
        print(f"  - High compat (>0.7): {(results['compatibility_score'] > 0.7).sum()}")
        print(f"  - Medium compat (0.5-0.7): {((results['compatibility_score'] >= 0.5) & (results['compatibility_score'] <= 0.7)).sum()}")
        print(f"  - Low compat (<0.5): {(results['compatibility_score'] < 0.5).sum()}")
    
    elif args.find_matches:
        # Find top matches
        print(f"\nFinding top {args.top_k} matches for athlete {args.find_matches}...")
        matches = predictor.find_top_matches(args.find_matches, top_k=args.top_k)
        
        print("\n" + "=" * 80)
        print(f"TOP {args.top_k} MATCHES FOR ATHLETE {args.find_matches}")
        print("=" * 80)
        print(matches.to_string(index=False))
        
        matches.to_csv(f'top_matches_{args.find_matches}.csv', index=False)
        print(f"\n✓ Saved to top_matches_{args.find_matches}.csv")
    
    else:
        print("Error: Please specify one of --athlete_a/--athlete_b, --batch_predict, or --find_matches")
        parser.print_help()


if __name__ == '__main__':
    main()
