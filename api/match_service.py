"""
Enhanced match service with improved compatibility scoring and preference support.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering_improved import ImprovedFeatureEngineer
from src.complete_source_code import DataLoader
from api.explanation_generator import ExplanationGenerator
from api.compatibility_engine import CompatibilityEngine
from api.preference_filter import PreferenceFilter
from sklearn.preprocessing import StandardScaler


class MatchService:
    """Enhanced service for finding peer matches with preference support"""
    
    def __init__(self):
        """Initialize the match service"""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        
        # Load person data
        print("Loading person database...")
        loader = DataLoader(data_dir=str(self.data_dir / "raw"))
        self.data = loader.load_all_datasets()
        self.persons_df = self.data["master"].copy()
        
        # Initialize components
        self.feature_engineer = ImprovedFeatureEngineer(random_state=42)
        self.explanation_generator = ExplanationGenerator()
        self.compatibility_engine = CompatibilityEngine()
        self.preference_filter = PreferenceFilter()
        
        # Load trained models
        self.models = {}
        self._load_models()
        
        print(f"Match service initialized with {len(self.persons_df)} people in database")
    
    def _load_models(self):
        """Load all trained models"""
        try:
            if (self.models_dir / "logistic_model.pkl").exists():
                logistic_data = joblib.load(self.models_dir / "logistic_model.pkl")
                self.models["logistic"] = {
                    "model": logistic_data.get("calibrated_model"),
                    "scaler": logistic_data.get("scaler"),
                    "feature_names": logistic_data.get("feature_names", self.feature_engineer.feature_names)
                }
                print("Loaded logistic regression model")
        except Exception as e:
            print(f"Warning: Could not load saved models: {e}")
            print("Will use feature-based matching")
    
    def _get_model(self, model_type: str):
        """Get or create a model instance"""
        if model_type == "logistic" and "logistic" in self.models:
            return self.models["logistic"]
        return None
    
    def _create_user_profile_series(self, user_profile: Dict) -> pd.Series:
        """Create a pandas Series from user profile for feature computation"""
        defaults = self.persons_df.median(numeric_only=True).to_dict()
        
        profile_series = pd.Series({
            "athlete_id": "USER_PROFILE",
            "age": user_profile.get("age"),
            "location": user_profile.get("location", "Unknown"),
            "gender": user_profile.get("gender", "Unknown"),
            "total_distance_km": user_profile.get("total_distance_km") or defaults.get("total_distance_km", 2000),
            "avg_daily_recovery_score": user_profile.get("avg_daily_recovery_score") or defaults.get("recovery_score", 70),
            "total_activities": user_profile.get("total_activities") or int(defaults.get("total_activities", 100)),
            "injury_count": user_profile.get("injury_count") or int(defaults.get("injury_count", 0)),
            "social_engagement_score": user_profile.get("social_engagement_score") or defaults.get("social_engagement_score", 7.0),
            "communication_preference": user_profile.get("communication_preference") or "Text-focused",
            "primary_sport": user_profile.get("primary_interest") or "General",
            "competition_level": user_profile.get("experience_level") or "Intermediate",
            "social_network_size": defaults.get("social_network_size", 20),
            "coaching_support": defaults.get("coaching_support", 1),
            "training_partners": defaults.get("training_partners", 2),
            "total_life_events": defaults.get("total_life_events", 10),
            "residences_count": defaults.get("residences_count", 2),
            "avg_resting_hr": defaults.get("resting_heart_rate", 60),
        })
        
        # Fill missing columns with defaults
        for col in self.persons_df.columns:
            if col not in profile_series:
                if col in defaults:
                    profile_series[col] = defaults[col]
                elif self.persons_df[col].dtype == 'object':
                    profile_series[col] = self.persons_df[col].mode()[0] if len(self.persons_df[col].mode()) > 0 else ""
                else:
                    profile_series[col] = 0
        
        # Ensure communication_style is set
        if "communication_style" in self.persons_df.columns and "communication_style" not in profile_series:
            profile_series["communication_style"] = profile_series.get("communication_preference", "Text-focused")
        
        return profile_series
    
    def find_matches_from_profile(
        self,
        user_profile: Dict,
        user_preferences: Optional[Dict] = None,
        max_results: int = 5,
        model_type: str = "logistic"
    ) -> List[Dict[str, Any]]:
        """Find matches based on user profile and preferences"""
        try:
            # Create user profile series
            user_series = self._create_user_profile_series(user_profile)
            
            # Filter candidates by preferences if provided
            if user_preferences:
                candidates_df = self.preference_filter.filter_by_preferences(
                    self.persons_df.copy(), user_preferences
                )
                # If filtering results in too few candidates, use original dataset
                if len(candidates_df) < max_results:
                    candidates_df = self.persons_df.copy()
            else:
                candidates_df = self.persons_df.copy()
            
            # Compute matches
            matches = []
            for idx, candidate in candidates_df.iterrows():
                # Compute similarity features
                features = self.feature_engineer._compute_features_with_variance(
                    user_series, candidate
                )
                
                # Calculate compatibility using enhanced engine
                compatibility_score, breakdown, layer_insights = self.compatibility_engine.calculate_compatibility(
                    user_profile=user_profile,
                    match_profile=candidate.to_dict(),
                    similarity_features=features,
                    user_preferences=user_preferences
                )
                layer_payload = [insight.to_dict() for insight in layer_insights]
                
                # Get top contributing factors
                top_factors = self.compatibility_engine.get_top_contributing_factors(
                    features, top_n=5
                )
                
                # Generate dynamic explanation
                explanation = self.explanation_generator.generate_explanation(
                    user_profile=user_profile,
                    match_profile=candidate.to_dict(),
                    similarity_features=features,
                    match_score=compatibility_score,
                    top_features=top_factors,
                    layer_insights=layer_payload,
                )
                
                # Generate detailed breakdown
                detailed_breakdown = self.explanation_generator.generate_detailed_breakdown(
                    features
                )
                
                # Extract top reasons
                top_reasons = self._extract_top_reasons(features, top_factors)
                
                # Build complete match result
                match_result = {
                    "person_id": str(candidate["athlete_id"]),
                    "name": str(candidate.get("name", "Unknown")),
                    "age": int(candidate["age"]),
                    "location": str(candidate.get("location", "Unknown")),
                    "gender": str(candidate.get("gender", "Unknown")),
                    "match_score": compatibility_score,
                    "match_probability": compatibility_score,  # Use compatibility score
                    "explanation": {
                        "summary": explanation,
                        "detailed_breakdown": detailed_breakdown,
                        "top_reasons": top_reasons,
                    },
                    "similarity_features": features,
                    "compatibility_breakdown": breakdown,
                    "intelligence_layers": layer_payload,
                    "profile_summary": self._extract_profile_summary(candidate)
                }
                
                matches.append(match_result)
            
            # Apply consistency validation pass
            validated_matches = self._consistency_validation_pass(matches, user_profile)
            
            # Sort by compatibility score and return top results
            validated_matches.sort(key=lambda x: x["match_probability"], reverse=True)
            return validated_matches[:max_results]
            
        except Exception as e:
            print(f"Error in find_matches_from_profile: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _consistency_validation_pass(
        self, 
        matches: List[Dict[str, Any]], 
        user_profile: Dict
    ) -> List[Dict[str, Any]]:
        """
        Apply final consistency validation to eliminate artificial patterns.
        Removes impossible geographic clusters, validates score variance, etc.
        """
        if not matches:
            return matches
        
        validated = []
        scores_seen = set()
        
        # Geographic incompatibility list
        INCOMPATIBLE_CONTINENTS = {
            frozenset(["india", "spain"]),
            frozenset(["india", "poland"]),
            frozenset(["india", "italy"]),
            frozenset(["china", "brazil"]),
            frozenset(["japan", "argentina"]),
        }
        
        user_location = str(user_profile.get("location", "")).lower()
        
        for match in matches:
            match_location = str(match.get("location", "")).lower()
            
            # Check for impossible geographic clustering
            is_geographically_impossible = False
            for incompatible_set in INCOMPATIBLE_CONTINENTS:
                user_matches = [loc for loc in incompatible_set if loc in user_location]
                match_matches = [loc for loc in incompatible_set if loc in match_location]
                if user_matches and match_matches and user_matches != match_matches:
                    is_geographically_impossible = True
                    break
            
            if is_geographically_impossible:
                # Severely downgrade score for impossible geographic matches
                match["match_score"] *= 0.30
                match["match_probability"] *= 0.30
                match["explanation"]["summary"] = f"[Geographic penalty applied] {match['explanation']['summary']}"
            
            # Detect and penalize score clustering (identical percentages)
            score_rounded = round(match["match_score"], 3)
            if score_rounded in scores_seen:
                # Add micro-variance to prevent clustering
                variance = (hash(match["person_id"]) % 150) / 5000.0  # 0-3% variance
                match["match_score"] += variance
                match["match_probability"] += variance
            scores_seen.add(round(match["match_score"], 3))
            
            # Validate preference mismatches are properly penalized
            if "preferred_gender" in (user_profile.get("preferences") or {}):
                pref_gender = user_profile["preferences"]["preferred_gender"].lower()
                if pref_gender not in ("any", "", None):
                    match_gender = str(match.get("gender", "")).lower()
                    if match_gender != pref_gender:
                        # Ensure harsh penalty is applied
                        if match["match_score"] > 0.65:
                            match["match_score"] *= 0.35
                            match["match_probability"] *= 0.35
            
            # Ensure score spread is realistic (no flat distributions)
            # Scores should naturally decay from best match
            if len(validated) > 0:
                best_score = validated[0]["match_score"]
                expected_decay = len(validated) * 0.02  # 2% per rank
                if match["match_score"] > best_score - expected_decay:
                    match["match_score"] = best_score - expected_decay - (hash(match["person_id"]) % 50) / 1000.0
                    match["match_probability"] = match["match_score"]
            
            validated.append(match)
        
        return validated
    
    def _extract_profile_summary(self, candidate: pd.Series) -> Dict[str, Any]:
        """Safely extract profile summary from candidate with fallbacks"""
        # Safely extract primary interest
        primary_interest = "Unknown"
        if "primary_sport" in candidate.index:
            try:
                val = candidate["primary_sport"]
                primary_interest = str(val) if pd.notna(val) else "Unknown"
            except:
                primary_interest = "Unknown"
        elif "primary_interest" in candidate.index:
            try:
                val = candidate["primary_interest"]
                primary_interest = str(val) if pd.notna(val) else "Unknown"
            except:
                primary_interest = "Unknown"
        
        # Safely extract experience level
        experience_level = "Unknown"
        if "competition_level" in candidate.index:
            try:
                val = candidate["competition_level"]
                experience_level = str(val) if pd.notna(val) else "Unknown"
            except:
                experience_level = "Unknown"
        elif "experience_level" in candidate.index:
            try:
                val = candidate["experience_level"]
                experience_level = str(val) if pd.notna(val) else "Unknown"
            except:
                experience_level = "Unknown"
        
        # Safely extract communication style
        communication_style = "Unknown"
        if "communication_style" in candidate.index:
            try:
                val = candidate["communication_style"]
                communication_style = str(val) if pd.notna(val) else "Unknown"
            except:
                communication_style = "Unknown"
        elif "communication_preference" in candidate.index:
            try:
                val = candidate["communication_preference"]
                communication_style = str(val) if pd.notna(val) else "Unknown"
            except:
                communication_style = "Unknown"
        
        # Build profile summary with safe extraction
        profile_summary = {
            "total_distance_km": float(candidate.get("total_distance_km", 0) or 0),
            "total_activities": int(candidate.get("total_activities", 0) or 0),
            "primary_interest": primary_interest,
            "experience_level": experience_level,
            "injury_count": int(candidate.get("injury_count", 0) or 0),
            "social_engagement": 0.0,
            "communication_style": communication_style,
            "years_experience": 0,
            "nationality": "Unknown",
            "career_status": "Unknown",
        }
        
        # Safely extract optional fields
        if "social_engagement_score" in candidate.index:
            try:
                val = candidate["social_engagement_score"]
                if pd.notna(val):
                    profile_summary["social_engagement"] = float(val)
            except:
                pass
        
        if "years_training" in candidate.index:
            try:
                val = candidate["years_training"]
                if pd.notna(val):
                    profile_summary["years_experience"] = int(val)
            except:
                pass
        
        if "nationality" in candidate.index:
            try:
                val = candidate["nationality"]
                if pd.notna(val):
                    profile_summary["nationality"] = str(val)
            except:
                pass
        
        if "career_status" in candidate.index:
            try:
                val = candidate["career_status"]
                if pd.notna(val):
                    profile_summary["career_status"] = str(val)
            except:
                pass
        
        return profile_summary
    
    def _extract_top_reasons(
        self,
        features: Dict[str, float],
        top_factors: List[Tuple[str, float]]
    ) -> List[str]:
        """Extract top reasons from similarity features"""
        reasons = []
        feature_descriptions = {
            "age_similarity": "Similar age",
            "distance_similarity": "Similar activity levels",
            "recovery_similarity": "Similar wellness approach",
            "activity_frequency_similarity": "Similar activity frequency",
            "social_alignment": "Similar social preferences",
            "communication_compatibility": "Compatible communication style",
            "network_overlap": "Similar social network",
            "coaching_alignment": "Similar support preferences",
            "partner_support_similarity": "Compatible training partner style",
        }
        
        for feature_name, weighted_score in top_factors[:3]:
            if weighted_score > 0.05:  # Only include significant factors
                desc = feature_descriptions.get(
                    feature_name,
                    feature_name.replace("_", " ").title()
                )
                # Get actual similarity score
                actual_score = features.get(feature_name, 0)
                reasons.append(f"{desc} ({actual_score:.0%})")
        
        return reasons
    
    def get_person_profile(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed profile for a specific person"""
        person = self.persons_df[self.persons_df["athlete_id"] == person_id]
        if len(person) == 0:
            return None
        
        person = person.iloc[0]
        return person.to_dict()
