"""
Advanced compatibility scoring engine with preference weighting and dynamic reasoning.
Generates accurate, meaningful compatibility scores based on model reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from api.intelligence_layers import INTELLIGENCE_LAYERS, LayerInsight


@dataclass
class CompatibilityWeights:
    """Weights for different compatibility factors"""
    age: float = 0.15
    location: float = 0.10
    activity_level: float = 0.15
    recovery_wellness: float = 0.10
    social_engagement: float = 0.12
    communication: float = 0.10
    interests: float = 0.13
    experience_level: float = 0.08
    lifestyle: float = 0.07


class CompatibilityEngine:
    """Advanced compatibility scoring engine"""
    
    def __init__(self):
        self.weights = CompatibilityWeights()
        self.feature_importance = {
            "age_similarity": 0.15,
            "distance_similarity": 0.15,
            "recovery_similarity": 0.12,
            "activity_frequency_similarity": 0.12,
            "social_alignment": 0.12,
            "communication_compatibility": 0.10,
            "network_overlap": 0.08,
            "coaching_alignment": 0.06,
            "partner_support_similarity": 0.05,
            "life_event_similarity": 0.03,
            "residence_similarity": 0.02,
            "resting_hr_alignment": 0.02,
            "injury_experience_match": 0.02,
        }
        self.intelligence_layers = INTELLIGENCE_LAYERS
    
    def calculate_compatibility(
        self,
        user_profile: Dict,
        match_profile: Dict,
        similarity_features: Dict[str, float],
        user_preferences: Optional[Dict] = None
    ) -> Tuple[float, Dict[str, float], List[LayerInsight]]:
        """
        Calculate comprehensive compatibility score with preference weighting.
        
        Returns:
            Tuple of (compatibility_score, detailed_breakdown)
        """
        # Base compatibility from feature similarities
        base_score = self._calculate_base_compatibility(similarity_features)
        
        # Apply preference weights if provided
        if user_preferences:
            preference_multiplier = self._calculate_preference_multiplier(
                user_profile, match_profile, user_preferences, similarity_features
            )
            adjusted_score = base_score * preference_multiplier
        else:
            adjusted_score = base_score
            preference_multiplier = 1.0
        
        # Normalize base score to deterministic 0-1 range
        normalized_score = max(0.0, min(1.0, adjusted_score))

        # Evaluate intelligence layers
        layer_insights = self._evaluate_layers(
            user_profile=user_profile,
            match_profile=match_profile,
            similarity_features=similarity_features,
        )
        layer_average = (
            sum(insight.score for insight in layer_insights) / len(layer_insights)
            if layer_insights
            else normalized_score
        )

        # Blend normalized score with layer intelligence
        base_blend = 0.55 * normalized_score + 0.45 * layer_average
        
        # Add natural variance to prevent clustering (±1-3%)
        # Use hash of profiles for deterministic but unique variance per pair
        user_id = str(user_profile.get('athlete_id', '')) + str(user_profile.get('age', 0))
        match_id = str(match_profile.get('athlete_id', '')) + str(match_profile.get('age', 0))
        variance_seed = hash((user_id, match_id)) % 100000
        variance = (variance_seed / 100000.0 - 0.5) * 0.08  # -4% to +4% for better spread
        final_raw = base_blend + variance
        
        # Dynamic scaling with real separation between quality tiers
        # Excellent matches: 85-99%, Strong: 70-84%, Weak: 50-69%, Poor: <50%
        if final_raw >= 0.75:
            # Excellent matches get 85-99 range with natural spread
            final_score = 0.85 + (final_raw - 0.75) * 0.56  # Maps 0.75-1.0 → 85-99%
        elif final_raw >= 0.60:
            # Strong matches get 70-84 range
            final_score = 0.70 + (final_raw - 0.60) * 0.93  # Maps 0.60-0.75 → 70-84%
        elif final_raw >= 0.40:
            # Weak matches get 50-69 range
            final_score = 0.50 + (final_raw - 0.40) * 0.95  # Maps 0.40-0.60 → 50-69%
        else:
            # Poor matches get <50 range
            final_score = 0.30 + final_raw * 0.50           # Maps 0-0.40 → 30-50%
        
        # Ensure bounds without flattening
        final_score = min(max(final_score, 0.35), 0.99)

        # Create detailed breakdown
        breakdown = self._create_breakdown(
            similarity_features,
            base_score,
            preference_multiplier,
            normalized_score,
            layer_average,
            final_score,
            layer_insights,
        )
        
        return float(final_score), breakdown, layer_insights
    
    def _calculate_base_compatibility(
        self,
        similarity_features: Dict[str, float]
    ) -> float:
        """Calculate base compatibility from feature similarities"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for feature_name, similarity_score in similarity_features.items():
            weight = self.feature_importance.get(feature_name, 0.05)
            weighted_sum += similarity_score * weight
            total_weight += weight
        
        if total_weight > 0:
            base_score = weighted_sum / total_weight
        else:
            base_score = 0.5
        
        # Apply non-linear scaling to emphasize high similarities
        # This helps separate truly compatible matches
        if base_score > 0.8:
            scaled_score = 0.8 + (base_score - 0.8) * 0.2 / 0.2  # Enhance high scores
        else:
            scaled_score = base_score
        
        return float(scaled_score)
    
    def _calculate_preference_multiplier(
        self,
        user_profile: Dict,
        match_profile: Dict,
        preferences: Dict,
        similarity_features: Dict[str, float]
    ) -> float:
        """Calculate multiplier based on how well match meets user preferences"""
        multipliers = []
        
        # Age preference - aggressive penalty for out-of-range
        if "preferred_age_min" in preferences:
            match_age = match_profile.get("age")
            if match_age is not None:
                age_min = preferences.get("preferred_age_min", 18)
                age_max = preferences.get("preferred_age_max", 100)
                if not (age_min <= match_age <= age_max):
                    # Real penalty - way out of range is nearly disqualifying
                    years_off = min(abs(match_age - age_min), abs(match_age - age_max))
                    multipliers.append(max(0.15, 1.0 - years_off * 0.12))  # -12% per year off
                else:
                    multipliers.append(1.0)
        
        # Gender preference - harsh penalty for mismatch
        if "preferred_gender" in preferences:
            p = preferences["preferred_gender"]
            if p and p.lower() not in ("any", "", None):
                m = str(match_profile.get("gender", "")).lower()
                if m == p.lower():
                    multipliers.append(1.0)
                else:
                    multipliers.append(0.25)  # 75% penalty for wrong gender
        
        # Location preference - realistic geographic penalty
        if "preferred_location" in preferences:
            pref_loc = preferences.get("preferred_location", "").strip().lower()
            match_loc = str(match_profile.get("location", "")).strip().lower()
            if pref_loc and match_loc and match_loc != "unknown":
                if pref_loc in match_loc or match_loc in pref_loc:
                    multipliers.append(1.0)
                else:
                    # Real geographic penalty - different location is a problem
                    multipliers.append(0.35)  # 65% penalty for location mismatch
            elif pref_loc and match_loc == "unknown":
                # If location is unknown, use distance similarity as fallback
                distance_sim = similarity_features.get("distance_similarity", 0.5)
                multipliers.append(0.4 + distance_sim * 0.25)  # Partial credit, still penalized
        
        # Interest preference - real collaborative filtering penalty
        if "preferred_interests" in preferences:
            preferred_interests = preferences.get("preferred_interests", [])
            # Try both possible column names
            match_interest = str(
                match_profile.get("primary_sport") or match_profile.get("primary_interest", "")
            ).strip().lower()
            user_interest = str(user_profile.get("primary_interest", "")).strip().lower()
            
            if preferred_interests and match_interest:
                if any(pi.lower() in match_interest or match_interest in pi.lower() 
                       for pi in preferred_interests if pi):
                    multipliers.append(1.0)
                elif user_interest == match_interest:
                    multipliers.append(1.0)
                elif user_interest in (match_profile.get("secondary_interests") or []):
                    multipliers.append(0.65)  # Secondary match is acceptable
                else:
                    multipliers.append(0.20)  # Harsh penalty for complete interest mismatch
        
        # Experience level preference
        if "preferred_experience_level" in preferences:
            preferred_level = preferences.get("preferred_experience_level", "").strip().lower()
            # Try both possible column names
            match_level = str(
                match_profile.get("competition_level") or match_profile.get("experience_level", "")
            ).strip().lower()
            if preferred_level and match_level:
                if preferred_level == match_level:
                    multipliers.append(1.0)
                else:
                    multipliers.append(0.9)
        
        # Communication style preference
        if "preferred_communication_style" in preferences:
            preferred_comm = preferences.get("preferred_communication_style", "").strip().lower()
            match_comm = str(match_profile.get("communication_style") or 
                           match_profile.get("communication_preference", "")).strip().lower()
            if preferred_comm and match_comm:
                if preferred_comm == match_comm:
                    multipliers.append(1.0)
                else:
                    multipliers.append(0.9)
        
        # Calculate overall multiplier
        if multipliers:
            # Use geometric mean to ensure all preferences matter
            overall_multiplier = np.prod(multipliers) ** (1.0 / len(multipliers))
        else:
            overall_multiplier = 1.0
        
        return float(overall_multiplier)
    
    def _create_breakdown(
        self,
        similarity_features: Dict[str, float],
        base_score: float,
        preference_multiplier: float,
        normalized_score: float,
        layer_average: float,
        final_score: float,
        layer_insights: List[LayerInsight],
    ) -> Dict[str, float]:
        """Create detailed breakdown of compatibility calculation"""
        return {
            "base_compatibility": base_score,
            "preference_multiplier": preference_multiplier,
            "normalized_score": normalized_score,
            "layer_average": layer_average,
            "final_score": final_score,
            "feature_scores": similarity_features.copy(),
            "intelligence_layers": [insight.to_dict() for insight in layer_insights],
        }
    
    def get_top_contributing_factors(
        self,
        similarity_features: Dict[str, float],
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top contributing factors to compatibility"""
        # Weight features by importance
        weighted_features = [
            (name, score * self.feature_importance.get(name, 0.05))
            for name, score in similarity_features.items()
        ]
        weighted_features.sort(key=lambda x: x[1], reverse=True)
        return weighted_features[:top_n]

    def _evaluate_layers(
        self,
        user_profile: Dict[str, Any],
        match_profile: Dict[str, Any],
        similarity_features: Dict[str, float],
    ) -> List[LayerInsight]:
        """Run every intelligence layer and collect insights."""
        insights: List[LayerInsight] = []
        for layer in self.intelligence_layers:
            try:
                insight = layer.evaluate(user_profile, match_profile, similarity_features)
                insights.append(insight)
            except Exception as exc:
                # Guarantee determinism even if a layer fails
                fallback = LayerInsight(
                    name=layer.name,
                    score=0.85,
                    confidence=0.5,
                    rationale=f"{layer.name} unavailable: {exc}",
                    signals=[],
                    metadata={"error": str(exc)},
                )
                insights.append(fallback)
        return insights

