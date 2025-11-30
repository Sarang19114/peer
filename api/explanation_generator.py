"""
Dynamic explanation generator for match results.
Generates natural language explanations based on feature similarities and model reasoning.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class ExplanationGenerator:
    """Generates dynamic explanations for match results based on feature analysis"""
    
    def __init__(self):
        """Initialize the explanation generator"""
        self.feature_descriptions = {
            "age_similarity": "age",
            "distance_similarity": "activity level and training volume",
            "recovery_similarity": "recovery patterns and wellness approach",
            "activity_frequency_similarity": "activity frequency and consistency",
            "injury_experience_match": "injury history and recovery experience",
            "social_alignment": "social engagement and community involvement",
            "communication_compatibility": "communication style and preferences",
            "network_overlap": "social network size and connections",
            "coaching_alignment": "coaching and support preferences",
            "partner_support_similarity": "training partner preferences",
            "life_event_similarity": "life experiences and background",
            "residence_similarity": "residential history and mobility",
            "resting_hr_alignment": "fitness level and physiological markers",
        }
    
    def generate_explanation(
        self,
        user_profile: Dict,
        match_profile: Dict,
        similarity_features: Dict[str, float],
        match_score: float,
        top_features: List[Tuple[str, float]],
        layer_insights: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a natural language explanation for why this is a good match.
        
        Args:
            user_profile: User's profile data
            match_profile: Matched person's profile data
            similarity_features: Dictionary of similarity scores
            match_score: Overall match score
            top_features: List of (feature_name, score) tuples for top contributing features
        
        Returns:
            Natural language explanation string
        """
        explanations = []
        
        # Overall match quality with realistic language
        if match_score >= 0.90:
            quality = "very high compatibility"
            prefix = "shows"
        elif match_score >= 0.80:
            quality = "strong compatibility"
            prefix = "indicates"
        elif match_score >= 0.70:
            quality = "good potential alignment"
            prefix = "suggests"
        elif match_score >= 0.55:
            quality = "moderate compatibility"
            prefix = "shows"
        else:
            quality = "limited alignment"
            prefix = "indicates"
        
        explanations.append(
            f"Analysis {prefix} {quality} at {match_score:.1%}."
        )
        
        # Top contributing factors with realistic assessment
        if top_features:
            top_factors = []
            for feature_name, score in top_features[:3]:  # Top 3 factors
                if score > 0.7:
                    factor_desc = self.feature_descriptions.get(
                        feature_name,
                        feature_name.replace("_", " ")
                    )
                    top_factors.append(f"{factor_desc} ({score * 100:.0f}%)")
            
            if top_factors:
                if len(top_factors) == 1:
                    explanations.append(
                        f"Primary alignment factor: {top_factors[0]}."
                    )
                elif len(top_factors) == 2:
                    explanations.append(
                        f"Key alignment factors: {top_factors[0]}, {top_factors[1]}."
                    )
                else:
                    explanations.append(
                        f"Main compatibility drivers: {top_factors[0]}, {top_factors[1]}, {top_factors[2]}."
                    )
        
        # Intelligence layers - only include if score is notable
        if layer_insights:
            best_layer = max(layer_insights, key=lambda layer: layer.get("score", 0), default=None)
            if best_layer and best_layer.get("score", 0) >= 0.75:
                explanations.append(best_layer.get("rationale", "Layer rationale unavailable."))
                temporal_metadata = best_layer.get("metadata", {})
                if temporal_metadata and temporal_metadata.get("predictions"):
                    horizon = max(temporal_metadata["predictions"], key=lambda p: p.get("probability", 0.0))
                    explanations.append(
                        f"Temporal model: {horizon['probability'] * 100:.0f}% retention at {horizon['horizon']}."
                    )

        # Specific comparisons
        specific_reasons = self._generate_specific_reasons(
            user_profile, match_profile, similarity_features
        )
        if specific_reasons:
            explanations.extend(specific_reasons)
        
        # Personality and lifestyle alignment
        personality_match = self._analyze_personality_match(
            user_profile, match_profile, similarity_features
        )
        if personality_match:
            explanations.append(personality_match)
        
        return " ".join(explanations)
    
    def _generate_specific_reasons(
        self,
        user_profile: Dict,
        match_profile: Dict,
        similarity_features: Dict[str, float]
    ) -> List[str]:
        """Generate specific reasons based on profile comparisons"""
        reasons = []
        
        # Age similarity
        if "age_similarity" in similarity_features:
            age_sim = similarity_features["age_similarity"]
            if age_sim > 0.8:
                user_age = user_profile.get("age")
                match_age = match_profile.get("age")
                if user_age is not None and match_age is not None:
                    try:
                        user_age = float(user_age)
                        match_age = float(match_age)
                        age_diff = abs(user_age - match_age)
                        if age_diff <= 2:
                            reasons.append(f"Age difference minimal ({int(age_diff)}y), likely similar life context.")
                        elif age_diff <= 5:
                            reasons.append(f"Ages within {int(age_diff)} years, reasonable life stage overlap.")
                    except (ValueError, TypeError):
                        pass
        
        # Location similarity
        user_loc = str(user_profile.get("location", "")).strip()
        match_loc = str(match_profile.get("location", "")).strip()
        if user_loc and match_loc and user_loc.lower() == match_loc.lower():
            reasons.append(f"Same location ({user_loc}), coordination feasible.")
        
        # Activity level alignment
        if "distance_similarity" in similarity_features:
            dist_sim = similarity_features["distance_similarity"]
            if dist_sim > 0.75:
                user_km = user_profile.get("total_distance_km", 0)
                match_km = match_profile.get("total_distance_km", 0)
                if user_km and match_km:
                    reasons.append(f"Training volumes comparable ({user_km:.0f}km vs {match_km:.0f}km).")
        
        # Communication style
        if "communication_compatibility" in similarity_features:
            comm_sim = similarity_features["communication_compatibility"]
            if comm_sim > 0.8:
                user_comm = user_profile.get("communication_preference", "")
                # Try both possible column names
                match_comm = match_profile.get("communication_style") or match_profile.get("communication_preference", "")
                if user_comm and match_comm:
                    if user_comm.lower() == match_comm.lower():
                        reasons.append(f"Communication preferences match ({match_comm.lower()}).")
        
        # Social alignment
        if "social_alignment" in similarity_features:
            social_sim = similarity_features["social_alignment"]
            if social_sim > 0.75:
                user_social = user_profile.get("social_engagement_score", 5)
                match_social = match_profile.get("social_engagement_score", 5)
                if user_social and match_social:
                    reasons.append(f"Social engagement similar ({user_social:.1f}/10 vs {match_social:.1f}/10).")
        
        return reasons
    
    def _analyze_personality_match(
        self,
        user_profile: Dict,
        match_profile: Dict,
        similarity_features: Dict[str, float]
    ) -> str:
        """Analyze personality and lifestyle compatibility"""
        high_similarities = [
            (name, score)
            for name, score in similarity_features.items()
            if score > 0.7
        ]
        
        if len(high_similarities) >= 5:
            return f"Multi-dimensional alignment across {len(high_similarities)} factors."
        elif len(high_similarities) >= 3:
            return f"Moderate alignment in {len(high_similarities)} key areas."
        elif len(high_similarities) >= 1:
            return "Limited overlap, may have complementary differences."
        else:
            return "Weak feature alignment, compatibility uncertain."
    
    def generate_detailed_breakdown(
        self,
        similarity_features: Dict[str, float]
    ) -> List[Dict[str, any]]:
        """Generate detailed breakdown of all similarity features"""
        breakdown = []
        
        for feature_name, score in sorted(
            similarity_features.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            feature_desc = self.feature_descriptions.get(
                feature_name,
                feature_name.replace("_", " ").title()
            )
            
            if score >= 0.85:
                level = "Very High"
                color = "success"
            elif score >= 0.70:
                level = "High"
                color = "info"
            elif score >= 0.50:
                level = "Moderate"
                color = "warning"
            elif score >= 0.30:
                level = "Low"
                color = "default"
            else:
                level = "Very Low"
                color = "error"
            
            breakdown.append({
                "feature": feature_desc,
                "score": float(score),
                "percentage": f"{score * 100:.1f}%",
                "level": level,
                "color": color,
            })
        
        return breakdown

