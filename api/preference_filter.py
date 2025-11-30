"""
Preference filtering system for matchmaking.
Filters and ranks candidates based on user preferences.
"""

from typing import Dict, List, Optional
import pandas as pd


class PreferenceFilter:
    """Filters and ranks matches based on user preferences"""
    
    def filter_by_preferences(
        self,
        candidates: pd.DataFrame,
        preferences: Dict
    ) -> pd.DataFrame:
        """Filter candidates based on user preferences"""
        filtered = candidates.copy()
        
        # Age range filter
        if "preferred_age_min" in preferences and "preferred_age_max" in preferences:
            age_min = preferences.get("preferred_age_min", 18)
            age_max = preferences.get("preferred_age_max", 100)
            if age_min and age_max:
                filtered = filtered[
                    (filtered["age"] >= age_min) & 
                    (filtered["age"] <= age_max)
                ]
        
        # Gender filter
        if "preferred_gender" in preferences:
            preferred_gender = preferences.get("preferred_gender", "").strip()
            if preferred_gender and preferred_gender.lower() != "any":
                filtered = filtered[
                    filtered["gender"].str.lower() == preferred_gender.lower()
                ]
        
        # Location filter (fuzzy matching)
        if "preferred_location" in preferences:
            preferred_loc = preferences.get("preferred_location", "").strip().lower()
            if preferred_loc:
                filtered = filtered[
                    filtered["location"].str.lower().str.contains(
                        preferred_loc, na=False
                    ) | filtered["location"].str.lower().str.contains(
                        preferred_loc.split(",")[0] if "," in preferred_loc else preferred_loc,
                        na=False
                    )
                ]
        
        # Interest filter
        if "preferred_interests" in preferences:
            preferred_interests = preferences.get("preferred_interests", [])
            if preferred_interests:
                # Try both possible column names
                interest_col = None
                if "primary_sport" in filtered.columns:
                    interest_col = "primary_sport"
                elif "primary_interest" in filtered.columns:
                    interest_col = "primary_interest"
                
                if interest_col:
                    interest_filter = pd.Series([False] * len(filtered), index=filtered.index)
                    for interest in preferred_interests:
                        if interest:
                            interest_filter |= filtered[interest_col].str.lower().str.contains(
                                interest.lower(), na=False
                            )
                    filtered = filtered[interest_filter]
        
        # Experience level filter
        if "preferred_experience_level" in preferences:
            preferred_level = preferences.get("preferred_experience_level", "").strip()
            if preferred_level:
                # Try both possible column names
                level_col = None
                if "competition_level" in filtered.columns:
                    level_col = "competition_level"
                elif "experience_level" in filtered.columns:
                    level_col = "experience_level"
                
                if level_col:
                    filtered = filtered[
                        filtered[level_col].str.lower() == preferred_level.lower()
                    ]
        
        return filtered
    
    def calculate_preference_score(
        self,
        candidate: pd.Series,
        preferences: Dict
    ) -> float:
        """Calculate how well a candidate matches preferences (0-1)"""
        score = 1.0
        factors = []
        
        # Age match
        if "preferred_age_min" in preferences and "preferred_age_max" in preferences:
            age_min = preferences.get("preferred_age_min", 18)
            age_max = preferences.get("preferred_age_max", 100)
            candidate_age = candidate.get("age")
            if candidate_age and age_min and age_max:
                if age_min <= candidate_age <= age_max:
                    factors.append(1.0)
                else:
                    age_diff = min(abs(candidate_age - age_min), abs(candidate_age - age_max))
                    factors.append(max(0.0, 1.0 - (age_diff / 20.0)))
        
        # Gender match
        if "preferred_gender" in preferences:
            preferred = preferences.get("preferred_gender", "").strip().lower()
            if preferred and preferred != "any":
                candidate_gender = str(candidate.get("gender", "")).strip().lower()
                factors.append(1.0 if candidate_gender == preferred else 0.5)
        
        # Location match
        if "preferred_location" in preferences:
            preferred_loc = preferences.get("preferred_location", "").strip().lower()
            candidate_loc = str(candidate.get("location", "")).strip().lower()
            if preferred_loc and candidate_loc:
                if preferred_loc in candidate_loc or candidate_loc in preferred_loc:
                    factors.append(1.0)
                else:
                    factors.append(0.7)
        
        # Interest match
        if "preferred_interests" in preferences:
            preferred_interests = preferences.get("preferred_interests", [])
            # Try both possible column names
            candidate_interest = str(
                candidate.get("primary_sport") or candidate.get("primary_interest", "")
            ).strip().lower()
            if preferred_interests and candidate_interest:
                match_found = any(
                    pi.lower() in candidate_interest or candidate_interest in pi.lower()
                    for pi in preferred_interests if pi
                )
                factors.append(1.0 if match_found else 0.8)
        
        if factors:
            score = sum(factors) / len(factors)
        
        return float(score)

