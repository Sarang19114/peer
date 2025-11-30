"""
Improved Feature Engineering for Athlete Peer Matching
======================================================

This module addresses the shortcomings identified in the initial prototype by:
    • Generating similarity features with real variance (no constant columns)
    • Providing diagnostic statistics (mean/std/range) for every feature
    • Returning the raw pair dataframe for downstream analysis/debugging

All feature values are scaled to the [0, 1] range for interpretability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FeatureSet:
    """Container for engineered feature artefacts."""

    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    pair_ids: List[Tuple[str, str]]
    df_pairs: pd.DataFrame
    feature_stats: pd.DataFrame


class ImprovedFeatureEngineer:
    """
    Generates informative pairwise features between athletes.

    The class is deterministic (via random_state) to support reproducible
    experiments while still sampling a diverse set of candidate peers.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.feature_names: List[str] = [
            "age_similarity",
            "distance_similarity",
            "recovery_similarity",
            "activity_frequency_similarity",
            "injury_experience_match",
            "social_alignment",
            "communication_compatibility",
            "network_overlap",
            "coaching_alignment",
            "partner_support_similarity",
            "life_event_similarity",
            "residence_similarity",
            "resting_hr_alignment",
        ]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def create_features(
        self,
        df_athletes: pd.DataFrame,
        n_pairs_per_athlete: int = 12,
    ) -> FeatureSet:
        """
        Build a synthetic pair dataset with diagnostic statistics.

        Args:
            df_athletes: Master athlete dataframe.
            n_pairs_per_athlete: Number of peers sampled for each athlete.

        Returns:
            FeatureSet dataclass containing engineered artefacts.
        """
        if df_athletes.empty:
            raise ValueError("Athlete dataframe is empty; cannot create features.")

        pairs: List[Dict[str, float]] = []

        for idx, athlete in df_athletes.iterrows():
            candidate_indices = np.delete(np.arange(len(df_athletes)), idx)
            n_samples = min(n_pairs_per_athlete, len(candidate_indices))

            sampled_indices = self._rng.choice(
                candidate_indices,
                size=n_samples,
                replace=False,
            )

            for comp_idx in sampled_indices:
                peer = df_athletes.iloc[comp_idx]
                features = self._compute_features_with_variance(athlete, peer)
                compatibility = self._calculate_true_score(athlete, peer)
                label = 1 if compatibility > 0.55 else 0

                pairs.append(
                    {
                        "athlete1_id": athlete["athlete_id"],
                        "athlete2_id": peer["athlete_id"],
                        **features,
                        "true_match_score": compatibility,
                        "label": label,
                    }
                )

        df_pairs = pd.DataFrame(pairs)

        if df_pairs.empty:
            raise RuntimeError("No athlete pairs were generated; check input data.")

        feature_stats = (
            df_pairs[self.feature_names]
            .agg(["mean", "std", "min", "max", "nunique"])
            .T.rename(
                columns={
                    "mean": "mean",
                    "std": "std",
                    "min": "min",
                    "max": "max",
                    "nunique": "unique_values",
                }
            )
        )

        pair_ids = list(
            zip(df_pairs["athlete1_id"].tolist(), df_pairs["athlete2_id"].tolist())
        )

        return FeatureSet(
            X=df_pairs[self.feature_names].to_numpy(),
            y=df_pairs["label"].to_numpy(),
            feature_names=self.feature_names,
            pair_ids=pair_ids,
            df_pairs=df_pairs,
            feature_stats=feature_stats,
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _safe_value(series: pd.Series, column: str, default: float = 0.0) -> float:
        """Return a numeric value from the series, guarding against NaNs."""
        value = series.get(column, default)
        if pd.isna(value):
            return default
        return float(value)

    def _compute_features_with_variance(
        self, athlete1: pd.Series, athlete2: pd.Series
    ) -> Dict[str, float]:
        """
        Compute similarity metrics that vary between 0 and 1.
        """

        age1 = self._safe_value(athlete1, "age")
        age2 = self._safe_value(athlete2, "age")
        age_diff = abs(age1 - age2)
        age_similarity = max(0.0, 1.0 - age_diff / 20.0)

        dist1 = self._safe_value(athlete1, "total_distance_km")
        dist2 = self._safe_value(athlete2, "total_distance_km")
        distance_similarity = max(0.0, 1.0 - abs(dist1 - dist2) / 1200.0)

        recovery1 = self._safe_value(athlete1, "avg_daily_recovery_score")
        recovery2 = self._safe_value(athlete2, "avg_daily_recovery_score")
        recovery_similarity = max(0.0, 1.0 - abs(recovery1 - recovery2) / 40.0)

        activities1 = self._safe_value(athlete1, "total_activities")
        activities2 = self._safe_value(athlete2, "total_activities")
        activity_frequency_similarity = max(
            0.0, 1.0 - abs(activities1 - activities2) / 60.0
        )

        injuries1 = self._safe_value(athlete1, "injury_count")
        injuries2 = self._safe_value(athlete2, "injury_count")
        injury_experience_match = max(0.0, 1.0 - abs(injuries1 - injuries2) / 5.0)

        social1 = self._safe_value(athlete1, "social_engagement_score", default=5.0)
        social2 = self._safe_value(athlete2, "social_engagement_score", default=5.0)
        social_alignment = max(0.0, 1.0 - abs(social1 - social2) / 10.0)

        # Try both possible field names for communication preference
        comm1 = str(athlete1.get("communication_preference") or athlete1.get("communication_style", "")).strip().lower()
        comm2 = str(athlete2.get("communication_preference") or athlete2.get("communication_style", "")).strip().lower()
        communication_compatibility = 1.0 if comm1 == comm2 and comm1 else 0.7

        network1 = self._safe_value(athlete1, "social_network_size", default=10.0)
        network2 = self._safe_value(athlete2, "social_network_size", default=10.0)
        network_overlap = max(0.0, 1.0 - abs(network1 - network2) / 40.0)

        coaching1 = self._safe_value(athlete1, "coaching_support", default=0.0)
        coaching2 = self._safe_value(athlete2, "coaching_support", default=0.0)
        coaching_alignment = max(0.0, 1.0 - abs(coaching1 - coaching2) / 3.0)

        partners1 = self._safe_value(athlete1, "training_partners", default=0.0)
        partners2 = self._safe_value(athlete2, "training_partners", default=0.0)
        partner_support_similarity = max(
            0.0, 1.0 - abs(partners1 - partners2) / 6.0
        )

        life_events1 = self._safe_value(athlete1, "total_life_events", default=10.0)
        life_events2 = self._safe_value(athlete2, "total_life_events", default=10.0)
        life_event_similarity = max(0.0, 1.0 - abs(life_events1 - life_events2) / 25.0)

        residences1 = self._safe_value(athlete1, "residences_count", default=1.0)
        residences2 = self._safe_value(athlete2, "residences_count", default=1.0)
        residence_similarity = max(0.0, 1.0 - abs(residences1 - residences2) / 6.0)

        resting_hr1 = self._safe_value(athlete1, "avg_resting_hr", default=60.0)
        resting_hr2 = self._safe_value(athlete2, "avg_resting_hr", default=60.0)
        resting_hr_alignment = max(0.0, 1.0 - abs(resting_hr1 - resting_hr2) / 25.0)

        return {
            "age_similarity": age_similarity,
            "distance_similarity": distance_similarity,
            "recovery_similarity": recovery_similarity,
            "activity_frequency_similarity": activity_frequency_similarity,
            "injury_experience_match": injury_experience_match,
            "social_alignment": social_alignment,
            "communication_compatibility": communication_compatibility,
            "network_overlap": network_overlap,
            "coaching_alignment": coaching_alignment,
            "partner_support_similarity": partner_support_similarity,
            "life_event_similarity": life_event_similarity,
            "residence_similarity": residence_similarity,
            "resting_hr_alignment": resting_hr_alignment,
        }

    def _calculate_true_score(
        self, athlete1: pd.Series, athlete2: pd.Series
    ) -> float:
        """
        Generate a soft compatibility score (0-1) used for labelling.
        """

        age_score = max(
            0.0,
            1.0
            - abs(
                self._safe_value(athlete1, "age")
                - self._safe_value(athlete2, "age")
            )
            / 15.0,
        )

        volume_score = max(
            0.0,
            1.0
            - abs(
                self._safe_value(athlete1, "total_distance_km")
                - self._safe_value(athlete2, "total_distance_km")
            )
            / 900.0,
        )

        recovery_score = max(
            0.0,
            1.0
            - abs(
                self._safe_value(athlete1, "avg_daily_recovery_score")
                - self._safe_value(athlete2, "avg_daily_recovery_score")
            )
            / 30.0,
        )

        activity_score = max(
            0.0,
            1.0
            - abs(
                self._safe_value(athlete1, "total_activities")
                - self._safe_value(athlete2, "total_activities")
            )
            / 25.0,
        )

        injury_match = (
            1.0
            if (
                (self._safe_value(athlete1, "injury_count") > 0)
                == (self._safe_value(athlete2, "injury_count") > 0)
            )
            else 0.5
        )

        compatibility = (
            0.30 * age_score
            + 0.25 * volume_score
            + 0.20 * recovery_score
            + 0.15 * activity_score
            + 0.10 * injury_match
        )

        adjusted = compatibility - 0.1
        return float(np.clip(adjusted, 0.0, 1.0))


