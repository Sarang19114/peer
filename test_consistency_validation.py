"""
Test script to validate consistency improvements in the matchmaking system.
Verifies:
- No clustered/identical scores
- Realistic geographic validation
- Proper preference penalties
- Natural variance in metrics
- Score separation between quality tiers
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.match_service import MatchService
from collections import Counter
import statistics


def test_consistency_validation():
    """Test the consistency validation pass"""
    print("=" * 80)
    print("CONSISTENCY VALIDATION TEST")
    print("=" * 80)
    
    # Initialize service
    print("\n1. Initializing match service...")
    service = MatchService()
    
    # Test profile with specific preferences
    test_profile = {
        "age": 28,
        "location": "Mumbai, India",
        "gender": "Male",
        "total_distance_km": 2500,
        "avg_daily_recovery_score": 75,
        "total_activities": 150,
        "injury_count": 2,
        "social_engagement_score": 7.5,
        "communication_preference": "Text-focused",
        "primary_interest": "Running",
        "experience_level": "Intermediate",
    }
    
    test_preferences = {
        "preferred_age_min": 25,
        "preferred_age_max": 35,
        "preferred_gender": "Male",
        "preferred_location": "Mumbai",
        "preferred_interests": ["Running", "Cycling"],
    }
    
    print("\n2. Finding matches with preferences...")
    matches = service.find_matches_from_profile(
        user_profile=test_profile,
        user_preferences=test_preferences,
        max_results=10
    )
    
    print(f"\n3. Analyzing {len(matches)} matches...\n")
    
    # Test 1: Score variance (no clustering)
    print("TEST 1: Score Variance")
    print("-" * 40)
    scores = [m["match_score"] for m in matches]
    rounded_scores = [round(s, 2) for s in scores]
    score_counts = Counter(rounded_scores)
    duplicates = {score: count for score, count in score_counts.items() if count > 1}
    
    if duplicates:
        print(f"❌ FAILED: Found {len(duplicates)} duplicate scores: {duplicates}")
    else:
        print("✅ PASSED: All scores are unique")
    
    if len(scores) > 1:
        variance = statistics.variance(scores)
        std_dev = statistics.stdev(scores)
        print(f"   Score variance: {variance:.6f}")
        print(f"   Standard deviation: {std_dev:.4f}")
        if std_dev < 0.01:
            print(f"   ⚠️  WARNING: Very low variance, scores may be too clustered")
        elif std_dev > 0.05:
            print(f"   ✅ Good variance: Scores are well-distributed")
        else:
            print(f"   ℹ️  Moderate variance: Acceptable spread")
    
    # Test 2: Score separation
    print("\n\nTEST 2: Score Separation")
    print("-" * 40)
    if len(scores) >= 2:
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        print(f"   Score range: {min_score:.3f} to {max_score:.3f} (Δ{score_range:.3f})")
        
        if score_range < 0.05:
            print(f"   ❌ FAILED: Range too narrow, matches not differentiated")
        elif score_range > 0.15:
            print(f"   ✅ PASSED: Excellent separation between matches")
        else:
            print(f"   ⚠️  WARNING: Moderate separation, could be better")
    
    # Test 3: Geographic validation
    print("\n\nTEST 3: Geographic Validation")
    print("-" * 40)
    user_location = test_profile["location"].lower()
    print(f"   User location: {test_profile['location']}")
    
    impossible_matches = []
    for match in matches:
        match_location = match["location"].lower()
        # Check for intercontinental mismatches
        incompatible_pairs = [
            ("india", "spain"),
            ("india", "poland"),
            ("india", "italy"),
            ("india", "usa"),
        ]
        
        for loc1, loc2 in incompatible_pairs:
            if loc1 in user_location and loc2 in match_location:
                impossible_matches.append({
                    "name": match["name"],
                    "location": match["location"],
                    "score": match["match_score"]
                })
            elif loc2 in user_location and loc1 in match_location:
                impossible_matches.append({
                    "name": match["name"],
                    "location": match["location"],
                    "score": match["match_score"]
                })
    
    if impossible_matches:
        print(f"   ⚠️  Found {len(impossible_matches)} geographically impossible matches:")
        for m in impossible_matches:
            print(f"      - {m['name']} in {m['location']} (score: {m['score']:.3f})")
            if m["score"] > 0.50:
                print(f"        ❌ FAILED: Score too high for impossible match")
            else:
                print(f"        ✅ PASSED: Properly penalized")
    else:
        print("   ✅ PASSED: No impossible geographic matches")
    
    # Test 4: Preference penalties
    print("\n\nTEST 4: Preference Penalty Enforcement")
    print("-" * 40)
    gender_mismatches = []
    age_violations = []
    
    for match in matches:
        # Check gender mismatch
        if match["gender"].lower() != test_preferences["preferred_gender"].lower():
            gender_mismatches.append({
                "name": match["name"],
                "gender": match["gender"],
                "score": match["match_score"]
            })
        
        # Check age range
        if not (test_preferences["preferred_age_min"] <= match["age"] <= test_preferences["preferred_age_max"]):
            age_violations.append({
                "name": match["name"],
                "age": match["age"],
                "score": match["match_score"]
            })
    
    if gender_mismatches:
        print(f"   Found {len(gender_mismatches)} gender mismatches:")
        for m in gender_mismatches:
            print(f"      - {m['name']} ({m['gender']}) score: {m['score']:.3f}")
            if m["score"] > 0.50:
                print(f"        ❌ FAILED: Insufficient penalty for gender mismatch")
            else:
                print(f"        ✅ PASSED: Properly penalized")
    else:
        print("   ✅ PASSED: All matches respect gender preference")
    
    if age_violations:
        print(f"   Found {len(age_violations)} age range violations:")
        for m in age_violations:
            print(f"      - {m['name']} (age {m['age']}) score: {m['score']:.3f}")
            if m["score"] > 0.60:
                print(f"        ❌ FAILED: Insufficient penalty for age mismatch")
            else:
                print(f"        ✅ PASSED: Properly penalized")
    else:
        print("   ✅ PASSED: All matches within age range")
    
    # Test 5: Natural variation in metrics
    print("\n\nTEST 5: Natural Variation in Metrics")
    print("-" * 40)
    activities = [m["profile_summary"]["total_activities"] for m in matches]
    distances = [m["profile_summary"]["total_distance_km"] for m in matches]
    
    activities_unique = len(set(activities))
    distances_unique = len(set(distances))
    
    print(f"   Unique activity counts: {activities_unique}/{len(activities)}")
    print(f"   Unique distance values: {distances_unique}/{len(distances)}")
    
    if activities_unique < len(activities) * 0.5:
        print(f"   ⚠️  WARNING: Low activity count variation")
    else:
        print(f"   ✅ PASSED: Good activity variation")
    
    if distances_unique < len(distances) * 0.5:
        print(f"   ⚠️  WARNING: Low distance variation")
    else:
        print(f"   ✅ PASSED: Good distance variation")
    
    # Test 6: Explanation quality
    print("\n\nTEST 6: Explanation Realism")
    print("-" * 40)
    generic_phrases = [
        "excellent match",
        "perfect alignment",
        "ideal compatibility",
        "flawless match",
        "ensuring smooth interactions",
    ]
    
    explanations_with_generic = []
    for match in matches:
        summary = match["explanation"]["summary"].lower()
        found_generic = [phrase for phrase in generic_phrases if phrase in summary]
        if found_generic:
            explanations_with_generic.append({
                "name": match["name"],
                "phrases": found_generic
            })
    
    if explanations_with_generic:
        print(f"   ⚠️  Found {len(explanations_with_generic)} explanations with generic phrases:")
        for e in explanations_with_generic[:3]:
            print(f"      - {e['name']}: {e['phrases']}")
    else:
        print("   ✅ PASSED: No overly generic phrases detected")
    
    # Display top 3 matches
    print("\n\n" + "=" * 80)
    print("TOP 3 MATCHES")
    print("=" * 80)
    for i, match in enumerate(matches[:3], 1):
        print(f"\n{i}. {match['name']} (ID: {match['person_id']})")
        print(f"   Score: {match['match_score']:.3f} ({match['match_score']*100:.1f}%)")
        print(f"   Age: {match['age']}, Gender: {match['gender']}")
        print(f"   Location: {match['location']}")
        print(f"   Experience: {match['profile_summary']['experience_level']}")
        print(f"   Interest: {match['profile_summary']['primary_interest']}")
        print(f"   Activities: {match['profile_summary']['total_activities']}")
        print(f"   Distance: {match['profile_summary']['total_distance_km']:.0f}km")
        print(f"\n   Explanation:")
        explanation_lines = match['explanation']['summary'].split('. ')
        for line in explanation_lines[:3]:
            if line.strip():
                print(f"   • {line.strip()}.")
    
    print("\n" + "=" * 80)
    print("CONSISTENCY VALIDATION TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_consistency_validation()
