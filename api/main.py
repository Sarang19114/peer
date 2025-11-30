"""
FastAPI backend for peer matchmaking system.
Provides endpoints for profile-based matching with dynamic explanations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.match_service import MatchService

app = FastAPI(
    title="Peer Matchmaking API",
    description="API for finding compatible matches based on user profiles",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize match service
match_service = MatchService()


class UserProfile(BaseModel):
    """User profile model for matchmaking"""
    # Basic Information
    age: int = Field(..., ge=18, le=100, description="User's age")
    location: str = Field(..., min_length=1, description="User's location/city")
    gender: str = Field(..., description="User's gender")
    
    # Activity & Lifestyle
    total_distance_km: Optional[float] = Field(None, ge=0, description="Total activity distance in km")
    total_activities: Optional[int] = Field(None, ge=0, description="Total number of activities")
    avg_daily_recovery_score: Optional[float] = Field(None, ge=0, le=100, description="Average recovery score")
    
    # Personality & Preferences
    social_engagement_score: Optional[float] = Field(None, ge=0, le=10, description="Social engagement level (0-10)")
    communication_preference: Optional[str] = Field(None, description="Preferred communication style")
    primary_interest: Optional[str] = Field(None, description="Primary interest or activity")
    experience_level: Optional[str] = Field(None, description="Experience level (Beginner, Intermediate, Advanced, Expert)")
    
    # Additional Details
    injury_count: Optional[int] = Field(None, ge=0, description="Number of past injuries")
    years_experience: Optional[int] = Field(None, ge=0, description="Years of experience in primary interest")
    
    # Personality Traits (optional)
    personality_traits: Optional[List[str]] = Field(None, description="List of personality traits")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Additional preferences")


class UserPreferences(BaseModel):
    """User preferences for matching"""
    preferred_age_min: Optional[int] = Field(None, ge=18, le=100, description="Minimum preferred age")
    preferred_age_max: Optional[int] = Field(None, ge=18, le=100, description="Maximum preferred age")
    preferred_gender: Optional[str] = Field(None, description="Preferred gender (or 'Any')")
    preferred_location: Optional[str] = Field(None, description="Preferred location/city")
    preferred_interests: Optional[List[str]] = Field(None, description="List of preferred interests")
    preferred_experience_level: Optional[str] = Field(None, description="Preferred experience level")
    preferred_communication_style: Optional[str] = Field(None, description="Preferred communication style")
    location_radius_km: Optional[float] = Field(None, ge=0, description="Location radius in km (for future use)")


class MatchRequest(BaseModel):
    """Complete match request with profile and preferences"""
    profile: UserProfile
    preferences: Optional[UserPreferences] = None


class IntelligenceLayerSignal(BaseModel):
    label: str
    value: str
    weight: float = Field(..., ge=0, le=1)


class IntelligenceLayer(BaseModel):
    name: str
    score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    rationale: str
    signals: List[IntelligenceLayerSignal]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MatchExplanation(BaseModel):
    """Explanation for a match"""
    summary: str
    detailed_breakdown: List[Dict[str, Any]]
    top_reasons: List[str]


class MatchResult(BaseModel):
    """Single match result with explanation"""
    person_id: str
    name: str
    age: int
    location: str
    gender: str
    match_score: float
    match_probability: float
    explanation: MatchExplanation
    profile_summary: Dict[str, Any]
    similarity_features: Dict[str, float]
    compatibility_breakdown: Optional[Dict[str, Any]] = None
    intelligence_layers: List[IntelligenceLayer]


class MatchResponse(BaseModel):
    """Response model for matches"""
    matches: List[MatchResult]
    total_found: int
    model_used: str
    user_profile_summary: Dict[str, Any]


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Peer Matchmaking API",
        "version": "2.0.0",
        "endpoints": {
            "find_matches": "/api/matches",
            "get_person": "/api/person/{person_id}",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "peer-matchmaking",
        "version": "2.0.0"
    }


@app.post("/api/matches", response_model=MatchResponse)
async def find_matches(request: MatchRequest):
    """Find top 5 matches based on user profile and preferences"""
    try:
        # Convert to dicts (using model_dump for Pydantic v2 compatibility)
        profile_dict = request.profile.model_dump(exclude_none=True)
        preferences_dict = request.preferences.model_dump(exclude_none=True) if request.preferences else None
        
        print(f"Received profile: {profile_dict}")
        if preferences_dict:
            print(f"Received preferences: {preferences_dict}")
        
        # Find matches
        matches = match_service.find_matches_from_profile(
            user_profile=profile_dict,
            user_preferences=preferences_dict,
            max_results=5,
            model_type="logistic"
        )
        
        print(f"Found {len(matches)} matches")
        if matches:
            print(f"Top match score: {matches[0]['match_probability']:.2%}")
        
        # Create response
        return MatchResponse(
            matches=matches,
            total_found=len(matches),
            model_used="logistic",
            user_profile_summary={
                "age": request.profile.age,
                "location": request.profile.location,
                "gender": request.profile.gender,
            }
        )
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in find_matches: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/person/{person_id}")
async def get_person(person_id: str):
    """Get detailed information about a specific person"""
    try:
        person = match_service.get_person_profile(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        return person
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
