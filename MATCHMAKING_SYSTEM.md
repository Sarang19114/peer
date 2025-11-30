# Peer Matchmaking System

A complete, general-purpose peer matchmaking platform that finds compatible matches based on user profiles with dynamic, AI-generated explanations.

## System Overview

The system has been completely refactored from an athlete-specific matching tool to a general-purpose matchmaking platform. Users input their profile information, and the system automatically finds the top 5 most compatible matches with detailed, dynamically generated explanations.

## Key Features

### 1. **Profile-Based Matching**
- Users input their complete profile through a multi-step form
- System evaluates the profile against the entire database
- Returns top 5 matches automatically (no search parameters needed)

### 2. **Dynamic Explanation Generation**
- Every match includes a natural language explanation
- Explanations are generated based on actual feature similarities
- No hard-coded explanations - everything is computed dynamically
- Includes:
  - Overall match summary
  - Top contributing factors
  - Specific compatibility reasons
  - Detailed breakdown of all similarity features

### 3. **Professional Frontend**
- Clean, modern UI built with Material-UI
- Multi-step form with validation
- Responsive design for all devices
- Intuitive user experience
- Visual match cards with expandable details

### 4. **Robust Backend**
- FastAPI-based REST API
- Profile processing and matching
- Explanation generation engine
- Model integration for accurate scoring

## Architecture

### Backend (`api/`)

#### `main.py`
- FastAPI application
- `/api/matches` - POST endpoint for finding matches
- `/api/person/{id}` - GET endpoint for person details
- `/api/health` - Health check endpoint

#### `match_service.py`
- Core matching logic
- Profile processing
- Feature computation
- Model integration
- Match ranking and selection

#### `explanation_generator.py`
- Dynamic explanation generation
- Feature analysis
- Natural language generation
- Compatibility reasoning

### Frontend (`frontend/src/`)

#### `components/ProfileForm.tsx`
- Multi-step profile input form
- Form validation
- Step navigation
- Match submission

#### `components/MatchResults.tsx`
- Match display component
- Explanation visualization
- Detailed breakdown view
- Responsive card layout

#### `api/client.ts`
- API client with TypeScript types
- Request/response handling
- Error management

## User Flow

1. **Profile Input**
   - Step 1: Basic Information (age, location, gender)
   - Step 2: Activity & Lifestyle (distance, activities, recovery, social engagement)
   - Step 3: Personality & Preferences (communication, interests, experience level)

2. **Match Generation**
   - System processes user profile
   - Computes similarity with all people in database
   - Ranks matches by compatibility score
   - Generates dynamic explanations for each match

3. **Results Display**
   - Top 5 matches displayed as cards
   - Each card shows:
     - Match score and probability
     - Natural language explanation
     - Top compatibility factors
     - Detailed breakdown (expandable)
     - Profile summary

## Explanation System

The explanation generator creates dynamic, contextual explanations based on:

1. **Overall Match Quality**: Excellent, Strong, Good, or Moderate
2. **Top Contributing Factors**: The 3 highest similarity scores
3. **Specific Comparisons**: Age, location, activity level, communication style
4. **Personality Analysis**: Overall compatibility assessment
5. **Detailed Breakdown**: All 13 similarity features with scores

### Example Explanation

> "This is a strong match with a compatibility score of 78%. You share strong alignment in age, activity level and training volume, and social engagement. You're very close in age, which often leads to shared life experiences and perspectives. You have similar activity levels and training volumes, suggesting compatible lifestyles. Your profiles show strong overall compatibility across multiple dimensions."

## Data Model

### User Profile
```typescript
{
  age: number;
  location: string;
  gender: string;
  total_distance_km?: number;
  total_activities?: number;
  avg_daily_recovery_score?: number;
  social_engagement_score?: number;
  communication_preference?: string;
  primary_interest?: string;
  experience_level?: string;
  injury_count?: number;
  years_experience?: number;
}
```

### Match Result
```typescript
{
  person_id: string;
  name: string;
  age: number;
  location: string;
  gender: string;
  match_score: number;
  match_probability: number;
  explanation: {
    summary: string;
    detailed_breakdown: Array<{
      feature: string;
      score: number;
      percentage: string;
      level: string;
      color: string;
    }>;
    top_reasons: string[];
  };
  profile_summary: {...};
  similarity_features: {...};
}
```

## Running the System

### Backend
```bash
cd api
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

### POST `/api/matches`
Find matches based on user profile.

**Request:**
```json
{
  "age": 35,
  "location": "Stockholm",
  "gender": "Male",
  "total_distance_km": 2000,
  "social_engagement_score": 7.5,
  ...
}
```

**Response:**
```json
{
  "matches": [...],
  "total_found": 5,
  "model_used": "logistic",
  "user_profile_summary": {...}
}
```

## Technical Highlights

1. **No Hard-coded Explanations**: All explanations are generated dynamically based on actual data
2. **Feature-based Reasoning**: Explanations reference specific similarity scores
3. **Contextual Analysis**: Explanations adapt based on match quality and feature combinations
4. **Scalable Architecture**: Easy to add new features or explanation types
5. **Type Safety**: Full TypeScript coverage for frontend
6. **Error Handling**: Comprehensive error handling throughout

## Future Enhancements

- Machine learning-based explanation generation
- Personality trait analysis
- Preference learning from user feedback
- Advanced filtering options
- Match history and tracking
- Real-time notifications

