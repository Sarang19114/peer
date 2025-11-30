/**
 * API client for peer matchmaking service
 */

export interface UserProfile {
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
  personality_traits?: string[];
  preferences?: Record<string, any>;
}

export interface UserPreferences {
  preferred_age_min?: number;
  preferred_age_max?: number;
  preferred_gender?: string;
  preferred_location?: string;
  preferred_interests?: string[];
  preferred_experience_level?: string;
  preferred_communication_style?: string;
  location_radius_km?: number;
}

export interface MatchExplanation {
  summary: string;
  detailed_breakdown: Array<{
    feature: string;
    score: number;
    percentage: string;
    level: string;
    color: string;
  }>;
  top_reasons: string[];
}

export interface IntelligenceLayerSignal {
  label: string;
  value: string;
  weight: number;
}

export interface IntelligenceLayer {
  name: string;
  score: number;
  confidence: number;
  rationale: string;
  signals: IntelligenceLayerSignal[];
  metadata?: Record<string, any>;
}

export interface CompatibilityBreakdown {
  base_compatibility: number;
  preference_multiplier: number;
  normalized_score: number;
  layer_average: number;
  final_score: number;
  feature_scores: Record<string, number>;
  intelligence_layers?: IntelligenceLayer[];
}

export interface MatchResult {
  person_id: string;
  name: string;
  age: number;
  location: string;
  gender: string;
  match_score: number;
  match_probability: number;
  explanation: MatchExplanation;
  compatibility_breakdown: CompatibilityBreakdown;
  intelligence_layers: IntelligenceLayer[];
  profile_summary: {
    total_distance_km: number;
    total_activities: number;
    primary_interest: string;
    experience_level: string;
    injury_count: number;
    social_engagement: number;
    communication_style?: string;
    years_experience?: number;
    nationality?: string;
    career_status?: string;
  };
  similarity_features: Record<string, number>;
}

export interface MatchResponse {
  matches: MatchResult[];
  total_found: number;
  model_used: string;
  user_profile_summary: {
    age: number;
    location: string;
    gender: string;
  };
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async findMatches(profile: UserProfile, preferences?: UserPreferences): Promise<MatchResponse> {
    const requestBody = {
      profile,
      preferences: preferences || null,
    };
    
    const response = await fetch(`${this.baseUrl}/api/matches`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async getPerson(personId: string) {
    const response = await fetch(`${this.baseUrl}/api/person/${personId}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }
}

export const apiClient = new ApiClient();

// Explicit re-exports for better module resolution
