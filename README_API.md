# Athlete Peer Matching API

FastAPI backend for finding athlete peer matches based on customizable parameters.

## Features

- RESTful API with comprehensive parameter support
- Multiple ML model support (Logistic Regression, Random Forest, Neural Network, etc.)
- Real-time match generation
- Detailed similarity feature computation
- CORS enabled for frontend integration

## Getting Started

### Prerequisites

- Python 3.8+
- Trained models in `models/` directory (run `run_pipeline.py` first)
- Data files in `data/raw/` directory

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure models are trained:
```bash
python run_pipeline.py
```

3. Start the API server:
```bash
cd api
python main.py
```

Or using uvicorn directly:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### POST `/api/matches`

Find athlete matches based on parameters.

**Request Body:**
```json
{
  "age": 45,
  "location": "Stockholm",
  "gender": "Male",
  "total_distance_km": 2000,
  "avg_daily_recovery_score": 75,
  "total_activities": 100,
  "max_results": 20,
  "model_type": "logistic"
}
```

**Response:**
```json
{
  "matches": [
    {
      "athlete_id": "ATH_001",
      "name": "Athlete Name",
      "age": 45,
      "location": "Stockholm",
      "gender": "Male",
      "match_score": 0.85,
      "match_probability": 0.82,
      "similarity_features": {...},
      "profile_summary": {...}
    }
  ],
  "total_found": 20,
  "model_used": "logistic",
  "search_params": {...}
}
```

### GET `/api/parameters`

Get available parameter options and ranges.

### GET `/api/models`

Get list of available models.

### GET `/api/athlete/{athlete_id}`

Get detailed profile for a specific athlete.

## Model Types

- `logistic`: Logistic Regression (fast, interpretable)
- `random_forest`: Random Forest (robust, feature importance)
- `neural_network`: Neural Network (complex patterns)
- `weighted_knn`: Weighted KNN (similarity-based)
- `cosine`: Cosine Similarity (simple similarity)

## Project Structure

```
api/
├── main.py              # FastAPI application
├── match_service.py     # Match finding logic
└── __init__.py
```

