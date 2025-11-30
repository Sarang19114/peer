# Athlete Peer Matching System

A comprehensive machine learning system for finding compatible athlete training partners, with a professional frontend interface for customizing match parameters.

## Features

- **Multiple ML Models**: Choose from Logistic Regression, Random Forest, Neural Network, Weighted KNN, or Cosine Similarity
- **Comprehensive Parameter Control**: Adjust age, location, gender, activity metrics, recovery scores, and more
- **Real-time Match Generation**: Get instant results with detailed similarity breakdowns
- **Professional UI**: Clean, responsive interface built with React and Material-UI
- **RESTful API**: FastAPI backend with full documentation

## Project Structure

```
peer/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   └── match_service.py   # Match finding logic
├── frontend/              # React frontend
│   ├── src/
│   │   ├── api/          # API client
│   │   └── components/   # UI components
├── models/                # Trained ML models
├── data/                  # Data files
│   ├── raw/              # Source data
│   └── processed/       # Processed features
├── src/                   # Core Python modules
│   ├── feature_engineering_improved.py
│   └── complete_source_code.py
└── results/               # Model evaluation results
```

## Quick Start

### 1. Install Dependencies

**Backend:**
```bash
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### 2. Train Models (if not already done)

```bash
python run_pipeline.py
```

This will:
- Load and process athlete data
- Generate feature sets
- Train multiple ML models
- Save models to `models/` directory

### 3. Start the Backend API

```bash
cd api
python main.py
```

Or using uvicorn:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 4. Start the Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

### Using the Web Interface

1. Open `http://localhost:5173` in your browser
2. Select a model type (default: Logistic Regression)
3. Configure search parameters:
   - **Age**: Exact age or age range
   - **Location & Gender**: Filter by location and gender
   - **Activity Metrics**: Set total distance, activities, recovery score
   - **Preferences**: Choose sport, competition level, communication style
4. Click "Find Matches" to see results
5. Expand any match card to view detailed similarity breakdowns

### Using the API Directly

**Find Matches:**
```bash
curl -X POST "http://localhost:8000/api/matches" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "location": "Stockholm",
    "total_distance_km": 2000,
    "max_results": 10,
    "model_type": "logistic"
  }'
```

**Get Available Parameters:**
```bash
curl "http://localhost:8000/api/parameters"
```

## Model Types

- **logistic**: Logistic Regression - Fast, interpretable baseline model
- **random_forest**: Random Forest - Robust ensemble with feature importance
- **neural_network**: Neural Network - Deep learning for complex patterns
- **weighted_knn**: Weighted KNN - Similarity-based matching with weighted features
- **cosine**: Cosine Similarity - Simple similarity-based matching

## Available Parameters

- **Age**: Exact age or min/max range
- **Location**: City/location filter
- **Gender**: Gender preference
- **Total Distance (km)**: Total training distance
- **Recovery Score**: Average daily recovery score (0-100)
- **Total Activities**: Number of activities
- **Injury Count**: Number of injuries
- **Social Engagement Score**: Social engagement level (0-10)
- **Communication Preference**: Communication style
- **Primary Sport**: Sport preference
- **Competition Level**: Competition level (Beginner, Intermediate, Advanced, Elite Amateur)

## Development

### Backend Development

The API is built with FastAPI. Key files:
- `api/main.py`: API endpoints and request/response models
- `api/match_service.py`: Core matching logic and model integration

### Frontend Development

The frontend is built with React + TypeScript + Material-UI. Key files:
- `frontend/src/components/MatchFinder.tsx`: Main search interface
- `frontend/src/components/MatchResults.tsx`: Results display
- `frontend/src/api/client.ts`: API client

### Running Tests

```bash
# Backend tests
pytest api/tests/

# Frontend tests (if configured)
cd frontend
npm test
```

## Documentation

- [Frontend Documentation](README_FRONTEND.md)
- [API Documentation](README_API.md)

## Requirements

- Python 3.8+
- Node.js 18+
- npm or yarn

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

