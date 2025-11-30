# Athlete Peer Matching Frontend

A professional, production-ready frontend interface for finding athlete peer matches based on customizable parameters.

## Features

- **Comprehensive Parameter Controls**: Adjust age, location, gender, activity metrics, and more
- **Multiple Model Support**: Choose from Logistic Regression, Random Forest, Neural Network, Weighted KNN, or Cosine Similarity
- **Real-time Match Results**: View matches with detailed similarity breakdowns
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Professional UI**: Clean, modern interface built with Material-UI

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000` (or configure via environment variable)

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (optional, defaults to `http://localhost:8000`):
```bash
VITE_API_URL=http://localhost:8000
```

4. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Usage

1. **Select Model Type**: Choose which machine learning model to use for matching
2. **Set Age Criteria**: Use exact age or age range
3. **Configure Location & Gender**: Filter by location and gender preferences
4. **Adjust Activity Metrics**: Set total distance, activities, recovery score, and social engagement
5. **Set Additional Preferences**: Choose primary sport, competition level, communication style, and injury count
6. **Find Matches**: Click "Find Matches" to see results
7. **View Details**: Expand any match card to see detailed similarity breakdowns

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts          # API client for backend communication
│   ├── components/
│   │   ├── MatchFinder.tsx    # Main search interface
│   │   └── MatchResults.tsx   # Results display component
│   ├── App.tsx                 # Main app component with theme
│   └── main.tsx               # Entry point
├── package.json
└── vite.config.ts
```

## Technologies

- **React 19**: UI framework
- **TypeScript**: Type safety
- **Material-UI (MUI)**: Component library and theming
- **Vite**: Build tool and dev server

## API Integration

The frontend communicates with the FastAPI backend. Ensure the backend is running before using the frontend.

See `api/main.py` for available endpoints.

