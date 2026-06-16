# 🏈 Gridiron Intelligence

Score margin & total prediction engine for NFL and NCAAF.
Predicts the final score, margin, and combined total for any matchup using statistical models trained on historical game data.

---

## How It Works

- **NFL**: Trains on 2023 season game data from ESPN's public API. Builds rolling offensive/defensive efficiency ratings per team (last 6–8 games). Predicts margin and total using a Ridge + Gradient Boosting ensemble.
- **NCAAF**: Uses SP+ ratings (the single best college football predictor) from CollegeFootballData.io, combined with game-level historical data. Same model architecture.
- **Confidence intervals**: 80% CI shown for both margin and total. Typical RMSE is ~9 pts (NFL) and ~14 pts (CFB) for margin — the model is honest about uncertainty.

---

## Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The backend will fetch historical data and train models on startup (~30–60 seconds).
Visit `http://localhost:8000/docs` for the interactive API docs.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`

---

## Deploy to Render (one-click)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect your GitHub repo
4. Render reads `render.yaml` and deploys both services automatically
5. After deploy, copy the **API service URL** (e.g. `https://gridiron-api.onrender.com`)
6. In Render dashboard → `gridiron-frontend` → Environment → set:
   ```
   NEXT_PUBLIC_API_URL = https://gridiron-api.onrender.com
   ```
7. Redeploy frontend

> **Note**: On Render's free tier, the backend spins down after 15 min of inactivity. First request after sleep takes ~30 seconds (retraining models). Upgrade to Starter ($7/mo) to keep it warm.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | API health + ready status |
| GET | `/status` | Model training results + stats |
| GET | `/teams/nfl` | List of NFL teams |
| GET | `/teams/cfb` | List of CFB teams |
| POST | `/predict` | Predict a matchup |
| GET | `/schedule/nfl?week=1&season=2024` | NFL schedule for a week |
| GET | `/schedule/cfb?week=1&season=2024` | CFB schedule for a week |

### POST /predict

```json
{
  "league": "NFL",
  "home_team": "Kansas City Chiefs",
  "away_team": "Buffalo Bills",
  "neutral_site": false
}
```

Response:
```json
{
  "home_team": "Kansas City Chiefs",
  "away_team": "Buffalo Bills",
  "league": "NFL",
  "prediction": {
    "predicted_home_score": 27.4,
    "predicted_away_score": 23.1,
    "predicted_margin": 4.3,
    "predicted_total": 50.5,
    "margin_80_lo": -7.2,
    "margin_80_hi": 15.8,
    "total_80_lo": 35.2,
    "total_80_hi": 65.8,
    "home_win_prob": 0.682,
    "model_trained": true
  },
  "key_factors": [...]
}
```

---

## Model Notes

The models are trained on 2023 season data and will update when you retrain (restart the backend with a newer `season` parameter in `data_fetcher.py`).

**What the models use:**
- NFL: Rolling pts scored / pts allowed (6–8 game window), home field advantage (~2.5 pts), offensive/defensive efficiency delta
- CFB: SP+ overall/offense/defense ratings, offense vs defense matchup ratings, home field advantage (~3 pts)

**What they don't (yet) include:**
- Injury reports
- Weather
- Travel distance/fatigue
- Coaching changes
- In-season momentum shifts beyond the rolling window

These are the next logical additions. The architecture is set up to add feature columns cleanly.

---

## Project Structure

```
gridiron/
├── backend/
│   ├── main.py            # FastAPI app
│   ├── data_fetcher.py    # ESPN + CFBD data ingestion
│   ├── feature_engine.py  # Feature engineering
│   ├── models.py          # Ridge + GBM ensemble
│   └── requirements.txt
├── frontend/
│   ├── src/pages/index.tsx    # Main UI
│   ├── src/styles/globals.css
│   └── package.json
├── render.yaml            # Render deploy config
└── README.md
```
