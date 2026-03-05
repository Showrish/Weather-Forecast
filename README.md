# 🌤️ WeatherCast ML

> Predictive weather forecasting web application powered by machine learning — trained on **92 million** real-world weather records from NOAA's Global Surface Summary of Day (GSOD) dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-orange?style=flat-square)
![DuckDB](https://img.shields.io/badge/DuckDB-Analytics-yellow?style=flat-square)

---

## 🔍 Overview

WeatherCast ML takes today's weather conditions as input and predicts **tomorrow's weather** using three independently trained LightGBM models. It features a live geolocation button that auto-fills all inputs using the Open-Meteo API, and a rich dark-themed UI.

---

## 🤖 Models

| Model | Task | Performance |
|---|---|---|
| 🌡️ Temperature | Predict tomorrow's avg temperature (°F) | R² = 0.9669 · MAE = 2.88°F |
| 🌧️ Rain Prediction | Binary classification — will it rain? | Outputs Yes/No + probability % |
| ⚠️ Extreme Weather | 5-class alert detection | Normal / Heat / Cold / Storm / Snow |

All models trained with **LightGBM** using 45 engineered features including lag temperatures, rolling averages, cyclical seasonal encodings, and atmospheric pressure trends.

---

## 🚀 Features

- **Live weather input** — "Your Location" button fetches real-time data via Open-Meteo API using browser geolocation
- **Yesterday's data** — fetches completed full-day averages from Open-Meteo's historical archive for accurate predictions
- **Smart randomizer** — 12 pre-loaded real-world scenarios (NYC summer, Chicago blizzard, Miami storm, etc.)
- **Contextual alerts** — dynamic hints like *❄️ Snow likely tomorrow* or *☂️ Bring an umbrella*
- **Model info modals** — click any model card to see architecture, features used, and how it works
- **JSON API** — `/api/predict` endpoint for programmatic access

---

## 🗂️ Project Structure

```
Weather Forecast/
├── main.py                  # FastAPI backend — loads models, routes, prediction logic
├── requirements.txt         # Dependencies
├── templates/
│   └── index.html           # Frontend — dark UI, modals, geolocation, randomizer
├── static/
│   └── style.css            # Styling — glassmorphism, animations, responsive
└── models/                  # LightGBM .pkl model files (not tracked in git)
    ├── weather_model_92M.pkl
    ├── rain_prediction_model.pkl
    └── extreme_weather_model.pkl
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Showrish/Weather-Forecast.git
cd Weather-Forecast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your trained models to the /models folder

# 4. Run
uvicorn main:app --reload
```

Then open `http://localhost:8000` in your browser.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| ML Models | LightGBM |
| Data Processing | DuckDB · Pandas · NumPy |
| Training Data | NOAA GSOD — 92M records |
| Live Weather | Open-Meteo API (free, no key) |
| Frontend | Vanilla HTML/CSS/JS |

---

## 📡 API

```
GET /api/predict?latitude=40.71&longitude=-74.01&temp=72&dewp=55
                &max_temp=78&min_temp=65&month=6&day_of_year=172
```

**Response:**
```json
{
  "tomorrow_temperature_f": 74.3,
  "rain": {
    "prediction": "No",
    "probability_percent": 18.4
  },
  "extreme_weather": {
    "code": 0,
    "alert": "Normal"
  },
  "hint": "🌤️ Comfortable conditions tomorrow"
}
```

---

---

## 📊 Data & Training

- **Dataset:** NOAA Global Surface Summary of Day (GSOD)
- **Size:** 92,209,203 records · 27 features
- **Coverage:** Global weather stations · Year range 2000–2024
- **Feature engineering:** 7/14/30-day lag temps, rolling averages, cyclical month/day encodings, humidity proxy, wind energy
- **Training pipeline:** DuckDB for big data processing → LightGBM with early stopping → Optuna hyperparameter tuning

---

## 📈 Data Insights

### Missing Data per Feature
![Missing Data](Data%20Insights/1_missing.png)

### Temperature Distribution
![Temperature Distribution](Data%20Insights/2_temp_dist.png)

### Monthly Climate Trends
![Monthly Trends](Data%20Insights/3_monthly.png)

### Year-over-Year Temperature Trend
![Yearly Trend](Data%20Insights/4_yearly.png)

### Feature Correlation Matrix
![Correlation Matrix](Data%20Insights/5_corr.png)

### Weather Event Frequency
![Weather Events](Data%20Insights/6_events.png)

### Hottest & Coldest Stations
![Extreme Stations](Data%20Insights/7_extreme_stations.png)

### Wind Speed & Pressure vs Temperature
![Wind & Pressure](Data%20Insights/8_wind_pressure.png)

---

## 👤 Author

**Showrish** — Big Data Project 2026