"""
🌡️ Weather Forecast API
Serves 3 ML models: Temperature, Rain, Extreme Weather
"""

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import math

app = FastAPI(title="Weather Forecast API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===============================================================
# LOAD MODELS
# ===============================================================
print("Loading models...")

temp_pkg      = joblib.load("models/weather_model_92M.pkl")
temp_model    = temp_pkg['model']
temp_features = temp_pkg['features']
temp_medians  = temp_pkg['feature_medians']

rain_pkg      = joblib.load("models/rain_prediction_model.pkl")
rain_model    = rain_pkg['model']
rain_features = rain_pkg['features']
rain_medians  = rain_pkg['feature_medians']

extreme_pkg      = joblib.load("models/extreme_weather_model.pkl")
extreme_model    = extreme_pkg['model']
extreme_features = extreme_pkg['features']
extreme_medians  = extreme_pkg['feature_medians']
extreme_classes  = extreme_pkg['class_names']

print("✓ All models loaded!")

# ===============================================================
# RANDOMIZED EXAMPLE INPUTS
# ===============================================================
RANDOM_SCENARIOS = [
    {"name": "☀️ NYC Summer Heat",       "latitude": 40.71, "longitude": -74.01, "elevation": 10,   "temp": 88,  "dewp": 72, "slp": 1012, "wdsp": 7,  "max_temp": 94,  "min_temp": 78,  "prcp": 0.0,  "month": 7,  "day_of_year": 195},
    {"name": "❄️ Chicago Blizzard",      "latitude": 41.85, "longitude": -87.65, "elevation": 182,  "temp": 14,  "dewp": 8,  "slp": 1030, "wdsp": 22, "max_temp": 20,  "min_temp": 8,   "prcp": 0.4,  "month": 1,  "day_of_year": 20},
    {"name": "🌧️ Seattle Rain",          "latitude": 47.61, "longitude": -122.33,"elevation": 56,   "temp": 52,  "dewp": 49, "slp": 1002, "wdsp": 9,  "max_temp": 57,  "min_temp": 47,  "prcp": 0.8,  "month": 11, "day_of_year": 315},
    {"name": "🔥 Phoenix Desert Heat",   "latitude": 33.45, "longitude": -112.07,"elevation": 331,  "temp": 108, "dewp": 30, "slp": 1006, "wdsp": 5,  "max_temp": 115, "min_temp": 90,  "prcp": 0.0,  "month": 7,  "day_of_year": 200},
    {"name": "🌤️ LA Spring",            "latitude": 34.05, "longitude": -118.24,"elevation": 71,   "temp": 72,  "dewp": 52, "slp": 1016, "wdsp": 6,  "max_temp": 78,  "min_temp": 62,  "prcp": 0.0,  "month": 4,  "day_of_year": 105},
    {"name": "⛈️ Miami Thunderstorm",    "latitude": 25.77, "longitude": -80.19, "elevation": 2,    "temp": 82,  "dewp": 76, "slp": 1008, "wdsp": 14, "max_temp": 88,  "min_temp": 78,  "prcp": 1.2,  "month": 8,  "day_of_year": 225},
    {"name": "🌨️ Denver Snow",           "latitude": 39.74, "longitude": -104.98,"elevation": 1609, "temp": 28,  "dewp": 22, "slp": 1020, "wdsp": 15, "max_temp": 35,  "min_temp": 22,  "prcp": 0.3,  "month": 3,  "day_of_year": 75},
    {"name": "🍂 Boston Fall",           "latitude": 42.36, "longitude": -71.06, "elevation": 9,    "temp": 52,  "dewp": 42, "slp": 1018, "wdsp": 10, "max_temp": 58,  "min_temp": 45,  "prcp": 0.1,  "month": 10, "day_of_year": 285},
    {"name": "🌬️ Dallas Windstorm",      "latitude": 32.78, "longitude": -96.80, "elevation": 139,  "temp": 65,  "dewp": 48, "slp": 1005, "wdsp": 28, "max_temp": 72,  "min_temp": 58,  "prcp": 0.05, "month": 4,  "day_of_year": 110},
    {"name": "🌫️ San Francisco Fog",     "latitude": 37.77, "longitude": -122.42,"elevation": 52,   "temp": 58,  "dewp": 55, "slp": 1014, "wdsp": 12, "max_temp": 62,  "min_temp": 52,  "prcp": 0.0,  "month": 6,  "day_of_year": 170},
    {"name": "🌪️ Oklahoma Tornado Risk", "latitude": 35.47, "longitude": -97.52, "elevation": 360,  "temp": 74,  "dewp": 66, "slp": 998,  "wdsp": 20, "max_temp": 82,  "min_temp": 68,  "prcp": 0.6,  "month": 5,  "day_of_year": 140},
    {"name": "🌊 Houston Humidity",      "latitude": 29.76, "longitude": -95.37, "elevation": 15,   "temp": 88,  "dewp": 80, "slp": 1010, "wdsp": 8,  "max_temp": 94,  "min_temp": 82,  "prcp": 0.2,  "month": 8,  "day_of_year": 220},
]

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================
def prepare_features(data: dict, features: list, medians: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    if 'month' in data:
        df['month_sin'] = math.sin(2 * math.pi * data['month'] / 12)
        df['month_cos'] = math.cos(2 * math.pi * data['month'] / 12)
    if 'day_of_year' in data:
        df['day_sin'] = math.sin(2 * math.pi * data['day_of_year'] / 365)
        df['day_cos'] = math.cos(2 * math.pi * data['day_of_year'] / 365)
    if 'TEMP' in data and 'DEWP' in data:
        df['humidity_proxy'] = data['TEMP'] - data['DEWP']
    if 'MAX' in data and 'MIN' in data:
        df['temp_range'] = data['MAX'] - data['MIN']
    if 'WDSP' in data:
        df['wind_energy'] = data['WDSP'] ** 2
    for feat in features:
        if feat not in df.columns:
            df[feat] = medians.get(feat, 0)
    df = df[features].fillna(pd.Series(medians))
    return df

def build_input_data(lat, lon, elev, temp, dewp, slp, wdsp, max_t, min_t, prcp, month, doy):
    return {
        'LATITUDE': lat, 'LONGITUDE': lon, 'ELEVATION': elev,
        'TEMP': temp, 'DEWP': dewp, 'SLP': slp, 'STP': slp - 10,
        'VISIB': 10, 'WDSP': wdsp, 'MXSPD': wdsp * 1.5, 'GUST': wdsp * 2,
        'MAX': max_t, 'MIN': min_t, 'PRCP': prcp, 'SNDP': 0,
        'year': datetime.now().year, 'month': month, 'day_of_year': doy,
        'TEMP_lag_1': temp, 'TEMP_lag_3': temp - 2, 'TEMP_lag_7': temp - 5, 'TEMP_lag_14': temp - 8,
        'DEWP_lag_1': dewp, 'SLP_lag_1': slp, 'WDSP_lag_1': wdsp, 'PRCP_lag_1': prcp,
        'TEMP_roll_7': temp, 'TEMP_roll_30': temp - 3,
        'PRCP_roll_7': prcp * 7, 'PRCP_roll_30': prcp * 30,
        'WDSP_roll_7_std': wdsp * 0.3, 'SLP_roll_7_mean': slp,
        'fog': 0, 'rain': 1 if prcp > 0 else 0,
        'snow': 0, 'hail': 0, 'thunder': 0, 'tornado': 0
    }

def get_weather_icon(extreme_class, rain, temp):
    if extreme_class == 3: return "⛈️"
    if extreme_class == 4: return "❄️"
    if extreme_class == 1: return "🔥"
    if extreme_class == 2: return "🥶"
    if rain == 1: return "🌧️"
    if temp > 80: return "☀️"
    if temp > 60: return "🌤️"
    return "☁️"

def get_alert_color(extreme_class):
    return {0: "#4CAF50", 1: "#FF5722", 2: "#2196F3", 3: "#9C27B0", 4: "#00BCD4"}.get(extreme_class, "#4CAF50")

def get_weather_hint(extreme_class: int, rain_pred: int, rain_prob: float, predicted_temp: float, input_data: dict) -> dict:
    hints = []
    temp = predicted_temp
    prcp = input_data.get('PRCP', 0)
    wdsp = input_data.get('WDSP', 0)

    # Snow / freezing conditions
    if rain_pred == 1 and temp <= 32:
        hints.append(("❄️ Snow likely tomorrow", "snow"))
    elif rain_pred == 1 and 32 < temp <= 35:
        hints.append(("🌨️ Sleet or freezing rain possible", "sleet"))
    elif rain_pred == 0 and temp <= 28:
        hints.append(("🧊 Frost overnight — roads may be icy", "frost"))

    # Rain
    if rain_pred == 1 and rain_prob >= 70:
        hints.append(("☂️ Bring an umbrella", "rain"))
    elif rain_pred == 1 and rain_prob >= 40:
        hints.append(("🌂 Light rain possible", "rain"))

    # Wind
    if wdsp >= 25:
        hints.append(("💨 Very windy — secure loose objects", "wind"))
    elif wdsp >= 15:
        hints.append(("🌬️ Breezy conditions expected", "wind"))

    # Extreme class
    if extreme_class == 1:
        hints.append(("🥤 Stay hydrated and avoid midday sun", "heat"))
    elif extreme_class == 2:
        hints.append(("🧥 Bundle up — dangerously cold", "cold"))
    elif extreme_class == 3:
        hints.append(("⛈️ Seek shelter — storm conditions", "storm"))
    elif extreme_class == 4:
        hints.append(("🚗 Expect travel disruptions from snow", "snow"))

    # Default
    if not hints and temp > 75:
        hints.append(("😎 Mild and pleasant day ahead", "normal"))
    elif not hints and temp >= 50:
        hints.append(("🌤️ Comfortable conditions tomorrow", "normal"))
    elif not hints:
        hints.append(("🧤 Layer up — it'll be cold", "cold"))

    return {"text": hints[0][0], "type": hints[0][1], "all": [h[0] for h in hints]}

def run_predictions(input_data):
    temp_df = prepare_features(input_data, temp_features, temp_medians)
    predicted_temp = float(temp_model.predict(temp_df)[0])

    rain_df = prepare_features(input_data, rain_features, rain_medians)
    rain_pred = int(rain_model.predict(rain_df)[0])
    rain_prob = float(rain_model.predict_proba(rain_df)[0][1]) * 100

    extreme_df = prepare_features(input_data, extreme_features, extreme_medians)
    extreme_pred = int(extreme_model.predict(extreme_df)[0])
    extreme_label = extreme_classes[extreme_pred]

    hint = get_weather_hint(extreme_pred, rain_pred, rain_prob, predicted_temp, input_data)

    return {
        "predicted_temp": round(predicted_temp, 1),
        "rain_prediction": "Yes" if rain_pred == 1 else "No",
        "rain_probability": round(rain_prob, 1),
        "extreme_alert": extreme_label,
        "extreme_code": extreme_pred,
        "weather_icon": get_weather_icon(extreme_pred, rain_pred, predicted_temp),
        "alert_color": get_alert_color(extreme_pred),
        "hint": hint,
    }

# ===============================================================
# ROUTES
# ===============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "scenarios": RANDOM_SCENARIOS
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    latitude: float = Form(...), longitude: float = Form(...), elevation: float = Form(0),
    temp: float = Form(...), dewp: float = Form(...), slp: float = Form(1013),
    wdsp: float = Form(5), max_temp: float = Form(...), min_temp: float = Form(...),
    prcp: float = Form(0), month: int = Form(...), day_of_year: int = Form(...)
):
    input_data = build_input_data(latitude, longitude, elevation, temp, dewp, slp, wdsp, max_temp, min_temp, prcp, month, day_of_year)
    results = run_predictions(input_data)
    results["input_temp"] = temp
    results["input_location"] = f"{latitude:.2f}°, {longitude:.2f}°"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "show_results": True,
        "scenarios": RANDOM_SCENARIOS
    })

@app.get("/api/scenarios")
async def get_scenarios():
    return JSONResponse(content=RANDOM_SCENARIOS)

@app.get("/api/predict")
async def api_predict(
    latitude: float, longitude: float, temp: float, dewp: float,
    max_temp: float, min_temp: float, month: int, day_of_year: int,
    wdsp: float = 5, prcp: float = 0, slp: float = 1013, elevation: float = 0
):
    input_data = build_input_data(latitude, longitude, elevation, temp, dewp, slp, wdsp, max_temp, min_temp, prcp, month, day_of_year)
    results = run_predictions(input_data)
    return {
        "tomorrow_temperature_f": results["predicted_temp"],
        "rain": {"prediction": results["rain_prediction"], "probability_percent": results["rain_probability"]},
        "extreme_weather": {"code": results["extreme_code"], "alert": results["extreme_alert"]},
        "hint": results["hint"]["text"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": 3}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)