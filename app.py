"""
app.py — Flask web application for Air Quality Forecasting.
Pages: / (dashboard), /city/<name>, /predict, /metrics, /api/current
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_artifact(path):
    with open(os.path.join(BASE_DIR, path), "rb") as f:
        return pickle.load(f)


# Load artifacts
model        = load_artifact("artifacts/model.pkl")
scaler       = load_artifact("artifacts/scaler.pkl")
city_encoder = load_artifact("artifacts/city_encoder.pkl")
feature_cols = load_artifact("artifacts/feature_cols.pkl")

with open(os.path.join(BASE_DIR, "artifacts/evaluation_report.json")) as f:
    eval_report = json.load(f)

with open(os.path.join(BASE_DIR, "config.yaml")) as f:
    import yaml
    config = yaml.safe_load(f)

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
AQI_COLORS = {1: "#16a34a", 2: "#65a30d", 3: "#ca8a04", 4: "#ea580c", 5: "#dc2626"}
CITIES = [c["name"] for c in config["cities"]]


def get_current_data():
    """Fetch live data from API for dashboard."""
    from src.data_ingestion import DataIngestion
    ingestion = DataIngestion(config)
    return ingestion.fetch_current_all()


def predict_aqi(city_name: str, hour: int = None, month: int = None,
                co: float = 200, no: float = 5, no2: float = 20,
                o3: float = 60, so2: float = 10, pm2_5: float = 35,
                pm10: float = 50, nh3: float = 5,
                aqi_lag_1: float = 2, aqi_lag_3: float = 2,
                aqi_lag_6: float = 2, pm2_5_lag_1: float = 35,
                pm2_5_lag_3: float = 35, pm2_5_lag_6: float = 35,
                aqi_rolling_mean_6: float = 2) -> dict:
    now = datetime.utcnow()
    if hour is None:
        hour = now.hour
    if month is None:
        month = now.month

    try:
        city_enc = int(city_encoder.transform([city_name])[0])
    except Exception:
        city_enc = 0

    input_dict = {
        "co": co, "no": no, "no2": no2, "o3": o3, "so2": so2,
        "pm2_5": pm2_5, "pm10": pm10, "nh3": nh3,
        "hour": hour, "day": now.day, "month": month,
        "dayofweek": now.weekday(), "is_weekend": int(now.weekday() >= 5),
        "city_encoded": city_enc,
        "aqi_lag_1": aqi_lag_1, "aqi_lag_3": aqi_lag_3, "aqi_lag_6": aqi_lag_6,
        "pm2_5_lag_1": pm2_5_lag_1, "pm2_5_lag_3": pm2_5_lag_3, "pm2_5_lag_6": pm2_5_lag_6,
        "aqi_rolling_mean_6": aqi_rolling_mean_6
    }

    # Align to training feature order
    row = pd.DataFrame([[input_dict.get(c, 0) for c in feature_cols]], columns=feature_cols)
    row_scaled = scaler.transform(row)
    pred = float(model.predict(row_scaled)[0])
    pred_aqi = int(np.clip(round(pred), 1, 5))

    return {
        "city": city_name,
        "predicted_aqi": pred_aqi,
        "aqi_label": AQI_LABELS[pred_aqi],
        "aqi_color": AQI_COLORS[pred_aqi],
        "raw_prediction": round(pred, 3)
    }


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    try:
        df = get_current_data()
        city_data = []
        if not df.empty:
            for city in CITIES:
                row = df[df["city"] == city]
                if not row.empty:
                    r = row.iloc[-1]
                    aqi = int(r["aqi"])
                    city_data.append({
                        "name": city,
                        "aqi": aqi,
                        "label": AQI_LABELS.get(aqi, "Unknown"),
                        "color": AQI_COLORS.get(aqi, "#94a3b8"),
                        "pm2_5": round(float(r["pm2_5"]), 2),
                        "pm10":  round(float(r["pm10"]), 2),
                        "no2":   round(float(r["no2"]), 2),
                        "o3":    round(float(r["o3"]), 2),
                        "timestamp": str(r["timestamp"])
                    })
        # Sort by AQI descending (worst first)
        city_data.sort(key=lambda x: x["aqi"], reverse=True)
        return render_template("dashboard.html", cities=city_data, updated=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    except Exception as e:
        return render_template("dashboard.html", cities=[], error=str(e), updated="N/A")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            city = request.form.get("city", "Delhi")
            pm2_5 = float(request.form.get("pm2_5", 35))
            pm10  = float(request.form.get("pm10", 50))
            no2   = float(request.form.get("no2", 20))
            o3    = float(request.form.get("o3", 60))
            co    = float(request.form.get("co", 200))
            so2   = float(request.form.get("so2", 10))
            result = predict_aqi(
                city_name=city, pm2_5=pm2_5, pm10=pm10,
                no2=no2, o3=o3, co=co, so2=so2,
                aqi_lag_1=pm2_5 / 17.5,
                aqi_lag_3=pm2_5 / 17.5,
                aqi_lag_6=pm2_5 / 17.5,
                pm2_5_lag_1=pm2_5, pm2_5_lag_3=pm2_5, pm2_5_lag_6=pm2_5,
                aqi_rolling_mean_6=pm2_5 / 17.5
            )
        except Exception as e:
            result = {"error": str(e)}
    return render_template("predict.html", cities=CITIES, result=result)


@app.route("/metrics")
def metrics():
    return render_template("metrics.html", report=eval_report)


@app.route("/api/current")
def api_current():
    try:
        df = get_current_data()
        if df.empty:
            return jsonify({"error": "No data"}), 500
        records = []
        for city in CITIES:
            row = df[df["city"] == city]
            if not row.empty:
                r = row.iloc[-1]
                aqi = int(r["aqi"])
                records.append({
                    "city": city,
                    "aqi": aqi,
                    "label": AQI_LABELS.get(aqi, "Unknown"),
                    "pm2_5": round(float(r["pm2_5"]), 2),
                    "pm10":  round(float(r["pm10"]), 2),
                    "no2":   round(float(r["no2"]), 2),
                    "timestamp": str(r["timestamp"])
                })
        return jsonify({"cities": records, "fetched_at": datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        result = predict_aqi(
            city_name=data.get("city", "Delhi"),
            pm2_5=float(data.get("pm2_5", 35)),
            pm10=float(data.get("pm10", 50)),
            no2=float(data.get("no2", 20)),
            o3=float(data.get("o3", 60)),
            co=float(data.get("co", 200)),
            so2=float(data.get("so2", 10)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": eval_report.get("model_name", "unknown")})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
