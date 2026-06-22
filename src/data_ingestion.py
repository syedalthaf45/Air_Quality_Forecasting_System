import requests
import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config, ensure_dirs

# Realistic baseline pollutant values per city (based on published AQI data)
CITY_BASELINES = {
    "Delhi":     {"pm2_5": 89, "pm10": 145, "co": 800,  "no": 12, "no2": 48, "o3": 35, "so2": 22, "nh3": 18, "aqi_base": 4},
    "Mumbai":    {"pm2_5": 45, "pm10": 78,  "co": 450,  "no": 7,  "no2": 32, "o3": 52, "so2": 14, "nh3": 9,  "aqi_base": 3},
    "Bengaluru": {"pm2_5": 32, "pm10": 55,  "co": 320,  "no": 5,  "no2": 28, "o3": 58, "so2": 10, "nh3": 6,  "aqi_base": 2},
    "Hyderabad": {"pm2_5": 38, "pm10": 64,  "co": 360,  "no": 6,  "no2": 30, "o3": 55, "so2": 12, "nh3": 7,  "aqi_base": 3},
    "Chennai":   {"pm2_5": 35, "pm10": 60,  "co": 340,  "no": 5,  "no2": 26, "o3": 60, "so2": 11, "nh3": 6,  "aqi_base": 2},
    "Kolkata":   {"pm2_5": 62, "pm10": 105, "co": 600,  "no": 10, "no2": 42, "o3": 40, "so2": 18, "nh3": 13, "aqi_base": 3},
    "Warangal":  {"pm2_5": 28, "pm10": 48,  "co": 280,  "no": 4,  "no2": 22, "o3": 62, "so2": 8,  "nh3": 5,  "aqi_base": 2},
    "Pune":      {"pm2_5": 40, "pm10": 70,  "co": 390,  "no": 6,  "no2": 29, "o3": 54, "so2": 12, "nh3": 7,  "aqi_base": 3},
    "Ahmedabad": {"pm2_5": 55, "pm10": 92,  "co": 520,  "no": 8,  "no2": 38, "o3": 44, "so2": 16, "nh3": 11, "aqi_base": 3},
    "Jaipur":    {"pm2_5": 50, "pm10": 85,  "co": 480,  "no": 8,  "no2": 35, "o3": 46, "so2": 15, "nh3": 10, "aqi_base": 3},
}


class DataIngestion:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("DataIngestion", config)
        self.api_key = config["api"]["key"]
        self.base_url = config["api"]["base_url"]
        self.cities = config["cities"]
        self.history_days = config["data"]["history_days"]
        ensure_dirs("data", "artifacts", "logs")

    def fetch_current(self, city: dict) -> dict | None:
        params = {"lat": city["lat"], "lon": city["lon"], "appid": self.api_key}
        try:
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("list", [])
            if items:
                item = items[0]
                return {
                    "city": city["name"], "lat": city["lat"], "lon": city["lon"],
                    "timestamp": datetime.utcfromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "aqi": item["main"]["aqi"],
                    **item["components"]
                }
        except Exception as e:
            self.logger.warning(f"API unavailable for {city['name']}: {e}")
        return None

    def get_baseline(self, city_name: str, lat: float, lon: float) -> dict:
        """Return realistic baseline for a city (fallback if API unavailable)."""
        b = CITY_BASELINES.get(city_name, {"pm2_5": 40, "pm10": 70, "co": 400, "no": 6,
                                            "no2": 30, "o3": 55, "so2": 12, "nh3": 8, "aqi_base": 3})
        return {"city": city_name, "lat": lat, "lon": lon,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "aqi": b["aqi_base"], "pm2_5": b["pm2_5"], "pm10": b["pm10"],
                "co": b["co"], "no": b["no"], "no2": b["no2"],
                "o3": b["o3"], "so2": b["so2"], "nh3": b["nh3"]}

    def generate_history(self, base: dict, days: int = 7) -> list:
        """
        Simulate realistic hourly air quality history from a base reading.
        Models daily cycles: pollution peaks during rush hours (8am, 6pm),
        dips overnight. Each city gets deterministic noise via city-name seed.
        """
        import random
        random.seed(abs(hash(base["city"])) % 10000)
        records = []
        now = datetime.utcnow()
        for hour_offset in range(days * 24, 0, -1):
            ts = now - timedelta(hours=hour_offset)
            hour = ts.hour
            # Rush hour peaks at 8am and 6pm
            rush = 1.0 + 0.35 * (np.exp(-((hour - 8) ** 2) / 8) + np.exp(-((hour - 18) ** 2) / 8))
            noise = lambda: random.uniform(0.88, 1.12)
            pm2_5 = max(2.0, base["pm2_5"] * rush * noise())
            pm10  = max(2.0, base["pm10"]  * rush * noise())
            co    = max(50.0, base["co"]   * noise())
            no    = max(0.0,  base["no"]   * noise())
            no2   = max(0.0,  base["no2"]  * rush * noise())
            o3    = max(0.0,  base["o3"]   * noise())
            so2   = max(0.0,  base["so2"]  * noise())
            nh3   = max(0.0,  base["nh3"]  * noise())
            # WHO-based AQI from PM2.5
            if pm2_5 <= 10:   aqi = 1
            elif pm2_5 <= 25: aqi = 2
            elif pm2_5 <= 50: aqi = 3
            elif pm2_5 <= 75: aqi = 4
            else:              aqi = 5
            records.append({
                "city": base["city"], "lat": base["lat"], "lon": base["lon"],
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "aqi": aqi,
                "co": round(co, 2), "no": round(no, 2), "no2": round(no2, 2),
                "o3": round(o3, 2), "so2": round(so2, 2),
                "pm2_5": round(pm2_5, 2), "pm10": round(pm10, 2), "nh3": round(nh3, 2)
            })
        return records

    def fetch_all_cities(self) -> pd.DataFrame:
        all_records = []
        api_available = False
        for city in self.cities:
            self.logger.info(f"Fetching data for {city['name']}...")
            current = self.fetch_current(city)
            if current:
                api_available = True
                base = current
                self.logger.info(f"  {city['name']}: Live API data received (AQI={current['aqi']}).")
            else:
                base = self.get_baseline(city["name"], city["lat"], city["lon"])
                self.logger.info(f"  {city['name']}: Using realistic baseline (API activating).")
            history = self.generate_history(base, days=self.history_days)
            history.append(base)
            all_records.extend(history)
            time.sleep(0.2)

        if not api_available:
            self.logger.warning("OpenWeatherMap API key not yet active (can take up to 2hrs after signup).")
            self.logger.warning("Running pipeline with realistic city baselines — swap to live data once key activates.")

        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=["city", "timestamp"])
        df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
        return df

    def fetch_current_all(self) -> pd.DataFrame:
        """For Flask live dashboard — tries API first, falls back to baselines."""
        all_records = []
        for city in self.cities:
            rec = self.fetch_current(city)
            if rec is None:
                rec = self.get_baseline(city["name"], city["lat"], city["lon"])
            all_records.append(rec)
            time.sleep(0.15)
        return pd.DataFrame(all_records)

    def run(self) -> pd.DataFrame:
        self.logger.info("=== Data Ingestion Started ===")
        df = self.fetch_all_cities()
        raw_path = self.config["data"]["raw_path"]
        df.to_csv(raw_path, index=False)
        self.logger.info(f"Raw data saved to {raw_path} | Shape: {df.shape}")
        self.logger.info(f"Cities: {df['city'].nunique()} | Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        self.logger.info("=== Data Ingestion Complete ===")
        return df


if __name__ == "__main__":
    config = load_config()
    df = DataIngestion(config).run()
    print(df.head())
    print(df["city"].value_counts())
