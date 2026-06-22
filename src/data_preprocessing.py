import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config, ensure_dirs


class DataPreprocessing:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("DataPreprocessing", config)
        self.scaler = StandardScaler()
        self.city_encoder = LabelEncoder()
        ensure_dirs("artifacts")

    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"]      = df["timestamp"].dt.hour
        df["day"]       = df["timestamp"].dt.day
        df["month"]     = df["timestamp"].dt.month
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        self.logger.info("Extracted time features: hour, day, month, dayofweek, is_weekend.")
        return df

    def encode_city(self, df: pd.DataFrame) -> pd.DataFrame:
        df["city_encoded"] = self.city_encoder.fit_transform(df["city"])
        self.logger.info(f"Label-encoded {df['city'].nunique()} cities.")
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features per city for time-series awareness."""
        df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
        for lag in [1, 3, 6]:
            df[f"aqi_lag_{lag}"] = df.groupby("city")["aqi"].shift(lag)
            df[f"pm2_5_lag_{lag}"] = df.groupby("city")["pm2_5"].shift(lag)
        df[f"aqi_rolling_mean_6"] = (
            df.groupby("city")["aqi"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )
        self.logger.info("Added lag features: aqi_lag_1/3/6, pm2_5_lag_1/3/6, aqi_rolling_mean_6.")
        return df

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = ["timestamp", "city", "lat", "lon"]
        existing = [c for c in drop if c in df.columns]
        df = df.drop(columns=existing)
        self.logger.info(f"Dropped non-feature columns: {existing}")
        return df

    def handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        self.logger.info(f"Dropped {dropped} rows with nulls (mainly from lag features). Remaining: {len(df)}")
        return df

    def scale_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c != target]
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        self.logger.info(f"Scaled {len(feature_cols)} feature columns with StandardScaler.")
        return df

    def save_artifacts(self, feature_cols: list):
        with open(self.config["model"]["scaler_path"], "wb") as f:
            pickle.dump(self.scaler, f)
        with open("artifacts/city_encoder.pkl", "wb") as f:
            pickle.dump(self.city_encoder, f)
        with open("artifacts/feature_cols.pkl", "wb") as f:
            pickle.dump(feature_cols, f)
        self.logger.info("Saved scaler, city encoder, and feature columns to artifacts/.")

    def split(self, df: pd.DataFrame):
        target = self.config["target_column"]
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["model"]["test_size"],
            random_state=self.config["model"]["random_state"]
        )
        self.logger.info(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
        return X_train, X_test, y_train, y_test

    def run(self, df: pd.DataFrame):
        self.logger.info("=== Data Preprocessing Started ===")
        df = self.parse_timestamps(df)
        df = self.encode_city(df)
        df = self.add_lag_features(df)
        df = self.drop_columns(df)
        df = self.handle_nulls(df)

        target = self.config["target_column"]
        feature_cols = [c for c in df.columns if c != target]

        df = self.scale_features(df, target)

        processed_path = self.config["data"]["processed_path"]
        df.to_csv(processed_path, index=False)
        self.logger.info(f"Processed data saved to {processed_path}")

        self.save_artifacts(feature_cols)
        X_train, X_test, y_train, y_test = self.split(df)
        self.logger.info("=== Preprocessing Complete ===")
        return X_train, X_test, y_train, y_test
