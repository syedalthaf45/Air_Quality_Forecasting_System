import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config


class DataValidation:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("DataValidation", config)
        self.val_cfg = config["validation"]
        self.errors = []
        self.warnings = []

    def check_required_columns(self, df: pd.DataFrame):
        required = self.val_cfg["required_columns"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            self.errors.append(f"Missing columns: {missing}")
            self.logger.error(f"Missing required columns: {missing}")
        else:
            self.logger.info(f"[PASS] All {len(required)} required columns present.")

    def check_null_percent(self, df: pd.DataFrame):
        max_null = self.val_cfg["max_null_percent"]
        null_pct = (df.isnull().sum() / len(df)) * 100
        high = null_pct[null_pct > max_null]
        if not high.empty:
            self.warnings.append(f"High null columns: {high.to_dict()}")
            self.logger.warning(f"Columns exceeding {max_null}% null: {high.to_dict()}")
        else:
            self.logger.info(f"[PASS] All columns within null threshold ({max_null}%).")

    def check_aqi_range(self, df: pd.DataFrame):
        aqi_min = self.val_cfg["aqi_min"]
        aqi_max = self.val_cfg["aqi_max"]
        invalid = df[(df["aqi"] < aqi_min) | (df["aqi"] > aqi_max)]
        if not invalid.empty:
            self.warnings.append(f"{len(invalid)} rows with AQI outside [{aqi_min},{aqi_max}].")
            self.logger.warning(f"{len(invalid)} rows have out-of-range AQI values.")
        else:
            self.logger.info(f"[PASS] All AQI values in valid range [{aqi_min}, {aqi_max}].")

    def check_cities(self, df: pd.DataFrame):
        expected = [c["name"] for c in self.config["cities"]]
        found = df["city"].unique().tolist()
        missing = [c for c in expected if c not in found]
        if missing:
            self.warnings.append(f"No data for cities: {missing}")
            self.logger.warning(f"Missing data for cities: {missing}")
        else:
            self.logger.info(f"[PASS] All {len(expected)} cities present in data.")

    def check_duplicates(self, df: pd.DataFrame):
        dupes = df.duplicated(subset=["city", "timestamp"]).sum()
        if dupes > 0:
            self.warnings.append(f"{dupes} duplicate (city, timestamp) rows found.")
            self.logger.warning(f"{dupes} duplicate rows detected.")
        else:
            self.logger.info("[PASS] No duplicate city+timestamp rows.")

    def check_pollutant_ranges(self, df: pd.DataFrame):
        checks = {
            "pm2_5": (0, 1000),
            "pm10":  (0, 2000),
            "co":    (0, 50000),
            "no2":   (0, 1000),
            "o3":    (0, 1000),
        }
        for col, (lo, hi) in checks.items():
            if col in df.columns:
                bad = df[(df[col] < lo) | (df[col] > hi)]
                if not bad.empty:
                    self.warnings.append(f"{col}: {len(bad)} out-of-range values.")
                    self.logger.warning(f"Pollutant {col}: {len(bad)} suspicious values.")
                else:
                    self.logger.info(f"[PASS] {col} within expected range.")

    def run(self, df: pd.DataFrame) -> bool:
        self.logger.info("=== Data Validation Started ===")
        self.check_required_columns(df)
        self.check_null_percent(df)
        self.check_aqi_range(df)
        self.check_cities(df)
        self.check_duplicates(df)
        self.check_pollutant_ranges(df)

        self.logger.info(f"Validation Summary — Errors: {len(self.errors)}, Warnings: {len(self.warnings)}")
        if self.errors:
            for e in self.errors:
                self.logger.error(f"  ERROR: {e}")
            raise ValueError(f"Validation failed with {len(self.errors)} error(s).")
        for w in self.warnings:
            self.logger.warning(f"  WARNING: {w}")
        self.logger.info("=== Data Validation Passed ===")
        return True
