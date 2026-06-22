import json
import pickle
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config, ensure_dirs


class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("ModelEvaluator", config)
        ensure_dirs("artifacts")

    def compute_metrics(self, y_true, y_pred) -> dict:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        # Within-1-AQI-category accuracy
        within_1 = float(np.mean(np.abs(y_pred - y_true) <= 1.0) * 100)
        return {
            "rmse":     round(rmse, 4),
            "mae":      round(mae, 4),
            "r2_score": round(r2, 4),
            "within_1_aqi_accuracy": round(within_1, 2)
        }

    def get_feature_importance(self, model, feature_names: list) -> dict:
        if hasattr(model, "feature_importances_"):
            fi = dict(sorted(
                zip(feature_names, model.feature_importances_.tolist()),
                key=lambda x: x[1], reverse=True
            ))
            return fi
        elif hasattr(model, "coef_"):
            fi = dict(sorted(
                zip(feature_names, np.abs(model.coef_).tolist()),
                key=lambda x: x[1], reverse=True
            ))
            return fi
        return {}

    def run(self, model, X_test, y_test, model_name: str, all_results: dict = None):
        self.logger.info("=== Model Evaluation Started ===")

        y_pred = model.predict(X_test)
        # Clip predictions to valid AQI range 1-5
        y_pred = np.clip(np.round(y_pred), 1, 5)

        metrics = self.compute_metrics(y_test.values, y_pred)
        fi = self.get_feature_importance(model, list(X_test.columns))

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"  MODEL: {model_name}")
        self.logger.info(f"{'='*50}")
        for k, v in metrics.items():
            self.logger.info(f"  {k.upper():<30}: {v}")
        if fi:
            top5 = list(fi.items())[:5]
            self.logger.info(f"  Top 5 Features: {top5}")

        comparison = {}
        if all_results:
            for name, res in all_results.items():
                comparison[name] = {"cv_rmse": round(res["cv_rmse"], 4)}

        report = {
            "model_name": model_name,
            "test_metrics": metrics,
            "top_features": dict(list(fi.items())[:10]),
            "model_comparison": comparison
        }

        with open(self.config["model"]["report_path"], "w") as f:
            json.dump(report, f, indent=4)
        self.logger.info(f"Report saved to {self.config['model']['report_path']}")
        self.logger.info("=== Evaluation Complete ===")
        return report
