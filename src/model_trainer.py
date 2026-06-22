import pickle
import sys
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config, ensure_dirs


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("ModelTrainer", config)
        ensure_dirs("artifacts")

    def get_candidates(self):
        rs = self.config["model"]["random_state"]
        return {
            "RandomForest": {
                "model": RandomForestRegressor(random_state=rs),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                }
            },
            "GradientBoosting": {
                "model": GradientBoostingRegressor(random_state=rs),
                "params": {
                    "n_estimators": [100, 150],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                }
            },
            "Ridge": {
                "model": Ridge(),
                "params": {
                    "alpha": [0.1, 1.0, 10.0, 100.0],
                }
            }
        }

    def train_with_grid_search(self, model, params, X_train, y_train, name):
        cv = KFold(
            n_splits=self.config["model"]["cv_folds"],
            shuffle=True,
            random_state=self.config["model"]["random_state"]
        )
        self.logger.info(f"GridSearchCV for {name}...")
        gs = GridSearchCV(model, params, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        rmse = (-gs.best_score_) ** 0.5
        self.logger.info(f"  {name} best params: {gs.best_params_}")
        self.logger.info(f"  {name} CV RMSE: {rmse:.4f}")
        return gs.best_estimator_, rmse

    def run(self, X_train, y_train):
        self.logger.info("=== Model Training Started ===")
        self.logger.info(f"Training samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")

        candidates = self.get_candidates()
        results = {}
        for name, cfg in candidates.items():
            est, rmse = self.train_with_grid_search(cfg["model"], cfg["params"], X_train, y_train, name)
            results[name] = {"estimator": est, "cv_rmse": rmse}

        best_name = min(results, key=lambda k: results[k]["cv_rmse"])
        best_model = results[best_name]["estimator"]
        self.logger.info(f"\n>> Best model: {best_name} (CV RMSE = {results[best_name]['cv_rmse']:.4f})")

        with open(self.config["model"]["model_path"], "wb") as f:
            pickle.dump(best_model, f)
        self.logger.info(f"Model saved to {self.config['model']['model_path']}")

        self.logger.info("=== Model Training Complete ===")
        return best_model, best_name, results
