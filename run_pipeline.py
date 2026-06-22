"""
run_pipeline.py — Runs the full Air Quality Forecasting pipeline.
Run once to fetch data, train model and save artifacts.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, get_logger
from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.data_preprocessing import DataPreprocessing
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator


def run_pipeline():
    config = load_config("config.yaml")
    logger = get_logger("Pipeline", config)

    logger.info("=" * 60)
    logger.info("   AIR QUALITY FORECASTING — ML PIPELINE STARTED")
    logger.info("=" * 60)

    logger.info("\n[STAGE 1] Data Ingestion — Fetching from OpenWeatherMap API")
    df = DataIngestion(config).run()

    logger.info("\n[STAGE 2] Data Validation")
    DataValidation(config).run(df)

    logger.info("\n[STAGE 3] Data Preprocessing")
    X_train, X_test, y_train, y_test = DataPreprocessing(config).run(df)

    logger.info("\n[STAGE 4] Model Training")
    model, model_name, all_results = ModelTrainer(config).run(X_train, y_train)

    logger.info("\n[STAGE 5] Model Evaluation")
    report = ModelEvaluator(config).run(model, X_test, y_test, model_name, all_results)

    logger.info("\n" + "=" * 60)
    logger.info("   PIPELINE COMPLETE")
    logger.info(f"   Best Model   : {model_name}")
    logger.info(f"   RMSE         : {report['test_metrics']['rmse']}")
    logger.info(f"   R² Score     : {report['test_metrics']['r2_score']}")
    logger.info(f"   Within-1 AQI : {report['test_metrics']['within_1_aqi_accuracy']}%")
    logger.info("=" * 60)
    return report


if __name__ == "__main__":
    run_pipeline()
