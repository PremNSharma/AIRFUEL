"""
Aircraft Fuel Consumption Model
Master Training Pipeline

Orchestrates: data generation → preprocessing → training → evaluation → XAI → saving
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import mlflow

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from training.data_generator import generate_dataset, save_dataset
from training.preprocessing import load_raw_data, preprocess_for_training, save_preprocessor
from training.model_trainer import (
    train_all_models, save_best_model, build_comparison_table, get_learning_curves
)
from training.explainability import XAIManager, EDAVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainingPipeline")


def run_pipeline(
    n_samples: int = 10000,
    force_regen: bool = False,
    run_cv: bool = True,
) -> dict:
    """Run the complete training pipeline."""

    logger.info("=" * 70)
    logger.info("  AIRCRAFT FUEL CONSUMPTION MODEL — TRAINING PIPELINE")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────
    # Step 1: Data Generation / Loading
    # ─────────────────────────────────────────────
    data_dir = Path(settings.DATA_PATH)
    raw_file = data_dir / settings.RAW_DATA_FILE

    if not raw_file.exists() or force_regen:
        logger.info("Step 1: Generating synthetic aviation dataset...")
        df = generate_dataset(n_samples=n_samples, random_seed=42)
        saved_files = save_dataset(df, str(data_dir), formats=["csv", "parquet", "excel"])
        logger.info(f"  Generated: {len(df)} records → {list(saved_files.values())}")
    else:
        logger.info(f"Step 1: Loading existing dataset from {raw_file}")
        df = load_raw_data(str(raw_file))
        logger.info(f"  Loaded: {len(df)} records")

    # ─────────────────────────────────────────────
    # Step 2: EDA
    # ─────────────────────────────────────────────
    logger.info("Step 2: Running exploratory data analysis...")
    eda_dir = Path("assets/eda")
    visualizer = EDAVisualizer(output_dir=str(eda_dir))

    try:
        visualizer.correlation_heatmap(df)
        visualizer.target_distribution(df)
        logger.info(f"  EDA plots saved to {eda_dir}")
    except Exception as e:
        logger.warning(f"  EDA visualization failed: {e}")

    # ─────────────────────────────────────────────
    # Step 3: Preprocessing
    # ─────────────────────────────────────────────
    logger.info("Step 3: Preprocessing dataset...")
    prep_data = preprocess_for_training(df, test_size=0.2, val_size=0.1)

    X_train = prep_data["X_train"]
    X_val = prep_data["X_val"]
    X_test = prep_data["X_test"]
    y_train = prep_data["y_train"]
    y_val = prep_data["y_val"]
    y_test = prep_data["y_test"]
    preprocessor = prep_data["preprocessor"]
    feature_names = prep_data["feature_names"]

    # Save preprocessor
    preprocessor_path = Path(settings.MODEL_SAVE_PATH) / settings.PREPROCESSOR_NAME
    save_preprocessor(preprocessor, str(preprocessor_path))
    logger.info(f"  Preprocessor saved: {preprocessor_path}")

    # ─────────────────────────────────────────────
    # Step 4: Model Training
    # ─────────────────────────────────────────────
    logger.info("Step 4: Training all models...")
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    results, best_model_name = train_all_models(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
        mlflow_uri=settings.MLFLOW_TRACKING_URI,
        run_cv=run_cv,
    )

    # ─────────────────────────────────────────────
    # Step 5: Save Best Model
    # ─────────────────────────────────────────────
    logger.info(f"Step 5: Saving best model ({best_model_name})...")
    best_model = results[best_model_name]["model"]
    best_metrics = results[best_model_name]["test_metrics"]

    metadata = {
        "model_name": best_model_name,
        "model_class": type(best_model).__name__,
        "test_metrics": best_metrics,
        "feature_names": feature_names,
        "training_samples": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "mlflow_run_id": results[best_model_name]["mlflow_run_id"],
    }

    model_path = save_best_model(
        model=best_model,
        save_dir=settings.MODEL_SAVE_PATH,
        model_name=settings.BEST_MODEL_NAME,
        metadata=metadata,
    )

    # ─────────────────────────────────────────────
    # Step 6: XAI
    # ─────────────────────────────────────────────
    logger.info("Step 6: Computing explainability (SHAP, LIME, Permutation)...")
    try:
        xai = XAIManager(
            model=best_model,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            output_dir="assets/xai",
        )

        global_exp = xai.get_global_explanations()

        # Save XAI results
        xai_path = Path(settings.MODEL_SAVE_PATH) / "xai_results.json"
        with open(xai_path, "w") as f:
            json.dump(global_exp, f, indent=2, default=str)
        logger.info(f"  XAI results saved: {xai_path}")

        # Feature importance plot
        if hasattr(best_model, "feature_importances_"):
            import numpy as np
            visualizer.feature_importance_plot(
                feature_names,
                best_model.feature_importances_,
                title=f"{best_model_name} — Feature Importance"
            )

        # Residual analysis
        y_pred = best_model.predict(X_test)
        visualizer.residual_analysis(y_test, y_pred)
        logger.info("  Residual analysis plots saved")

    except Exception as e:
        logger.error(f"  XAI failed: {e}")
        global_exp = {}

    # ─────────────────────────────────────────────
    # Step 7: Comparison Table
    # ─────────────────────────────────────────────
    logger.info("Step 7: Building model comparison table...")
    comparison_df = build_comparison_table(results)
    comparison_path = Path(settings.MODEL_SAVE_PATH) / "model_comparison.csv"
    comparison_df.to_csv(comparison_path)
    logger.info(f"  Comparison table saved: {comparison_path}")

    # ─────────────────────────────────────────────
    # Learning Curves
    # ─────────────────────────────────────────────
    logger.info("Step 8: Computing learning curves for best model...")
    try:
        lc_data = get_learning_curves(best_model, X_train, y_train, cv=3)
        lc_path = Path(settings.MODEL_SAVE_PATH) / "learning_curves.json"
        with open(lc_path, "w") as f:
            json.dump(lc_data, f, indent=2)
        logger.info(f"  Learning curves saved: {lc_path}")
    except Exception as e:
        logger.warning(f"  Learning curves failed: {e}")
        lc_data = {}

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Best Model: {best_model_name}")
    logger.info(f"  R² Score:   {best_metrics['r2_score']:.4f}")
    logger.info(f"  MAE:        {best_metrics['mae']:.2f} kg")
    logger.info(f"  RMSE:       {best_metrics['rmse']:.2f} kg")
    logger.info(f"  MAPE:       {best_metrics['mape']:.2f}%")
    logger.info("=" * 70)

    print("\n" + comparison_df[["Model", "R² Score", "RMSE (kg)", "MAPE (%)"]].to_string())

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "results": results,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "global_explanations": global_exp,
        "model_path": model_path,
        "learning_curves": lc_data,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aircraft Fuel Consumption Training Pipeline")
    parser.add_argument("--samples", type=int, default=10000, help="Number of synthetic samples")
    parser.add_argument("--force-regen", action="store_true", help="Force dataset regeneration")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    args = parser.parse_args()

    run_pipeline(
        n_samples=args.samples,
        force_regen=args.force_regen,
        run_cv=not args.no_cv,
    )
