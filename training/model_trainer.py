"""
Aircraft Fuel Consumption Model
Model Training & Comparison Module

Trains 15+ regression models, evaluates them, and selects the best.
Integrates with MLflow for experiment tracking.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (
    cross_val_score, KFold, learning_curve
)
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error, mean_absolute_percentage_error
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────
def get_model_registry() -> Dict[str, Any]:
    """Return all configured regression models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=15, n_jobs=-1, random_state=42
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        "Support Vector Regression": SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7, weights="distance"),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0
        )

    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=7,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbose=-1
        )

    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=7,
            random_state=42, verbose=0
        )

    return models


# ─────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute comprehensive regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {
        "r2_score": round(r2, 6),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
    }


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """Run k-fold cross-validation."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    cv_mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1))

    return {
        "cv_r2_mean": round(cv_r2.mean(), 6),
        "cv_r2_std": round(cv_r2.std(), 6),
        "cv_mae_mean": round(cv_mae.mean(), 4),
        "cv_rmse_mean": round(cv_rmse.mean(), 4),
    }


# ─────────────────────────────────────────────
# Training & Comparison
# ─────────────────────────────────────────────
def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    experiment_name: str = "aircraft_fuel_consumption",
    mlflow_uri: str = "./experiments/mlruns",
    run_cv: bool = True,
) -> Tuple[Dict, str]:
    """
    Train all models, evaluate, log to MLflow, and return results.

    Returns:
        (results_dict, best_model_name)
    """
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    models = get_model_registry()
    results = {}

    logger.info(f"Training {len(models)} models...")
    logger.info("=" * 60)

    for model_name, model in models.items():
        logger.info(f"Training: {model_name}")
        start_time = time.time()

        with mlflow.start_run(run_name=model_name):
            try:
                # Train
                model.fit(X_train, y_train)
                train_time = time.time() - start_time

                # Evaluate on all splits
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)

                train_metrics = compute_metrics(y_train, y_train_pred)
                val_metrics = compute_metrics(y_val, y_val_pred)
                test_metrics = compute_metrics(y_test, y_test_pred)

                # Cross-validation (optional for large models)
                cv_metrics = {}
                if run_cv and model_name not in ["Support Vector Regression", "KNN Regressor"]:
                    cv_metrics = cross_validate_model(model, X_train, y_train)

                # Log parameters
                try:
                    params = model.get_params()
                    # Limit to simple types for MLflow
                    loggable = {k: v for k, v in params.items()
                                if isinstance(v, (int, float, str, bool, type(None)))}
                    mlflow.log_params(loggable)
                except Exception:
                    pass

                # Log metrics
                mlflow.log_metrics({
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    **cv_metrics,
                    "training_time_s": round(train_time, 3),
                })

                # Log model artifact
                mlflow.sklearn.log_model(model, f"model_{model_name.replace(' ', '_').lower()}")

                # Feature importance
                feat_importance = None
                if hasattr(model, "feature_importances_"):
                    feat_importance = dict(zip(feature_names, model.feature_importances_))
                elif hasattr(model, "coef_"):
                    feat_importance = dict(zip(feature_names, np.abs(model.coef_)))

                results[model_name] = {
                    "model": model,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "cv_metrics": cv_metrics,
                    "training_time_s": round(train_time, 3),
                    "feature_importance": feat_importance,
                    "y_test_pred": y_test_pred.tolist(),
                    "mlflow_run_id": mlflow.active_run().info.run_id,
                }

                logger.info(
                    f"  ✓ {model_name}: R²={test_metrics['r2_score']:.4f}, "
                    f"RMSE={test_metrics['rmse']:.2f}, "
                    f"Time={train_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"  ✗ {model_name} failed: {e}")
                mlflow.log_param("error", str(e))

    # Select best model by test R²
    best_name = max(
        results.keys(),
        key=lambda k: results[k]["test_metrics"]["r2_score"]
    )
    logger.info(f"\n🏆 Best Model: {best_name}")
    logger.info(f"   R² = {results[best_name]['test_metrics']['r2_score']:.4f}")
    logger.info(f"   RMSE = {results[best_name]['test_metrics']['rmse']:.2f} kg")

    return results, best_name


def save_best_model(
    model: Any,
    save_dir: str,
    model_name: str = "best_model.joblib",
    metadata: Optional[dict] = None,
) -> str:
    """Save the best model with metadata."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / model_name
    joblib.dump(model, model_path)

    if metadata:
        meta_path = save_path / "model_metadata.json"
        import json
        # Convert non-serializable types
        serializable_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable_meta[k] = v
            else:
                serializable_meta[k] = str(v)
        with open(meta_path, "w") as f:
            json.dump(serializable_meta, f, indent=2)

    logger.info(f"Best model saved: {model_path}")
    return str(model_path)


def load_model(model_path: str) -> Any:
    """Load a serialized model."""
    return joblib.load(model_path)


def get_learning_curves(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """Compute learning curve data for the best model."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv, scoring="r2", n_jobs=-1,
    )
    return {
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "train_scores_std": train_scores.std(axis=1).tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist(),
        "val_scores_std": val_scores.std(axis=1).tolist(),
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a sorted model comparison DataFrame."""
    rows = []
    for model_name, info in results.items():
        row = {
            "Model": model_name,
            "R² Score": info["test_metrics"]["r2_score"],
            "MAE (kg)": info["test_metrics"]["mae"],
            "RMSE (kg)": info["test_metrics"]["rmse"],
            "MAPE (%)": info["test_metrics"]["mape"],
            "CV R² Mean": info["cv_metrics"].get("cv_r2_mean", None),
            "CV R² Std": info["cv_metrics"].get("cv_r2_std", None),
            "Train Time (s)": info["training_time_s"],
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("R² Score", ascending=False)
    df["Rank"] = range(1, len(df) + 1)
    return df.set_index("Rank")
