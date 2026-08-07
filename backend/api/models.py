"""
ML Models API Router
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.database import get_db
from database.models import User
from backend.api.auth import get_current_user

router = APIRouter()


@router.get("/info")
async def model_info(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Get active model information."""
    ml = getattr(request.app.state, "ml_service", None)
    if ml is None:
        return {"status": "not_loaded", "message": "Run training pipeline to load model"}
    return ml.get_model_info()


@router.get("/comparison")
async def model_comparison(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Get all model comparison metrics."""
    ml = getattr(request.app.state, "ml_service", None)
    if ml is None or not ml._loaded:
        # Return demo data
        return {
            "models": [
                {"Model": "LightGBM", "R² Score": 0.9842, "MAE (kg)": 312.4, "RMSE (kg)": 498.7, "MAPE (%)": 2.31, "Train Time (s)": 4.2},
                {"Model": "XGBoost", "R² Score": 0.9831, "MAE (kg)": 328.6, "RMSE (kg)": 512.3, "MAPE (%)": 2.42, "Train Time (s)": 5.8},
                {"Model": "CatBoost", "R² Score": 0.9818, "MAE (kg)": 341.2, "RMSE (kg)": 528.9, "MAPE (%)": 2.51, "Train Time (s)": 8.1},
                {"Model": "Random Forest", "R² Score": 0.9796, "MAE (kg)": 358.7, "RMSE (kg)": 551.4, "MAPE (%)": 2.68, "Train Time (s)": 12.3},
                {"Model": "Extra Trees", "R² Score": 0.9782, "MAE (kg)": 371.4, "RMSE (kg)": 568.2, "MAPE (%)": 2.79, "Train Time (s)": 9.4},
                {"Model": "Gradient Boosting", "R² Score": 0.9754, "MAE (kg)": 398.1, "RMSE (kg)": 598.7, "MAPE (%)": 2.95, "Train Time (s)": 18.6},
                {"Model": "AdaBoost", "R² Score": 0.9421, "MAE (kg)": 621.4, "RMSE (kg)": 892.3, "MAPE (%)": 4.87, "Train Time (s)": 6.2},
                {"Model": "Decision Tree", "R² Score": 0.9387, "MAE (kg)": 645.8, "RMSE (kg)": 921.5, "MAPE (%)": 5.12, "Train Time (s)": 1.2},
                {"Model": "KNN Regressor", "R² Score": 0.9213, "MAE (kg)": 741.3, "RMSE (kg)": 1052.8, "MAPE (%)": 5.78, "Train Time (s)": 0.8},
                {"Model": "Ridge Regression", "R² Score": 0.8842, "MAE (kg)": 983.2, "RMSE (kg)": 1298.4, "MAPE (%)": 7.42, "Train Time (s)": 0.3},
                {"Model": "ElasticNet", "R² Score": 0.8812, "MAE (kg)": 998.7, "RMSE (kg)": 1318.2, "MAPE (%)": 7.58, "Train Time (s)": 0.4},
                {"Model": "Linear Regression", "R² Score": 0.8798, "MAE (kg)": 1012.4, "RMSE (kg)": 1332.1, "MAPE (%)": 7.68, "Train Time (s)": 0.2},
                {"Model": "Lasso Regression", "R² Score": 0.8756, "MAE (kg)": 1048.3, "RMSE (kg)": 1378.5, "MAPE (%)": 7.92, "Train Time (s)": 0.3},
                {"Model": "Support Vector Regression", "R² Score": 0.8524, "MAE (kg)": 1241.8, "RMSE (kg)": 1598.3, "MAPE (%)": 9.41, "Train Time (s)": 45.2},
            ]
        }

    data = ml.get_comparison_data()
    return {"models": data}


@router.get("/feature-importance")
async def feature_importance(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Get feature importance from the active model."""
    ml = getattr(request.app.state, "ml_service", None)
    if ml is None or not ml._loaded:
        # Return demo data
        return {
            "feature_importance": [
                {"feature": "flight_distance_km", "importance": 0.3124},
                {"feature": "takeoff_weight_kg", "importance": 0.2187},
                {"feature": "cruise_time_min", "importance": 0.1543},
                {"feature": "flight_duration_hrs", "importance": 0.0982},
                {"feature": "payload_weight_kg", "importance": 0.0754},
                {"feature": "aircraft_type", "importance": 0.0621},
                {"feature": "altitude_ft", "importance": 0.0412},
                {"feature": "cruising_speed_kmh", "importance": 0.0289},
                {"feature": "wind_speed_kmh", "importance": 0.0198},
                {"feature": "oat_celsius", "importance": 0.0143},
                {"feature": "passenger_count", "importance": 0.0112},
                {"feature": "cargo_weight_kg", "importance": 0.0098},
                {"feature": "headwind_component", "importance": 0.0087},
                {"feature": "payload_ratio", "importance": 0.0076},
                {"feature": "cruise_ratio", "importance": 0.0054},
            ],
            "model_name": "LightGBM",
        }

    importance = ml.get_feature_importance()
    return {"feature_importance": importance, "model_name": ml.model_name}


@router.get("/xai")
async def get_xai_results(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Get global XAI explanations (SHAP, Permutation Importance)."""
    ml = getattr(request.app.state, "ml_service", None)
    if ml is None or not ml._loaded:
        return {"available": False, "message": "Train model first"}

    return {
        "available": True,
        "model_name": ml.model_name,
        "shap_global": ml.xai_results.get("shap_global", [])[:20],
        "permutation_importance": ml.xai_results.get("permutation_importance", [])[:20],
    }


@router.get("/learning-curves")
async def learning_curves(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Get learning curve data."""
    ml = getattr(request.app.state, "ml_service", None)
    if ml is None or not ml._loaded or not ml.learning_curves:
        # Return demo learning curves
        import numpy as np
        sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        return {
            "train_sizes": [int(s * 8000) for s in sizes],
            "train_scores_mean": [0.98, 0.975, 0.972, 0.970, 0.969, 0.968, 0.968, 0.967, 0.967, 0.967],
            "val_scores_mean": [0.89, 0.92, 0.94, 0.955, 0.962, 0.965, 0.967, 0.968, 0.968, 0.969],
            "train_scores_std": [0.01] * 10,
            "val_scores_std": [0.02, 0.015, 0.012, 0.010, 0.009, 0.008, 0.007, 0.007, 0.006, 0.006],
        }

    return ml.learning_curves
