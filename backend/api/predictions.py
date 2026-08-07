"""
Predictions API Router
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.database import get_db
from database.models import Prediction as PredictionModel, User
from backend.api.auth import get_current_user

router = APIRouter()


# ── Schemas ────────────────────────────────────────────────
class FlightInputFeatures(BaseModel):
    """Input features for fuel prediction."""
    aircraft_type: str = Field(..., example="Boeing 737-800")
    engine_type: str = Field(default="CFM56-7B", example="CFM56-7B")
    fuel_type: str = Field(default="Jet-A", example="Jet-A")
    engine_thrust_kn: float = Field(default=121.4, ge=0, example=121.4)

    flight_distance_km: float = Field(..., ge=100, le=20000, example=2500)
    flight_duration_hrs: float = Field(..., ge=0.5, le=25, example=3.2)
    altitude_ft: float = Field(default=35000, ge=5000, le=51000, example=35000)
    cruising_speed_kmh: float = Field(default=840, ge=400, le=1100, example=840)

    takeoff_weight_kg: float = Field(..., ge=10000, le=600000, example=65000)
    payload_weight_kg: float = Field(..., ge=0, le=100000, example=18000)
    passenger_count: int = Field(..., ge=0, le=900, example=165)
    cargo_weight_kg: float = Field(default=0, ge=0, le=100000, example=2500)

    oat_celsius: float = Field(default=-50, ge=-100, le=60, example=-50)
    wind_speed_kmh: float = Field(default=30, ge=0, le=400, example=30)
    wind_direction_deg: float = Field(default=180, ge=0, le=360, example=180)
    humidity_pct: float = Field(default=40, ge=0, le=100, example=40)
    pressure_hpa: float = Field(default=250, ge=100, le=1100, example=250)
    weather_condition: str = Field(default="Clear", example="Clear")

    taxi_time_min: float = Field(default=15, ge=0, le=120, example=15)
    climb_time_min: float = Field(default=20, ge=0, le=120, example=20)
    cruise_time_min: float = Field(default=150, ge=0, le=1200, example=150)
    descent_time_min: float = Field(default=20, ge=0, le=120, example=20)


class PredictionRequest(BaseModel):
    features: FlightInputFeatures
    include_recommendations: bool = True
    include_explanation: bool = False


class BatchPredictionRequest(BaseModel):
    flights: List[FlightInputFeatures]
    include_recommendations: bool = False


class PredictionResult(BaseModel):
    prediction_id: str
    predicted_fuel_kg: float
    predicted_co2_kg: float
    predicted_fuel_cost_usd: float
    fuel_efficiency_score: float
    confidence_interval: dict
    recommendations: Optional[List[dict]] = None
    explanation: Optional[dict] = None
    model_name: str
    created_at: datetime


class BatchPredictionResult(BaseModel):
    batch_id: str
    total_flights: int
    predictions: List[dict]
    summary: dict
    created_at: datetime


# ── Helpers ────────────────────────────────────────────────
def get_ml_service(request: Request):
    ml = getattr(request.app.state, "ml_service", None)
    if ml is None or not ml._loaded:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Please run the training pipeline first."
        )
    return ml


# ── Endpoints ──────────────────────────────────────────────
@router.post("/", response_model=PredictionResult, status_code=201)
async def predict_fuel_consumption(
    request: Request,
    body: PredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Predict fuel consumption for a single flight."""
    ml = get_ml_service(request)

    features_dict = body.features.model_dump()
    result = ml.predict(features_dict)

    prediction_id = f"PRED-{uuid.uuid4().hex[:12].upper()}"
    recommendations = None
    if body.include_recommendations:
        recommendations = ml.get_recommendations(features_dict, result["predicted_fuel_kg"])

    explanation = None
    if body.include_explanation and ml.xai_results:
        explanation = {"global_importance": ml.get_feature_importance()[:10]}

    # Persist to database
    db_pred = PredictionModel(
        prediction_id=prediction_id,
        user_id=current_user.id,
        input_features=features_dict,
        predicted_fuel_kg=result["predicted_fuel_kg"],
        predicted_co2_kg=result["predicted_co2_kg"],
        predicted_fuel_cost_usd=result["predicted_fuel_cost_usd"],
        fuel_efficiency_score=result["fuel_efficiency_score"],
        confidence_interval_lower=result["confidence_interval"]["lower"],
        confidence_interval_upper=result["confidence_interval"]["upper"],
        recommendations=recommendations,
        prediction_type="single",
    )
    db.add(db_pred)
    await db.commit()

    return PredictionResult(
        prediction_id=prediction_id,
        predicted_fuel_kg=result["predicted_fuel_kg"],
        predicted_co2_kg=result["predicted_co2_kg"],
        predicted_fuel_cost_usd=result["predicted_fuel_cost_usd"],
        fuel_efficiency_score=result["fuel_efficiency_score"],
        confidence_interval=result["confidence_interval"],
        recommendations=recommendations,
        explanation=explanation,
        model_name=ml.model_name,
        created_at=datetime.utcnow(),
    )


@router.post("/batch", response_model=BatchPredictionResult, status_code=201)
async def batch_predict(
    request: Request,
    body: BatchPredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Batch prediction for multiple flights."""
    if len(body.flights) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 flights per batch")

    ml = get_ml_service(request)
    batch_id = f"BATCH-{uuid.uuid4().hex[:10].upper()}"
    predictions = []
    total_fuel = 0
    total_co2 = 0
    total_cost = 0

    for i, flight in enumerate(body.flights):
        features_dict = flight.model_dump()
        result = ml.predict(features_dict)
        prediction_id = f"PRED-B{i:04d}-{uuid.uuid4().hex[:6].upper()}"

        total_fuel += result["predicted_fuel_kg"]
        total_co2 += result["predicted_co2_kg"]
        total_cost += result["predicted_fuel_cost_usd"]

        rec = ml.get_recommendations(features_dict, result["predicted_fuel_kg"]) if body.include_recommendations else []

        # Persist
        db_pred = PredictionModel(
            prediction_id=prediction_id,
            user_id=current_user.id,
            input_features=features_dict,
            predicted_fuel_kg=result["predicted_fuel_kg"],
            predicted_co2_kg=result["predicted_co2_kg"],
            predicted_fuel_cost_usd=result["predicted_fuel_cost_usd"],
            fuel_efficiency_score=result["fuel_efficiency_score"],
            recommendations=rec,
            prediction_type="batch",
        )
        db.add(db_pred)

        predictions.append({
            "prediction_id": prediction_id,
            "flight_index": i,
            **result,
            "recommendations": rec,
        })

    await db.commit()

    return BatchPredictionResult(
        batch_id=batch_id,
        total_flights=len(body.flights),
        predictions=predictions,
        summary={
            "total_fuel_kg": round(total_fuel, 2),
            "total_co2_kg": round(total_co2, 2),
            "total_fuel_cost_usd": round(total_cost, 2),
            "avg_fuel_kg": round(total_fuel / len(body.flights), 2),
            "avg_efficiency_score": round(
                sum(p["fuel_efficiency_score"] for p in predictions) / len(predictions), 1
            ),
        },
        created_at=datetime.utcnow(),
    )


@router.get("/history", response_model=List[dict])
async def prediction_history(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get prediction history for the current user."""
    result = await db.execute(
        select(PredictionModel)
        .where(PredictionModel.user_id == current_user.id)
        .order_by(desc(PredictionModel.created_at))
        .limit(limit)
        .offset(offset)
    )
    predictions = result.scalars().all()

    return [
        {
            "prediction_id": p.prediction_id,
            "predicted_fuel_kg": p.predicted_fuel_kg,
            "predicted_co2_kg": p.predicted_co2_kg,
            "predicted_fuel_cost_usd": p.predicted_fuel_cost_usd,
            "fuel_efficiency_score": p.fuel_efficiency_score,
            "prediction_type": p.prediction_type,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in predictions
    ]


@router.get("/{prediction_id}", response_model=dict)
async def get_prediction(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific prediction by ID."""
    result = await db.execute(
        select(PredictionModel).where(
            PredictionModel.prediction_id == prediction_id,
            PredictionModel.user_id == current_user.id,
        )
    )
    pred = result.scalar_one_or_none()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return {
        "prediction_id": pred.prediction_id,
        "input_features": pred.input_features,
        "predicted_fuel_kg": pred.predicted_fuel_kg,
        "predicted_co2_kg": pred.predicted_co2_kg,
        "predicted_fuel_cost_usd": pred.predicted_fuel_cost_usd,
        "fuel_efficiency_score": pred.fuel_efficiency_score,
        "confidence_interval": {
            "lower": pred.confidence_interval_lower,
            "upper": pred.confidence_interval_upper,
        },
        "recommendations": pred.recommendations,
        "prediction_type": pred.prediction_type,
        "created_at": pred.created_at.isoformat() if pred.created_at else None,
    }
