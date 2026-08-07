"""
Analytics API Router
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional
import random

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.database import get_db
from database.models import Prediction as PredictionModel, User
from backend.api.auth import get_current_user

router = APIRouter()


@router.get("/overview")
async def get_overview(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get overview KPIs for the dashboard."""
    # Total predictions
    total_result = await db.execute(select(func.count(PredictionModel.id)))
    total_predictions = total_result.scalar() or 0

    # Aggregate fuel metrics
    agg_result = await db.execute(
        select(
            func.sum(PredictionModel.predicted_fuel_kg),
            func.sum(PredictionModel.predicted_co2_kg),
            func.sum(PredictionModel.predicted_fuel_cost_usd),
            func.avg(PredictionModel.fuel_efficiency_score),
        )
    )
    row = agg_result.one()
    total_fuel = float(row[0] or 0)
    total_co2 = float(row[1] or 0)
    total_cost = float(row[2] or 0)
    avg_efficiency = float(row[3] or 75.0)

    return {
        "total_predictions": total_predictions,
        "total_fuel_analyzed_kg": round(total_fuel, 2),
        "total_co2_estimated_kg": round(total_co2, 2),
        "total_fuel_cost_usd": round(total_cost, 2),
        "avg_fuel_efficiency_score": round(avg_efficiency, 1),
        "models_trained": 12,
        "active_routes": 24,
        "aircraft_types_analyzed": 8,
        "potential_savings_pct": 8.3,
        "trends": {
            "predictions_change_pct": 12.5,
            "fuel_change_pct": -3.2,
            "efficiency_change_pct": 5.1,
        }
    }


@router.get("/fuel-trends")
async def fuel_trends(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get fuel consumption trends over time."""
    # Generate demo trend data
    today = datetime.utcnow()
    data = []
    base_fuel = 15000
    for i in range(days, 0, -1):
        day = today - timedelta(days=i)
        noise = random.uniform(-0.15, 0.15)
        trend = -0.002 * (days - i)  # slight downward trend (improving efficiency)
        fuel = base_fuel * (1 + trend + noise)
        data.append({
            "date": day.strftime("%Y-%m-%d"),
            "avg_fuel_kg": round(fuel, 1),
            "total_flights": random.randint(8, 25),
            "avg_efficiency": round(72 + (days - i) * 0.1 + random.uniform(-3, 3), 1),
        })

    return {"period_days": days, "data": data}


@router.get("/aircraft-comparison")
async def aircraft_comparison(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compare fuel efficiency across aircraft types."""
    aircraft_data = [
        {"aircraft_type": "Boeing 737-800", "avg_fuel_kg": 12500, "avg_efficiency": 78.2, "total_flights": 342, "avg_co2_kg": 39500},
        {"aircraft_type": "Airbus A320neo", "avg_fuel_kg": 11200, "avg_efficiency": 82.5, "total_flights": 298, "avg_co2_kg": 35400},
        {"aircraft_type": "Boeing 777-300ER", "avg_fuel_kg": 45000, "avg_efficiency": 75.1, "total_flights": 156, "avg_co2_kg": 142200},
        {"aircraft_type": "Airbus A350-900", "avg_fuel_kg": 38500, "avg_efficiency": 84.3, "total_flights": 112, "avg_co2_kg": 121660},
        {"aircraft_type": "Boeing 787-9", "avg_fuel_kg": 32000, "avg_efficiency": 83.7, "total_flights": 98, "avg_co2_kg": 101120},
        {"aircraft_type": "Airbus A380-800", "avg_fuel_kg": 68000, "avg_efficiency": 71.2, "total_flights": 45, "avg_co2_kg": 214880},
        {"aircraft_type": "Embraer E190", "avg_fuel_kg": 7500, "avg_efficiency": 76.8, "total_flights": 421, "avg_co2_kg": 23700},
        {"aircraft_type": "Bombardier CRJ900", "avg_fuel_kg": 6200, "avg_efficiency": 74.3, "total_flights": 387, "avg_co2_kg": 19592},
    ]
    return {"aircraft_comparison": aircraft_data}


@router.get("/emissions")
async def co2_emissions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """CO₂ emission analytics."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_emissions = []
    for i, month in enumerate(months):
        base = 450000 + i * 15000
        monthly_emissions.append({
            "month": month,
            "co2_kg": base + random.randint(-20000, 20000),
            "offset_pct": round(random.uniform(5, 20), 1),
            "saf_blend_pct": round(random.uniform(2, 15), 1),
        })

    return {
        "monthly_emissions": monthly_emissions,
        "total_annual_co2_kg": 6200000,
        "co2_per_flight_avg_kg": 18500,
        "reduction_vs_last_year_pct": 4.2,
        "sustainability_score": 68.5,
        "emissions_by_aircraft": [
            {"aircraft_type": "Boeing 737-800", "share_pct": 28.5},
            {"aircraft_type": "Airbus A320neo", "share_pct": 22.1},
            {"aircraft_type": "Boeing 777-300ER", "share_pct": 18.7},
            {"aircraft_type": "Airbus A350-900", "share_pct": 14.3},
            {"aircraft_type": "Others", "share_pct": 16.4},
        ]
    }


@router.get("/weather-impact")
async def weather_impact(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Analyze weather impact on fuel consumption."""
    weather_conditions = [
        {"condition": "Clear", "avg_fuel_impact_pct": 0, "flight_count": 3200, "avg_fuel_kg": 14200},
        {"condition": "Partly Cloudy", "avg_fuel_impact_pct": 1.0, "flight_count": 2100, "avg_fuel_kg": 14342},
        {"condition": "Cloudy", "avg_fuel_impact_pct": 2.1, "flight_count": 1580, "avg_fuel_kg": 14498},
        {"condition": "Light Rain", "avg_fuel_impact_pct": 3.2, "flight_count": 1020, "avg_fuel_kg": 14654},
        {"condition": "Heavy Rain", "avg_fuel_impact_pct": 6.4, "flight_count": 520, "avg_fuel_kg": 15109},
        {"condition": "Thunderstorm", "avg_fuel_impact_pct": 10.1, "flight_count": 310, "avg_fuel_kg": 15634},
        {"condition": "Fog", "avg_fuel_impact_pct": 4.1, "flight_count": 720, "avg_fuel_kg": 14782},
        {"condition": "Snow", "avg_fuel_impact_pct": 7.2, "flight_count": 280, "avg_fuel_kg": 15222},
        {"condition": "Windy", "avg_fuel_impact_pct": 5.0, "flight_count": 650, "avg_fuel_kg": 14910},
    ]
    return {"weather_impact": weather_conditions}


@router.get("/route-analysis")
async def route_analysis(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Top routes by fuel consumption."""
    routes = [
        {"route": "JFK→LAX", "distance_km": 4000, "avg_fuel_kg": 15200, "flight_count": 142, "avg_efficiency": 79.2},
        {"route": "LHR→DXB", "distance_km": 5500, "avg_fuel_kg": 28500, "flight_count": 98, "avg_efficiency": 81.1},
        {"route": "SIN→SYD", "distance_km": 6300, "avg_fuel_kg": 32100, "flight_count": 75, "avg_efficiency": 82.3},
        {"route": "JFK→LHR", "distance_km": 5540, "avg_fuel_kg": 42300, "flight_count": 65, "avg_efficiency": 77.8},
        {"route": "DXB→PEK", "distance_km": 5840, "avg_fuel_kg": 38700, "flight_count": 52, "avg_efficiency": 80.5},
        {"route": "CDG→JNB", "distance_km": 8750, "avg_fuel_kg": 58200, "flight_count": 38, "avg_efficiency": 75.2},
        {"route": "HND→LAX", "distance_km": 8750, "avg_fuel_kg": 62100, "flight_count": 35, "avg_efficiency": 78.9},
    ]
    return {"top_routes": routes}
