"""
Flights API Router
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from pydantic import BaseModel
from datetime import datetime
import random

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.database import get_db
from database.models import User, Flight
from backend.api.auth import get_current_user

router = APIRouter()


@router.get("/")
async def list_flights(
    limit: int = Query(default=50, le=500),
    offset: int = 0,
    aircraft_type: Optional[str] = None,
    departure: Optional[str] = None,
    arrival: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List flight records."""
    query = select(Flight)
    if aircraft_type:
        query = query.where(Flight.aircraft_type == aircraft_type)
    if departure:
        query = query.where(Flight.departure_airport == departure.upper())
    if arrival:
        query = query.where(Flight.arrival_airport == arrival.upper())

    query = query.order_by(desc(Flight.id)).limit(limit).offset(offset)
    result = await db.execute(query)
    flights = result.scalars().all()

    return {
        "flights": [
            {
                "id": f.id,
                "flight_id": f.flight_id,
                "aircraft_type": f.aircraft_type,
                "departure_airport": f.departure_airport,
                "arrival_airport": f.arrival_airport,
                "flight_distance_km": f.flight_distance_km,
                "flight_duration_hrs": f.flight_duration_hrs,
                "fuel_consumption_kg": f.fuel_consumption_kg,
                "passenger_count": f.passenger_count,
            }
            for f in flights
        ],
        "total": len(flights),
        "offset": offset,
        "limit": limit,
    }


@router.get("/stats")
async def flight_statistics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get aggregated flight statistics."""
    result = await db.execute(
        select(
            func.count(Flight.id),
            func.avg(Flight.flight_distance_km),
            func.avg(Flight.fuel_consumption_kg),
            func.sum(Flight.fuel_consumption_kg),
        )
    )
    row = result.one()

    # Add demo data for rich display
    return {
        "total_flights_in_db": int(row[0] or 0),
        "avg_distance_km": round(float(row[1] or 3200), 1),
        "avg_fuel_consumption_kg": round(float(row[2] or 14500), 1),
        "total_fuel_consumed_kg": round(float(row[3] or 0), 1),
        "top_aircraft_types": [
            {"type": "Boeing 737-800", "count": 342, "avg_fuel_kg": 12500},
            {"type": "Airbus A320neo", "count": 298, "avg_fuel_kg": 11200},
            {"type": "Boeing 777-300ER", "count": 156, "avg_fuel_kg": 45000},
            {"type": "Airbus A350-900", "count": 112, "avg_fuel_kg": 38500},
        ],
        "top_routes": [
            {"route": "JFK-LAX", "count": 145, "avg_fuel_kg": 15200},
            {"route": "LHR-DXB", "count": 98, "avg_fuel_kg": 28500},
            {"route": "SIN-SYD", "count": 75, "avg_fuel_kg": 32100},
        ],
    }


@router.get("/aircraft-types")
async def aircraft_types(
    current_user: User = Depends(get_current_user),
):
    """Get list of available aircraft types."""
    return {
        "aircraft_types": [
            {
                "type": "Boeing 737-800",
                "manufacturer": "Boeing",
                "engine": "CFM56-7B",
                "max_seats": 189,
                "cruise_speed_kmh": 842,
                "max_range_km": 5765,
            },
            {
                "type": "Boeing 777-300ER",
                "manufacturer": "Boeing",
                "engine": "GE90-115B",
                "max_seats": 396,
                "cruise_speed_kmh": 905,
                "max_range_km": 13650,
            },
            {
                "type": "Airbus A320neo",
                "manufacturer": "Airbus",
                "engine": "CFM LEAP-1A",
                "max_seats": 194,
                "cruise_speed_kmh": 833,
                "max_range_km": 6300,
            },
            {
                "type": "Airbus A350-900",
                "manufacturer": "Airbus",
                "engine": "Rolls-Royce Trent XWB",
                "max_seats": 369,
                "cruise_speed_kmh": 903,
                "max_range_km": 15000,
            },
            {
                "type": "Boeing 787-9",
                "manufacturer": "Boeing",
                "engine": "GEnx-1B",
                "max_seats": 296,
                "cruise_speed_kmh": 903,
                "max_range_km": 14140,
            },
            {
                "type": "Airbus A380-800",
                "manufacturer": "Airbus",
                "engine": "Rolls-Royce Trent 970",
                "max_seats": 853,
                "cruise_speed_kmh": 903,
                "max_range_km": 15200,
            },
            {
                "type": "Embraer E190",
                "manufacturer": "Embraer",
                "engine": "GE CF34-10E",
                "max_seats": 114,
                "cruise_speed_kmh": 870,
                "max_range_km": 4537,
            },
            {
                "type": "Bombardier CRJ900",
                "manufacturer": "Bombardier",
                "engine": "GE CF34-8C5",
                "max_seats": 90,
                "cruise_speed_kmh": 870,
                "max_range_km": 2957,
            },
        ]
    }
