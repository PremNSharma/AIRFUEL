"""
Aircraft Fuel Consumption Model
Synthetic Aviation Dataset Generator

Generates realistic aviation operational data for model training.
Includes realistic physics-based relationships between features and fuel consumption.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Aircraft Fleet Definitions
# ─────────────────────────────────────────────
AIRCRAFT_TYPES = {
    "Boeing 737-800": {
        "engine_type": "CFM56-7B",
        "fuel_type": "Jet-A",
        "max_seats": 189,
        "max_payload_kg": 20000,
        "max_takeoff_weight_kg": 79016,
        "base_fuel_burn_kg_hr": 2600,
        "cruise_speed_kmh": 842,
        "max_altitude_ft": 41100,
        "engine_thrust_kn": 121.4,
    },
    "Boeing 777-300ER": {
        "engine_type": "GE90-115B",
        "fuel_type": "Jet-A",
        "max_seats": 396,
        "max_payload_kg": 64140,
        "max_takeoff_weight_kg": 352400,
        "base_fuel_burn_kg_hr": 7800,
        "cruise_speed_kmh": 905,
        "max_altitude_ft": 43100,
        "engine_thrust_kn": 513.9,
    },
    "Airbus A320neo": {
        "engine_type": "CFM LEAP-1A",
        "fuel_type": "Jet-A",
        "max_seats": 194,
        "max_payload_kg": 20000,
        "max_takeoff_weight_kg": 79000,
        "base_fuel_burn_kg_hr": 2300,
        "cruise_speed_kmh": 833,
        "max_altitude_ft": 39800,
        "engine_thrust_kn": 120.0,
    },
    "Airbus A350-900": {
        "engine_type": "Rolls-Royce Trent XWB",
        "fuel_type": "Jet-A",
        "max_seats": 369,
        "max_payload_kg": 53000,
        "max_takeoff_weight_kg": 280000,
        "base_fuel_burn_kg_hr": 6200,
        "cruise_speed_kmh": 903,
        "max_altitude_ft": 43100,
        "engine_thrust_kn": 374.0,
    },
    "Boeing 787-9": {
        "engine_type": "GEnx-1B",
        "fuel_type": "Jet-A",
        "max_seats": 296,
        "max_payload_kg": 53000,
        "max_takeoff_weight_kg": 254011,
        "base_fuel_burn_kg_hr": 5400,
        "cruise_speed_kmh": 903,
        "max_altitude_ft": 43000,
        "engine_thrust_kn": 320.0,
    },
    "Airbus A380-800": {
        "engine_type": "Rolls-Royce Trent 970",
        "fuel_type": "Jet-A",
        "max_seats": 853,
        "max_payload_kg": 84000,
        "max_takeoff_weight_kg": 575000,
        "base_fuel_burn_kg_hr": 11300,
        "cruise_speed_kmh": 903,
        "max_altitude_ft": 43000,
        "engine_thrust_kn": 340.0,
    },
    "Embraer E190": {
        "engine_type": "GE CF34-10E",
        "fuel_type": "Jet-A",
        "max_seats": 114,
        "max_payload_kg": 11400,
        "max_takeoff_weight_kg": 52290,
        "base_fuel_burn_kg_hr": 1650,
        "cruise_speed_kmh": 870,
        "max_altitude_ft": 41000,
        "engine_thrust_kn": 82.3,
    },
    "Bombardier CRJ900": {
        "engine_type": "GE CF34-8C5",
        "fuel_type": "Jet-A",
        "max_seats": 90,
        "max_payload_kg": 10500,
        "max_takeoff_weight_kg": 38329,
        "base_fuel_burn_kg_hr": 1450,
        "cruise_speed_kmh": 870,
        "max_altitude_ft": 41000,
        "engine_thrust_kn": 59.0,
    },
}

AIRPORTS = [
    ("JFK", "New York", 40.6413, -73.7781, "USA"),
    ("LAX", "Los Angeles", 33.9425, -118.4081, "USA"),
    ("ORD", "Chicago", 41.9742, -87.9073, "USA"),
    ("DFW", "Dallas", 32.8998, -97.0403, "USA"),
    ("ATL", "Atlanta", 33.6407, -84.4277, "USA"),
    ("DEN", "Denver", 39.8561, -104.6737, "USA"),
    ("SFO", "San Francisco", 37.6213, -122.379, "USA"),
    ("SEA", "Seattle", 47.4502, -122.3088, "USA"),
    ("LHR", "London", 51.4775, -0.4614, "UK"),
    ("CDG", "Paris", 49.0097, 2.5479, "France"),
    ("FRA", "Frankfurt", 50.0379, 8.5622, "Germany"),
    ("AMS", "Amsterdam", 52.3086, 4.7639, "Netherlands"),
    ("DXB", "Dubai", 25.2528, 55.3644, "UAE"),
    ("SIN", "Singapore", 1.3644, 103.9915, "Singapore"),
    ("HND", "Tokyo", 35.5494, 139.7798, "Japan"),
    ("PEK", "Beijing", 40.0799, 116.6031, "China"),
    ("SYD", "Sydney", -33.9461, 151.177, "Australia"),
    ("GRU", "Sao Paulo", -23.4356, -46.4731, "Brazil"),
    ("JNB", "Johannesburg", -26.1367, 28.246, "South Africa"),
    ("BOM", "Mumbai", 19.0896, 72.8656, "India"),
    ("DEL", "Delhi", 28.5562, 77.1, "India"),
    ("ICN", "Seoul", 37.4602, 126.4407, "South Korea"),
    ("YYZ", "Toronto", 43.6777, -79.6248, "Canada"),
    ("MEX", "Mexico City", 19.4363, -99.0721, "Mexico"),
    ("MAD", "Madrid", 40.4936, -3.5668, "Spain"),
]

WEATHER_CONDITIONS = ["Clear", "Partly Cloudy", "Cloudy", "Light Rain", "Heavy Rain",
                       "Thunderstorm", "Fog", "Snow", "Haze", "Windy"]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two airport coordinates in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_fuel_consumption(row: dict, aircraft: dict) -> float:
    """
    Physics-informed fuel consumption calculation (kg).
    Combines aerodynamics, weight, weather, and operational factors.
    """
    base_burn = aircraft["base_fuel_burn_kg_hr"]
    flight_hrs = row["flight_duration_hrs"]
    distance_km = row["flight_distance_km"]

    # Weight factor: heavier aircraft burns more (~15% increase per 10% weight increase)
    max_mtow = aircraft["max_takeoff_weight_kg"]
    weight_ratio = row["takeoff_weight_kg"] / max_mtow
    weight_factor = 0.75 + 0.5 * weight_ratio

    # Altitude efficiency: optimal at cruise altitude (FL350-FL390)
    optimal_alt = 38000
    alt_dev = abs(row["altitude_ft"] - optimal_alt) / optimal_alt
    altitude_factor = 1.0 + 0.15 * alt_dev

    # Speed factor: deviation from optimal cruise speed
    optimal_speed = aircraft["cruise_speed_kmh"]
    speed_dev = abs(row["cruising_speed_kmh"] - optimal_speed) / optimal_speed
    speed_factor = 1.0 + 0.2 * speed_dev

    # Wind factor: headwind increases fuel burn, tailwind decreases
    wind_component = row["wind_speed_kmh"] * np.cos(np.radians(row["wind_direction_deg"]))
    wind_factor = 1.0 - 0.03 * (wind_component / 100)  # ±3% per 100 km/h

    # Temperature factor: ISA deviation affects engine efficiency
    isa_temp = 15 - 1.98 * (row["altitude_ft"] / 1000)
    temp_dev = row["oat_celsius"] - isa_temp
    temp_factor = 1.0 + 0.003 * temp_dev

    # Weather factor
    weather_factors = {
        "Clear": 1.0, "Partly Cloudy": 1.01, "Cloudy": 1.02,
        "Light Rain": 1.03, "Heavy Rain": 1.06, "Thunderstorm": 1.10,
        "Fog": 1.04, "Snow": 1.07, "Haze": 1.02, "Windy": 1.05,
    }
    weather_factor = weather_factors.get(row["weather_condition"], 1.0)

    # Phase-based fuel: taxi + climb burn more than cruise
    taxi_fuel = row["taxi_time_min"] * (base_burn / 60) * 1.3  # idle thrust
    climb_fuel = row["climb_time_min"] * (base_burn / 60) * 1.4  # high thrust
    cruise_fuel = row["cruise_time_min"] * (base_burn / 60) * weight_factor * altitude_factor * speed_factor * wind_factor * temp_factor
    descent_fuel = row["descent_time_min"] * (base_burn / 60) * 0.65  # reduced thrust

    total_fuel = (taxi_fuel + climb_fuel + cruise_fuel + descent_fuel) * weather_factor

    # Add reserve fuel (5% contingency)
    total_fuel *= 1.05

    return round(max(total_fuel, 100), 2)


def generate_dataset(n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic aviation dataset."""
    np.random.seed(random_seed)
    random.seed(random_seed)

    logger.info(f"Generating {n_samples} synthetic aviation records...")

    records = []
    aircraft_names = list(AIRCRAFT_TYPES.keys())
    start_date = datetime(2022, 1, 1)

    for i in range(n_samples):
        # ── Aircraft selection ─────────────────────────
        ac_name = random.choice(aircraft_names)
        ac = AIRCRAFT_TYPES[ac_name]

        # ── Route selection ────────────────────────────
        dep_idx, arr_idx = random.sample(range(len(AIRPORTS)), 2)
        dep = AIRPORTS[dep_idx]
        arr = AIRPORTS[arr_idx]

        distance_km = haversine_distance(dep[2], dep[3], arr[2], arr[3])
        # Ensure minimum distance
        distance_km = max(distance_km, 200)

        # ── Operational parameters ─────────────────────
        flight_speed = ac["cruise_speed_kmh"] * np.random.uniform(0.92, 1.05)
        flight_duration_hrs = (distance_km / flight_speed) + np.random.uniform(0.2, 0.8)

        # Phase times (minutes)
        taxi_time = np.random.normal(18, 5)
        climb_time = np.random.normal(distance_km * 0.025 + 10, 5)
        descent_time = np.random.normal(distance_km * 0.020 + 8, 4)
        cruise_time = max(5.0, flight_duration_hrs * 60 - taxi_time - climb_time - descent_time)

        # ── Weight parameters ──────────────────────────
        pax_count = int(ac["max_seats"] * np.random.uniform(0.55, 0.99))
        pax_weight = pax_count * np.random.normal(95, 10)  # avg passenger+baggage
        cargo_weight = np.random.exponential(ac["max_payload_kg"] * 0.3)
        cargo_weight = min(cargo_weight, ac["max_payload_kg"] - pax_weight)
        payload_weight = pax_weight + cargo_weight
        takeoff_weight = np.random.uniform(0.55, 0.98) * ac["max_takeoff_weight_kg"]
        takeoff_weight = max(takeoff_weight, payload_weight * 1.5)
        takeoff_weight = min(takeoff_weight, ac["max_takeoff_weight_kg"])

        # ── Environmental parameters ───────────────────
        altitude_ft = np.random.normal(36000, 3000)
        altitude_ft = np.clip(altitude_ft, 18000, ac["max_altitude_ft"])

        cruising_speed = flight_speed * np.random.uniform(0.96, 1.04)

        wind_speed = abs(np.random.normal(30, 25))
        wind_direction = np.random.uniform(0, 360)
        humidity = np.random.uniform(10, 95)

        oat_celsius = 15 - 1.98 * (altitude_ft / 1000) + np.random.normal(0, 8)
        pressure_hpa = 1013.25 * (1 - 2.2577e-5 * altitude_ft * 0.3048) ** 5.2561

        weather_condition = np.random.choice(
            WEATHER_CONDITIONS,
            p=[0.30, 0.20, 0.15, 0.10, 0.05, 0.03, 0.07, 0.04, 0.04, 0.02]
        )

        # ── Timestamp ──────────────────────────────────
        flight_date = start_date + timedelta(
            days=random.randint(0, 730),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        row = {
            "flight_id": f"FL{i:07d}",
            "flight_date": flight_date.strftime("%Y-%m-%d %H:%M:%S"),
            "aircraft_type": ac_name,
            "engine_type": ac["engine_type"],
            "fuel_type": ac["fuel_type"],
            "engine_thrust_kn": ac["engine_thrust_kn"],
            "departure_airport": dep[0],
            "departure_city": dep[1],
            "departure_country": dep[4],
            "arrival_airport": arr[0],
            "arrival_city": arr[1],
            "arrival_country": arr[4],
            "flight_distance_km": round(distance_km, 2),
            "flight_duration_hrs": round(flight_duration_hrs, 3),
            "altitude_ft": round(altitude_ft, 0),
            "cruising_speed_kmh": round(cruising_speed, 1),
            "takeoff_weight_kg": round(takeoff_weight, 0),
            "payload_weight_kg": round(payload_weight, 0),
            "passenger_count": pax_count,
            "cargo_weight_kg": round(max(cargo_weight, 0), 0),
            "oat_celsius": round(oat_celsius, 1),
            "wind_speed_kmh": round(wind_speed, 1),
            "wind_direction_deg": round(wind_direction, 0),
            "humidity_pct": round(humidity, 1),
            "pressure_hpa": round(pressure_hpa, 2),
            "weather_condition": weather_condition,
            "taxi_time_min": round(max(taxi_time, 5), 1),
            "climb_time_min": round(max(climb_time, 5), 1),
            "cruise_time_min": round(max(cruise_time, 5), 1),
            "descent_time_min": round(max(descent_time, 3), 1),
        }

        # ── Compute fuel consumption ───────────────────
        fuel_kg = compute_fuel_consumption(row, ac)
        row["fuel_consumption_kg"] = fuel_kg
        row["co2_emission_kg"] = round(fuel_kg * 3.16, 2)  # Jet-A CO₂ factor
        row["fuel_cost_usd"] = round(fuel_kg * 0.82, 2)    # avg jet fuel price

        records.append(row)

    df = pd.DataFrame(records)

    # ── Introduce realistic missingness (~2%) ──────────
    missing_cols = ["wind_speed_kmh", "humidity_pct", "oat_celsius", "pressure_hpa"]
    for col in missing_cols:
        mask = np.random.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    # ── Introduce a few duplicates (~0.5%) ─────────────
    dup_indices = np.random.choice(len(df), size=int(len(df) * 0.005), replace=False)
    df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

    logger.info(f"Generated dataset: {len(df)} records, {len(df.columns)} features")
    return df


def save_dataset(
    df: pd.DataFrame,
    output_dir: str,
    formats: Optional[list] = None
) -> dict:
    """Save dataset in multiple formats."""
    if formats is None:
        formats = ["csv", "parquet", "excel"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = {}

    if "csv" in formats:
        path = output_path / "aircraft_fuel_data.csv"
        df.to_csv(path, index=False)
        saved["csv"] = str(path)
        logger.info(f"Saved CSV: {path}")

    if "parquet" in formats:
        path = output_path / "aircraft_fuel_data.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        saved["parquet"] = str(path)
        logger.info(f"Saved Parquet: {path}")

    if "excel" in formats:
        path = output_path / "aircraft_fuel_data.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")
        saved["excel"] = str(path)
        logger.info(f"Saved Excel: {path}")

    if "json" in formats:
        path = output_path / "aircraft_fuel_data.json"
        df.to_json(path, orient="records", indent=2)
        saved["json"] = str(path)
        logger.info(f"Saved JSON: {path}")

    return saved


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    df = generate_dataset(n_samples=n)
    saved = save_dataset(df, "./data")
    print(f"\n✅ Dataset generated: {n} samples")
    print(f"   Files: {list(saved.values())}")
    print(f"   Target stats: min={df['fuel_consumption_kg'].min():.0f}kg, "
          f"max={df['fuel_consumption_kg'].max():.0f}kg, "
          f"mean={df['fuel_consumption_kg'].mean():.0f}kg")
