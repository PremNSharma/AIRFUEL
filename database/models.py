"""
Aircraft Fuel Consumption Model
Database Models (SQLAlchemy)

Defines all ORM models with normalized schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime,
    ForeignKey, Text, JSON, Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """System users (admins, analysts, operators)."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), default="analyst")  # admin | analyst | viewer
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    predictions = relationship("Prediction", back_populates="user", lazy="select")
    reports = relationship("Report", back_populates="user", lazy="select")

    def __repr__(self):
        return f"<User(username={self.username}, role={self.role})>"


class Aircraft(Base):
    """Aircraft fleet definitions."""
    __tablename__ = "aircraft"

    id = Column(Integer, primary_key=True, index=True)
    aircraft_type = Column(String(100), unique=True, nullable=False, index=True)
    manufacturer = Column(String(50))
    model = Column(String(50))
    engine_type = Column(String(100))
    fuel_type = Column(String(50), default="Jet-A")
    max_seats = Column(Integer)
    max_payload_kg = Column(Float)
    max_takeoff_weight_kg = Column(Float)
    base_fuel_burn_kg_hr = Column(Float)
    cruise_speed_kmh = Column(Float)
    max_altitude_ft = Column(Integer)
    engine_thrust_kn = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    flights = relationship("Flight", back_populates="aircraft_ref", lazy="select")

    def __repr__(self):
        return f"<Aircraft({self.aircraft_type})>"


class Flight(Base):
    """Historical and simulated flight records."""
    __tablename__ = "flights"

    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(String(20), unique=True, nullable=False, index=True)
    aircraft_id = Column(Integer, ForeignKey("aircraft.id"), nullable=True)
    aircraft_type = Column(String(100), index=True)
    flight_date = Column(DateTime, nullable=True)
    departure_airport = Column(String(10), index=True)
    departure_city = Column(String(100))
    departure_country = Column(String(50))
    arrival_airport = Column(String(10), index=True)
    arrival_city = Column(String(100))
    arrival_country = Column(String(50))
    flight_distance_km = Column(Float)
    flight_duration_hrs = Column(Float)
    altitude_ft = Column(Float)
    cruising_speed_kmh = Column(Float)
    takeoff_weight_kg = Column(Float)
    payload_weight_kg = Column(Float)
    passenger_count = Column(Integer)
    cargo_weight_kg = Column(Float)
    oat_celsius = Column(Float)
    wind_speed_kmh = Column(Float)
    wind_direction_deg = Column(Float)
    humidity_pct = Column(Float)
    pressure_hpa = Column(Float)
    weather_condition = Column(String(30))
    taxi_time_min = Column(Float)
    climb_time_min = Column(Float)
    cruise_time_min = Column(Float)
    descent_time_min = Column(Float)
    fuel_consumption_kg = Column(Float, nullable=True)  # actual (if available)
    co2_emission_kg = Column(Float, nullable=True)
    fuel_cost_usd = Column(Float, nullable=True)
    data_source = Column(String(20), default="synthetic")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    aircraft_ref = relationship("Aircraft", back_populates="flights")
    predictions = relationship("Prediction", back_populates="flight", lazy="select")

    __table_args__ = (
        Index("idx_flight_date", "flight_date"),
        Index("idx_flight_route", "departure_airport", "arrival_airport"),
    )

    def __repr__(self):
        return f"<Flight({self.flight_id}: {self.departure_airport}→{self.arrival_airport})>"


class MLModel(Base):
    """Registered ML models."""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    model_class = Column(String(100))
    version = Column(String(20), default="1.0.0")
    file_path = Column(String(500))
    preprocessor_path = Column(String(500))
    r2_score = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    training_samples = Column(Integer)
    n_features = Column(Integer)
    feature_names = Column(JSON)
    hyperparameters = Column(JSON)
    mlflow_run_id = Column(String(100))
    is_active = Column(Boolean, default=False)
    is_best = Column(Boolean, default=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    predictions = relationship("Prediction", back_populates="model", lazy="select")

    def __repr__(self):
        return f"<MLModel({self.name} v{self.version}, R²={self.r2_score:.4f})>"


class Experiment(Base):
    """MLflow experiment tracking records."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String(200), nullable=False)
    run_id = Column(String(100), unique=True, index=True)
    model_name = Column(String(100))
    status = Column(String(20), default="running")  # running | completed | failed
    parameters = Column(JSON)
    metrics = Column(JSON)
    tags = Column(JSON)
    artifact_uri = Column(String(500))
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration_s = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Experiment({self.run_id}: {self.model_name})>"


class Prediction(Base):
    """Fuel consumption predictions."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String(50), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    flight_id = Column(Integer, ForeignKey("flights.id"), nullable=True)
    # Input features
    input_features = Column(JSON)
    # Predictions
    predicted_fuel_kg = Column(Float)
    predicted_co2_kg = Column(Float)
    predicted_fuel_cost_usd = Column(Float)
    fuel_efficiency_score = Column(Float, nullable=True)  # 0-100
    # Actual (if available for comparison)
    actual_fuel_kg = Column(Float, nullable=True)
    prediction_error_pct = Column(Float, nullable=True)
    # Explanation
    shap_values = Column(JSON, nullable=True)
    lime_values = Column(JSON, nullable=True)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    # Recommendations
    recommendations = Column(JSON, nullable=True)
    # Metadata
    prediction_type = Column(String(20), default="single")  # single | batch
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="predictions")
    model = relationship("MLModel", back_populates="predictions")
    flight = relationship("Flight", back_populates="predictions")

    __table_args__ = (
        Index("idx_prediction_created", "created_at"),
    )

    def __repr__(self):
        return f"<Prediction({self.prediction_id}: {self.predicted_fuel_kg:.0f} kg)>"


class Analytics(Base):
    """Aggregated analytics records."""
    __tablename__ = "analytics"

    id = Column(Integer, primary_key=True, index=True)
    period = Column(String(20))  # daily | weekly | monthly
    period_date = Column(DateTime, index=True)
    total_flights = Column(Integer, default=0)
    total_fuel_consumed_kg = Column(Float, default=0.0)
    avg_fuel_per_flight_kg = Column(Float, default=0.0)
    total_co2_kg = Column(Float, default=0.0)
    total_fuel_cost_usd = Column(Float, default=0.0)
    avg_flight_efficiency = Column(Float, default=0.0)
    top_aircraft_type = Column(String(100), nullable=True)
    top_route = Column(String(50), nullable=True)
    anomaly_count = Column(Integer, default=0)
    additional_stats = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Analytics({self.period}: {self.period_date})>"


class Report(Base):
    """Generated reports."""
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(50), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String(200))
    report_type = Column(String(50))  # fuel_analysis | efficiency | emissions | custom
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    content = Column(JSON)
    file_path = Column(String(500), nullable=True)
    status = Column(String(20), default="completed")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="reports")

    def __repr__(self):
        return f"<Report({self.report_id}: {self.title})>"


class AuditLog(Base):
    """System audit logs."""
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    action = Column(String(100))
    resource = Column(String(100))
    resource_id = Column(String(50), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    details = Column(JSON, nullable=True)
    success = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_log_created", "created_at"),
        Index("idx_log_action", "action"),
    )

    def __repr__(self):
        return f"<AuditLog({self.action} by user={self.user_id})>"
