"""
Aircraft Fuel Consumption Model
Configuration Module
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────
    APP_NAME: str = "Aircraft Fuel Consumption Model"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # ── API Server ───────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = True

    # ── Frontend / CORS ──────────────────────────────────
    FRONTEND_URL: str = "http://localhost:5173"
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v: str) -> str:
        return v

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    # ── Security / JWT ───────────────────────────────────
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/database/aircraft_fuel.db"

    # ── MLflow ───────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = str(BASE_DIR / "experiments" / "mlruns")
    MLFLOW_EXPERIMENT_NAME: str = "aircraft_fuel_consumption"
    MLFLOW_ARTIFACT_ROOT: str = str(BASE_DIR / "experiments" / "artifacts")

    # ── Model ────────────────────────────────────────────
    MODEL_SAVE_PATH: str = str(BASE_DIR / "models")
    BEST_MODEL_NAME: str = "best_model.joblib"
    PREPROCESSOR_NAME: str = "preprocessor.joblib"
    MODEL_RETRAIN_THRESHOLD: float = 0.85

    # ── Data ─────────────────────────────────────────────
    DATA_PATH: str = str(BASE_DIR / "data")
    RAW_DATA_FILE: str = "aircraft_fuel_data.csv"
    PROCESSED_DATA_FILE: str = "processed_data.parquet"
    SYNTHETIC_SAMPLES: int = 10000

    # ── Rate Limiting ─────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20

    # ── Admin ─────────────────────────────────────────────
    ADMIN_USERNAME: str = "admin"
    ADMIN_EMAIL: str = "admin@aviationai.com"
    ADMIN_PASSWORD: str = "ChangeMe@2024!"

    # ── Paths (computed) ─────────────────────────────────
    @property
    def models_dir(self) -> Path:
        p = Path(self.MODEL_SAVE_PATH)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        p = Path(self.DATA_PATH)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def experiments_dir(self) -> Path:
        p = Path(self.MLFLOW_TRACKING_URI).parent
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


# Singleton
settings = Settings()
