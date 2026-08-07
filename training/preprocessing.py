"""
Aircraft Fuel Consumption Model
Data Preprocessing Pipeline

Handles missing values, outliers, encoding, scaling, and feature engineering.
Returns a serializable sklearn Pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OrdinalEncoder, OneHotEncoder, LabelEncoder,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Feature Groups
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "flight_distance_km", "flight_duration_hrs", "altitude_ft",
    "cruising_speed_kmh", "takeoff_weight_kg", "payload_weight_kg",
    "passenger_count", "cargo_weight_kg", "oat_celsius",
    "wind_speed_kmh", "wind_direction_deg", "humidity_pct",
    "pressure_hpa", "engine_thrust_kn", "taxi_time_min",
    "climb_time_min", "cruise_time_min", "descent_time_min",
]

CATEGORICAL_FEATURES = [
    "aircraft_type", "engine_type", "fuel_type", "weather_condition",
]

TARGET = "fuel_consumption_kg"

DROP_COLS = [
    "flight_id", "flight_date", "departure_airport", "departure_city",
    "departure_country", "arrival_airport", "arrival_city",
    "arrival_country", "co2_emission_kg", "fuel_cost_usd",
]


# ─────────────────────────────────────────────
# Custom Transformers
# ─────────────────────────────────────────────
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips outliers using the IQR method."""

    def __init__(self, factor: float = 3.0):
        self.factor = factor
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "OutlierClipper":
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = np.array(X, dtype=float)
        for i in range(X.shape[1]):
            X[:, i] = np.clip(X[:, i], self.lower_[i], self.upper_[i])
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates domain-specific aviation features."""

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()

        # ── Efficiency ratios ────────────────────────
        df["fuel_per_km_est"] = (
            df["flight_duration_hrs"] * 2500 / df["flight_distance_km"].replace(0, np.nan)
        ).fillna(0)

        df["payload_ratio"] = (
            df["payload_weight_kg"] / df["takeoff_weight_kg"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)

        df["passenger_load_factor"] = (
            df["passenger_count"] / 200
        ).clip(0, 1)

        # ── Phase time ratios ────────────────────────
        total_time = (
            df["taxi_time_min"] + df["climb_time_min"] +
            df["cruise_time_min"] + df["descent_time_min"]
        ).replace(0, np.nan)

        df["cruise_ratio"] = (df["cruise_time_min"] / total_time).fillna(0)
        df["ground_ratio"] = (df["taxi_time_min"] / total_time).fillna(0)

        # ── Speed efficiency ─────────────────────────
        df["speed_altitude_ratio"] = (
            df["cruising_speed_kmh"] / df["altitude_ft"].replace(0, np.nan) * 1000
        ).fillna(0)

        # ── Wind component (headwind positive = adverse) ─
        df["headwind_component"] = (
            df["wind_speed_kmh"] * np.cos(np.radians(df["wind_direction_deg"]))
        )

        # ── Temperature deviation from ISA ─────────────
        df["isa_temp_deviation"] = (
            df["oat_celsius"] - (15 - 1.98 * df["altitude_ft"] / 1000)
        )

        # ── Weight per seat ──────────────────────────
        df["weight_per_passenger"] = (
            df["takeoff_weight_kg"] / df["passenger_count"].replace(0, np.nan)
        ).fillna(0)

        # ── Distance bucket ──────────────────────────
        df["flight_range_category"] = pd.cut(
            df["flight_distance_km"],
            bins=[0, 1000, 3000, 6000, 999999],
            labels=[0, 1, 2, 3],  # Short, Medium, Long, Ultra-long
        ).astype(float).fillna(0)

        return df


# ─────────────────────────────────────────────
# Preprocessing Functions
# ─────────────────────────────────────────────
def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load data supporting CSV, Excel, Parquet, JSON."""
    path = Path(data_path)
    suffix = path.suffix.lower()
    loaders = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
    }
    if suffix not in loaders:
        raise ValueError(f"Unsupported file format: {suffix}")
    df = loaders[suffix](path)
    logger.info(f"Loaded {len(df)} records from {path.name}")
    return df


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Run data quality checks and return cleaned df + report."""
    report = {
        "initial_rows": len(df),
        "initial_cols": len(df.columns),
        "duplicates_removed": 0,
        "missing_pct": {},
        "outliers_detected": {},
    }

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    # Missing value analysis
    for col in df.columns:
        miss_pct = df[col].isna().mean() * 100
        if miss_pct > 0:
            report["missing_pct"][col] = round(miss_pct, 2)

    # Validate target
    if TARGET in df.columns:
        invalid_target = (df[TARGET] <= 0) | (df[TARGET].isna())
        df = df[~invalid_target]
        logger.info(f"Removed {invalid_target.sum()} rows with invalid target")

    report["final_rows"] = len(df)
    logger.info(f"Data validation: {report['duplicates_removed']} duplicates removed, "
                f"{len(report['missing_pct'])} cols with missings")
    return df, report


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, drop irrelevant columns."""
    # Feature engineering first
    engineer = FeatureEngineer()
    df = engineer.transform(df)

    # Drop irrelevant columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    scaler: str = "robust"
) -> Pipeline:
    """Build a full preprocessing pipeline using sklearn."""
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }

    numeric_pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("outlier_clip", OutlierClipper(factor=3.0)),
        ("scaler", scalers.get(scaler, RobustScaler())),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocessor", preprocessor)])


def get_feature_names(
    preprocessor_pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    """Extract feature names after transformation."""
    return numeric_features + categorical_features


def preprocess_for_training(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Full preprocessing pipeline for training."""
    from sklearn.model_selection import train_test_split

    logger.info("Starting preprocessing pipeline...")

    # 1. Validate
    df, report = validate_data(df)

    # 2. Feature engineering + split X/y
    X, y = prepare_features(df)

    # 3. Identify feature groups (after engineering)
    numeric_cols = [c for c in X.columns if X[c].dtype in [np.float64, np.int64, float, int]
                    and c not in CATEGORICAL_FEATURES]
    categorical_cols = [c for c in X.columns if c in CATEGORICAL_FEATURES or X[c].dtype == object]

    # 4. Train/test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 5. Train/val split
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_adjusted,
        random_state=random_state
    )

    # 6. Build and fit preprocessor
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)

    logger.info(f"Preprocessing complete:")
    logger.info(f"  Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")
    logger.info(f"  Features: {len(feature_names)}")

    return {
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train.values,
        "y_val": y_val.values,
        "y_test": y_test.values,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "validation_report": report,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
    }


def save_preprocessor(preprocessor: Pipeline, save_path: str) -> str:
    """Serialize preprocessor pipeline."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info(f"Preprocessor saved to {path}")
    return str(path)


def load_preprocessor(load_path: str) -> Pipeline:
    """Load serialized preprocessor."""
    return joblib.load(load_path)
