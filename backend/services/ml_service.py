"""
ML Service — Model Loading, Prediction, and Inference
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logger = logging.getLogger(__name__)


class MLService:
    """Service for ML model inference and explanations."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.feature_names: List[str] = []
        self.model_name: str = "Unknown"
        self.model_metadata: dict = {}
        self.xai_results: dict = {}
        self.comparison_table: Optional[pd.DataFrame] = None
        self.learning_curves: dict = {}
        self._loaded = False

    async def load_model(self) -> bool:
        """Load the best trained model and preprocessor."""
        model_dir = Path(settings.MODEL_SAVE_PATH)

        model_path = model_dir / settings.BEST_MODEL_NAME
        preprocessor_path = model_dir / settings.PREPROCESSOR_NAME
        meta_path = model_dir / "model_metadata.json"
        xai_path = model_dir / "xai_results.json"
        comparison_path = model_dir / "model_comparison.csv"
        lc_path = model_dir / "learning_curves.json"

        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Run training pipeline first.")
            return False

        self.model = joblib.load(model_path)
        logger.info(f"Model loaded: {type(self.model).__name__}")

        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded")

        if meta_path.exists():
            with open(meta_path) as f:
                self.model_metadata = json.load(f)
            self.model_name = self.model_metadata.get("model_name", type(self.model).__name__)
            self.feature_names = self.model_metadata.get("feature_names", [])

        if xai_path.exists():
            with open(xai_path) as f:
                self.xai_results = json.load(f)

        if comparison_path.exists():
            self.comparison_table = pd.read_csv(comparison_path)

        if lc_path.exists():
            with open(lc_path) as f:
                self.learning_curves = json.load(f)

        self._loaded = True
        return True

    def _preprocess_input(self, features: Dict) -> np.ndarray:
        """Preprocess a single prediction input."""
        from training.preprocessing import FeatureEngineer, NUMERIC_FEATURES, CATEGORICAL_FEATURES

        df = pd.DataFrame([features])

        # Feature engineering
        engineer = FeatureEngineer()
        df = engineer.transform(df)

        if self.preprocessor is not None:
            return self.preprocessor.transform(df)
        else:
            # Fallback: use numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return df[numeric_cols].values

    def predict(self, features: Dict) -> Dict:
        """Single prediction with fuel consumption, CO₂, and cost."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Run training pipeline first.")

        X = self._preprocess_input(features)
        predicted_fuel_kg = float(self.model.predict(X)[0])
        predicted_fuel_kg = max(predicted_fuel_kg, 50)  # minimum sanity check

        co2_kg = predicted_fuel_kg * 3.16  # Jet-A CO₂ emission factor
        fuel_cost_usd = predicted_fuel_kg * 0.82  # avg jet fuel price $/kg

        # Efficiency score (0-100, higher = more efficient)
        distance = features.get("flight_distance_km", 1000)
        passenger_count = features.get("passenger_count", 150)
        if distance > 0 and passenger_count > 0:
            fuel_per_pax_km = predicted_fuel_kg / (passenger_count * distance)
            # Industry benchmark: ~0.04 kg/pax/km for narrow-body
            efficiency = max(0, min(100, 100 * (1 - (fuel_per_pax_km - 0.03) / 0.05)))
        else:
            efficiency = 50.0

        # Simple confidence interval (±10% for demonstration)
        ci_lower = predicted_fuel_kg * 0.90
        ci_upper = predicted_fuel_kg * 1.10

        return {
            "predicted_fuel_kg": round(predicted_fuel_kg, 2),
            "predicted_co2_kg": round(co2_kg, 2),
            "predicted_fuel_cost_usd": round(fuel_cost_usd, 2),
            "fuel_efficiency_score": round(efficiency, 1),
            "confidence_interval": {
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2),
            },
        }

    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        """Batch predictions."""
        return [self.predict(f) for f in features_list]

    def get_recommendations(self, features: Dict, predicted_fuel_kg: float) -> List[Dict]:
        """Generate fuel optimization recommendations."""
        recommendations = []

        # Payload optimization
        payload_ratio = features.get("payload_weight_kg", 0) / max(features.get("takeoff_weight_kg", 1), 1)
        if payload_ratio > 0.85:
            recommendations.append({
                "category": "Weight Optimization",
                "priority": "high",
                "suggestion": "Reduce payload weight to below 85% of max. Estimated savings: 3-8% fuel.",
                "potential_savings_kg": round(predicted_fuel_kg * 0.05, 1),
            })

        # Speed optimization
        ac_cruise_speeds = {"Boeing 737-800": 842, "Airbus A320neo": 833}
        optimal_speed = 840  # generic
        current_speed = features.get("cruising_speed_kmh", optimal_speed)
        if current_speed > optimal_speed * 1.05:
            recommendations.append({
                "category": "Speed Optimization",
                "priority": "medium",
                "suggestion": f"Reduce cruising speed by {current_speed - optimal_speed:.0f} km/h to optimal Mach. Savings: 2-5%.",
                "potential_savings_kg": round(predicted_fuel_kg * 0.03, 1),
            })

        # Altitude optimization
        alt = features.get("altitude_ft", 35000)
        if alt < 32000:
            recommendations.append({
                "category": "Altitude Optimization",
                "priority": "medium",
                "suggestion": "Increase cruise altitude to FL350-FL390 for better fuel efficiency.",
                "potential_savings_kg": round(predicted_fuel_kg * 0.04, 1),
            })

        # Taxi time
        taxi = features.get("taxi_time_min", 15)
        if taxi > 25:
            recommendations.append({
                "category": "Ground Operations",
                "priority": "low",
                "suggestion": f"Taxi time of {taxi:.0f} min is high. Single-engine taxi can save 15-20% ground fuel.",
                "potential_savings_kg": round(predicted_fuel_kg * 0.02, 1),
            })

        # Weather routing
        wind = features.get("wind_speed_kmh", 0)
        if wind > 60:
            recommendations.append({
                "category": "Weather Routing",
                "priority": "medium",
                "suggestion": "Consider alternate routing to reduce headwind exposure. Optimal routing can save 3-6%.",
                "potential_savings_kg": round(predicted_fuel_kg * 0.04, 1),
            })

        if not recommendations:
            recommendations.append({
                "category": "Operational",
                "priority": "low",
                "suggestion": "Flight parameters are within optimal ranges. Continue monitoring.",
                "potential_savings_kg": 0,
            })

        return recommendations

    def get_model_info(self) -> dict:
        """Return model metadata and performance metrics."""
        return {
            "model_name": self.model_name,
            "model_class": type(self.model).__name__ if self.model else None,
            "is_loaded": self._loaded,
            "metadata": self.model_metadata,
            "xai_available": bool(self.xai_results),
            "comparison_available": self.comparison_table is not None,
        }

    def get_comparison_data(self) -> List[dict]:
        """Return model comparison table as list of dicts."""
        if self.comparison_table is None:
            return []
        return self.comparison_table.to_dict(orient="records")

    def get_feature_importance(self) -> List[dict]:
        """Return model feature importance."""
        if self.model is None:
            return []

        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
            names = self.feature_names[:len(imps)]
            pairs = sorted(zip(names, imps.tolist()), key=lambda x: x[1], reverse=True)
            return [{"feature": n, "importance": round(v, 6)} for n, v in pairs]

        if hasattr(self.model, "coef_"):
            coef = np.abs(self.model.coef_)
            names = self.feature_names[:len(coef)]
            pairs = sorted(zip(names, coef.tolist()), key=lambda x: x[1], reverse=True)
            return [{"feature": n, "importance": round(v, 6)} for n, v in pairs]

        # SHAP global importance fallback
        if self.xai_results and "shap_global" in self.xai_results:
            return self.xai_results["shap_global"]

        return []
