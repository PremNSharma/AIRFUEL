"""
Aircraft Fuel Consumption Model
Explainable AI Module

Implements SHAP, LIME, and Permutation Importance for model interpretability.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SHAP Explanations
# ─────────────────────────────────────────────
class SHAPExplainer:
    """SHAP-based global and local explanations."""

    def __init__(self, model: Any, X_train: np.ndarray, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self._setup_explainer(X_train)

    def _setup_explainer(self, X_train: np.ndarray) -> None:
        """Choose appropriate SHAP explainer based on model type."""
        try:
            import shap
            model_type = type(self.model).__name__

            tree_models = [
                "RandomForestRegressor", "GradientBoostingRegressor",
                "ExtraTreesRegressor", "DecisionTreeRegressor",
                "XGBRegressor", "LGBMRegressor", "CatBoostRegressor",
                "AdaBoostRegressor",
            ]

            if model_type in tree_models:
                # Sample for performance
                sample_size = min(500, X_train.shape[0])
                idx = np.random.choice(X_train.shape[0], sample_size, replace=False)
                X_sample = X_train[idx]
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X_sample)
                self.X_background = X_sample
            else:
                # Use KernelExplainer for linear/SVR/KNN
                sample_size = min(100, X_train.shape[0])
                idx = np.random.choice(X_train.shape[0], sample_size, replace=False)
                X_background = shap.kmeans(X_train, 50)
                self.explainer = shap.KernelExplainer(self.model.predict, X_background)
                X_sample = X_train[idx]
                self.shap_values = self.explainer.shap_values(X_sample)
                self.X_background = X_sample

            logger.info(f"SHAP explainer initialized: {type(self.explainer).__name__}")

        except Exception as e:
            logger.error(f"SHAP setup failed: {e}")
            self.explainer = None

    def get_global_importance(self) -> pd.DataFrame:
        """Compute mean absolute SHAP values for global feature importance."""
        if self.shap_values is None:
            return pd.DataFrame()

        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs_shap)],
            "shap_importance": mean_abs_shap,
        }).sort_values("shap_importance", ascending=False)
        return df

    def explain_instance(self, x_instance: np.ndarray) -> dict:
        """Generate local explanation for a single prediction."""
        if self.explainer is None:
            return {}

        try:
            shap_vals = self.explainer.shap_values(x_instance.reshape(1, -1))
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]

            contributions = dict(zip(
                self.feature_names[:len(shap_vals[0])],
                shap_vals[0].tolist()
            ))
            return {
                "shap_values": contributions,
                "base_value": float(self.explainer.expected_value)
                if hasattr(self.explainer, "expected_value")
                else 0.0,
            }
        except Exception as e:
            logger.error(f"Local explanation failed: {e}")
            return {}

    def save_summary_plot(self, save_path: str) -> Optional[str]:
        """Save SHAP summary plot."""
        if self.shap_values is None:
            return None

        try:
            import shap
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                self.X_background,
                feature_names=self.feature_names[:self.shap_values.shape[1]],
                show=False, max_display=20,
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        except Exception as e:
            logger.error(f"SHAP plot failed: {e}")
            return None


# ─────────────────────────────────────────────
# LIME Explanations
# ─────────────────────────────────────────────
class LIMEExplainer:
    """LIME-based local explanations for individual predictions."""

    def __init__(self, model: Any, X_train: np.ndarray, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._setup_explainer(X_train)

    def _setup_explainer(self, X_train: np.ndarray) -> None:
        try:
            from lime import lime_tabular
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                feature_names=self.feature_names[:X_train.shape[1]],
                mode="regression",
                discretize_continuous=True,
                random_state=42,
            )
            logger.info("LIME explainer initialized")
        except Exception as e:
            logger.error(f"LIME setup failed: {e}")

    def explain_instance(
        self,
        x_instance: np.ndarray,
        num_features: int = 10
    ) -> dict:
        """Generate LIME explanation for a single instance."""
        if self.explainer is None:
            return {}

        try:
            explanation = self.explainer.explain_instance(
                x_instance.flatten(),
                self.model.predict,
                num_features=num_features,
            )
            local_exp = explanation.as_list()
            return {
                "lime_contributions": [
                    {"feature": feat, "contribution": contrib}
                    for feat, contrib in local_exp
                ],
                "predicted_value": float(explanation.predicted_value),
                "intercept": float(explanation.intercept[1]) if isinstance(explanation.intercept, dict) else 0.0,
            }
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {}


# ─────────────────────────────────────────────
# Permutation Importance
# ─────────────────────────────────────────────
def compute_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Compute permutation feature importance."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score

    logger.info("Computing permutation importance...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": feature_names[:len(result.importances_mean)],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return df


# ─────────────────────────────────────────────
# EDA Visualizations
# ─────────────────────────────────────────────
class EDAVisualizer:
    """Generate EDA plots and save them to disk."""

    def __init__(self, output_dir: str = "assets/eda"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def correlation_heatmap(self, df: pd.DataFrame, target: str = "fuel_consumption_kg") -> str:
        """Generate correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number]).head(5000)
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=False, fmt=".2f",
            cmap="RdYlGn", center=0, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        path = str(self.output_dir / "correlation_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def target_distribution(self, df: pd.DataFrame, target: str = "fuel_consumption_kg") -> str:
        """Plot target variable distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(df[target].dropna(), bins=50, color="#3B82F6", alpha=0.8, edgecolor="white")
        axes[0].set_title(f"{target} Distribution", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Fuel Consumption (kg)", fontsize=10)
        axes[0].set_ylabel("Count", fontsize=10)
        axes[0].axvline(df[target].mean(), color="red", linestyle="--", label=f"Mean: {df[target].mean():.0f}")
        axes[0].legend()

        # Box plot by aircraft type
        if "aircraft_type" in df.columns:
            aircraft_order = df.groupby("aircraft_type")[target].median().sort_values().index
            df_plot = df[["aircraft_type", target]].dropna()
            axes[1].boxplot(
                [df_plot[df_plot["aircraft_type"] == a][target].values
                 for a in aircraft_order],
                labels=[a.split(" ")[1] for a in aircraft_order],
                patch_artist=True,
            )
            axes[1].set_title("Fuel Consumption by Aircraft Type", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Aircraft", fontsize=10)
            axes[1].set_ylabel("Fuel Consumption (kg)", fontsize=10)
            axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        path = str(self.output_dir / "target_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def feature_importance_plot(
        self, feature_names: List[str], importances: np.ndarray, title: str = "Feature Importance"
    ) -> str:
        """Plot feature importances."""
        # Take top 20
        n = min(20, len(feature_names))
        sorted_idx = np.argsort(importances)[-n:]
        features = [feature_names[i] for i in sorted_idx]
        imps = importances[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, n))
        bars = ax.barh(range(n), imps, color=colors)
        ax.set_yticks(range(n))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        path = str(self.output_dir / "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Plot residual analysis."""
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.3, color="#3B82F6", s=5)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted Values (kg)")
        axes[0].set_ylabel("Residuals (kg)")
        axes[0].set_title("Residuals vs Predicted")

        # Residual Distribution
        axes[1].hist(residuals, bins=50, color="#10B981", alpha=0.8, edgecolor="white")
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Residuals (kg)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Residual Distribution")

        # Actual vs Predicted
        max_val = max(y_true.max(), y_pred.max())
        axes[2].scatter(y_true, y_pred, alpha=0.3, color="#F59E0B", s=5)
        axes[2].plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Prediction")
        axes[2].set_xlabel("Actual Values (kg)")
        axes[2].set_ylabel("Predicted Values (kg)")
        axes[2].set_title("Actual vs Predicted")
        axes[2].legend()

        plt.tight_layout()
        path = str(self.output_dir / "residual_analysis.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path


# ─────────────────────────────────────────────
# Combined XAI Manager
# ─────────────────────────────────────────────
class XAIManager:
    """Centralized explainable AI manager."""

    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        output_dir: str = "assets/xai",
    ):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shap_explainer = SHAPExplainer(model, X_train, feature_names)
        self.lime_explainer = LIMEExplainer(model, X_train, feature_names)

        # Permutation importance
        logger.info("Computing permutation importance...")
        self.perm_importance = compute_permutation_importance(
            model, X_test, y_test, feature_names, n_repeats=5
        )

    def get_global_explanations(self) -> dict:
        """Return all global explanations."""
        shap_global = self.shap_explainer.get_global_importance()

        model_importance = {}
        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
            model_importance = dict(zip(
                self.feature_names[:len(imps)], imps.tolist()
            ))
        elif hasattr(self.model, "coef_"):
            coef = np.abs(self.model.coef_)
            model_importance = dict(zip(
                self.feature_names[:len(coef)], coef.tolist()
            ))

        return {
            "shap_global": shap_global.to_dict(orient="records") if not shap_global.empty else [],
            "model_feature_importance": model_importance,
            "permutation_importance": self.perm_importance.to_dict(orient="records"),
        }

    def explain_prediction(self, x_instance: np.ndarray) -> dict:
        """Generate full local explanation for one prediction."""
        shap_local = self.shap_explainer.explain_instance(x_instance)
        lime_local = self.lime_explainer.explain_instance(x_instance)
        prediction = float(self.model.predict(x_instance.reshape(1, -1))[0])

        return {
            "prediction": prediction,
            "shap_explanation": shap_local,
            "lime_explanation": lime_local,
        }
