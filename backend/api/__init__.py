"""
API Router — combines all sub-routers
"""

from fastapi import APIRouter

from backend.api.auth import router as auth_router
from backend.api.predictions import router as predictions_router
from backend.api.analytics import router as analytics_router
from backend.api.models import router as models_router
from backend.api.flights import router as flights_router
from backend.api.admin import router as admin_router

router = APIRouter()

router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
router.include_router(predictions_router, prefix="/predictions", tags=["Predictions"])
router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
router.include_router(models_router, prefix="/models", tags=["ML Models"])
router.include_router(flights_router, prefix="/flights", tags=["Flights"])
router.include_router(admin_router, prefix="/admin", tags=["Admin"])
