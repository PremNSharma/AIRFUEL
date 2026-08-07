"""
Aircraft Fuel Consumption Model
FastAPI Application — Main Entry Point
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from database.database import init_db
from backend.api import router as api_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("AircraftFuelAPI")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info("🚀 Aircraft Fuel Consumption API starting...")

    # Initialize database
    try:
        await init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    # Seed default admin user
    try:
        from backend.services.auth_service import seed_admin
        await seed_admin()
        logger.info("✓ Admin user seeded")
    except Exception as e:
        logger.warning(f"Admin seed skipped: {e}")

    # Load ML model
    try:
        from backend.services.ml_service import MLService
        ml = MLService()
        await ml.load_model()
        app.state.ml_service = ml
        logger.info(f"✓ ML Model loaded: {ml.model_name}")
    except Exception as e:
        logger.warning(f"ML model not loaded (train first): {e}")
        app.state.ml_service = None

    logger.info(f"✓ API ready at http://{settings.API_HOST}:{settings.API_PORT}")
    yield

    logger.info("👋 API shutting down...")


# ── Application ───────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="""
        **Aircraft Fuel Consumption Model API**

        An enterprise-grade ML-powered REST API for:
        - 🛩️ Fuel consumption prediction
        - 📊 Flight analytics and efficiency analysis
        - 🌿 CO₂ emission estimation
        - 🧠 Explainable AI (SHAP/LIME)
        - 📈 MLOps and experiment tracking

        **Authentication**: JWT Bearer tokens (use `/api/v1/auth/login`)
        """,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    # ── Root Endpoints ───────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "operational",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", tags=["Health"])
    async def health_check(request: Request):
        ml_status = "loaded" if getattr(request.app.state, "ml_service", None) else "not_loaded"
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
            "ml_model": ml_status,
            "database": "connected",
        }

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
