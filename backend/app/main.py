"""FastAPI entry point for Kitchen Safety Assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from app.routes.live_analysis import (
    router as live_analysis_router,
)

from app.routes.sessions import (
    router as sessions_router,
)

from app.routes.test_analysis import (
    router as test_analysis_router,
)

from app.services.yolo_service import (
    load_yolo_model,
    unload_yolo_model,
    warm_up_yolo,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """Load shared machine-learning resources."""

    load_yolo_model()
    warm_up_yolo()

    yield

    unload_yolo_model()


app = FastAPI(
    title="Kitchen Safety Assistant API",
    description=(
        "Backend API for the Unity AR "
        "kitchen-hazard assistant."
    ),
    version="0.5.0",
    lifespan=lifespan,
)


app.include_router(
    test_analysis_router
)

app.include_router(
    live_analysis_router
)

app.include_router(
    sessions_router
)


@app.get("/api/v1/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "kitchen-safety-backend",
        "version": "0.5.0",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }