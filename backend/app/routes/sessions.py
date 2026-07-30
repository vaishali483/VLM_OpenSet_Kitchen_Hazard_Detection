"""Cooking-session API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.core.settings import (
    ALLOWED_IMAGE_TYPES,
    MAX_IMAGE_SIZE_BYTES,
)
from app.routes.live_analysis import (
    _build_detection_items,
    _build_hazard_items,
)
from app.schemas.hazard import (
    HazardAnalysisResponse,
    ImageReceipt,
    ProcessingTimes,
)
from app.schemas.session import (
    CookingSessionRecord,
    SessionStartResponse,
    SessionStateResponse,
    SessionSummaryResponse,
)
from app.services.live_analysis_service import (
    GeminiServiceUnavailableError,
    analyse_live_image,
)
from app.services.session_service import (
    session_service,
)


router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["Cooking sessions"],
)


@router.post(
    "/start",
    response_model=SessionStartResponse,
)
async def start_session() -> SessionStartResponse:
    return session_service.start_session()


@router.post(
    "/{session_id}/pause",
    response_model=SessionStateResponse,
)
async def pause_session(
    session_id: str,
) -> SessionStateResponse:
    try:
        return session_service.pause_session(
            session_id
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error


@router.post(
    "/{session_id}/resume",
    response_model=SessionStateResponse,
)
async def resume_session(
    session_id: str,
) -> SessionStateResponse:
    try:
        return session_service.resume_session(
            session_id
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error


@router.post(
    "/{session_id}/stop",
    response_model=SessionSummaryResponse,
)
async def stop_session(
    session_id: str,
) -> SessionSummaryResponse:
    try:
        return session_service.stop_session(
            session_id
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.get(
    "/{session_id}",
    response_model=CookingSessionRecord,
)
async def get_session(
    session_id: str,
) -> CookingSessionRecord:
    try:
        return session_service.get_session(
            session_id
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.post(
    "/{session_id}/analyse-image",
    response_model=HazardAnalysisResponse,
)
async def analyse_session_image(
    session_id: str,
    image: Annotated[
        UploadFile,
        File(
            description=(
                "Sampled AR camera image"
            )
        ),
    ],
    capture_id: Annotated[
        str,
        Form(
            description=(
                "Unique Unity capture identifier"
            )
        ),
    ],
) -> HazardAnalysisResponse:
    try:
        session_service.require_active(
            session_id
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Only JPEG and PNG images are supported."
            ),
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The uploaded image exceeds 5 MB.",
        )

    try:
        with Image.open(
            BytesIO(image_bytes)
        ) as decoded_image:
            decoded_image.verify()

        with Image.open(
            BytesIO(image_bytes)
        ) as decoded_image:
            width, height = decoded_image.size

    except (
        UnidentifiedImageError,
        OSError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not a valid image."
            ),
        ) from error

    # Text only. This dictionary contains no images and no image paths.
    session_context = (
        session_service.build_text_context(
            session_id
        )
    )

    try:
        result = await run_in_threadpool(
            analyse_live_image,
            image_bytes=image_bytes,
            mime_type=(
                image.content_type
                or "image/jpeg"
            ),
            session_context=session_context,
        )

    except GeminiServiceUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is temporarily busy. "
                "Please retry after 15 seconds."
            ),
            headers={
                "Retry-After": "15",
            },
        ) from error

    except Exception as error:
        print(
            "Session analysis failed:",
            type(error).__name__,
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The session hazard analysis failed."
            ),
        ) from error

    generated = result.analysis

    hazards = _build_hazard_items(
        capture_id=capture_id,
        generated_hazards=generated.hazards,
        detections=result.detections,
    )

    detections = _build_detection_items(
        result.detections
    )

    response = HazardAnalysisResponse(
        event_type=(
            "hazard_analysis_completed"
        ),
        mode="session_yolo_vlm",
        capture_id=capture_id,
        completed_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
        image_received=ImageReceipt(
            filename=(
                image.filename
                or "ar_camera_frame.jpg"
            ),
            content_type=(
                image.content_type
                or "image/jpeg"
            ),
            size_bytes=len(image_bytes),
            width=width,
            height=height,
        ),
        hazards_present=bool(hazards),
        overall_safety_summary=(
            generated.overall_safety_summary
        ),
        hazards=hazards,
        uncertainties=generated.uncertainties,
        scene_summary=generated.scene_summary,
        current_activity=(
            generated.current_activity
        ),
        action_confidence=(
            generated.action_confidence
        ),
        scene_observations=(
            generated.scene_observations
        ),
        model_used=result.model_used,
        detections=detections,
        processing_times=ProcessingTimes(
            yolo_ms=round(result.yolo_ms, 2),
            gemini_ms=round(result.gemini_ms, 2),
            total_ms=round(result.total_ms, 2),
        ),
        session_id=session_id,
    )

    frame_index = session_service.record_analysis(
        session_id=session_id,
        capture_id=capture_id,
        image_bytes=image_bytes,
        content_type=(
            image.content_type
            or "image/jpeg"
        ),
        response=response,
    )

    response.frame_index = frame_index

    return response
