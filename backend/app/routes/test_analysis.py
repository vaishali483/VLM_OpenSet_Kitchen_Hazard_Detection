"""Temporary image-upload endpoint for Unity integration testing."""

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

from app.schemas.hazard import (
    HazardAnalysisResponse,
    HazardItem,
    HazardVisualisation,
    ImageReceipt,
)


router = APIRouter(
    prefix="/api/v1/test",
    tags=["Unity integration test"],
)


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024


@router.post(
    "/analyse-image",
    response_model=HazardAnalysisResponse,
)
async def analyse_test_image(
    image: Annotated[
        UploadFile,
        File(description="Kitchen image uploaded by Unity"),
    ],
    capture_id: Annotated[
        str,
        Form(description="Unique Unity capture identifier"),
    ],
) -> HazardAnalysisResponse:
    """
    Receive and validate one image.

    This endpoint does not run YOLO or Gemini yet.
    It returns a fixed hazard result to test the full Unity
    upload, JSON parsing and warning-display workflow.
    """

    if image.content_type not in ALLOWED_CONTENT_TYPES:
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
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            uploaded_image.verify()

        # Reopen after verify because verify consumes image state.
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            width, height = uploaded_image.size

    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from error

    except OSError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image could not be decoded.",
        ) from error

    return HazardAnalysisResponse(
        event_type="hazard_analysis_completed",
        mode="mock",
        capture_id=capture_id,
        completed_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
        image_received=ImageReceipt(
            filename=image.filename or "unknown.jpg",
            content_type=image.content_type,
            size_bytes=len(image_bytes),
            width=width,
            height=height,
        ),
        hazards_present=True,
        overall_safety_summary=(
            "One test hazard was returned successfully."
        ),
        hazards=[
            HazardItem(
                hazard_id="MOCK_H001",
                description=(
                    "The pot handle protrudes into the "
                    "active workspace."
                ),
                explanation=(
                    "The handle could be accidentally bumped "
                    "and cause the pot to tip."
                ),
                severity="medium",
                confidence=0.92,
                recommended_action=(
                    "Turn the pot handle inward."
                ),
                visualisation=HazardVisualisation(
                    mode="warning_label"
                ),
            )
        ],
        uncertainties=[
            (
                "This is a fixed mock result. "
                "The image has not yet been analysed by the VLM."
            )
        ],
    )