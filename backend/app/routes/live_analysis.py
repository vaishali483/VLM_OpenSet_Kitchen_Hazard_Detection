"""YOLO-grounded live Gemini analysis endpoint."""

from __future__ import annotations

import re
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
    LOCALISATION_BLOCKED_BOX_CLASSES,
    LOCALISATION_MAX_BOX_AREA,
    LOCALISATION_MIN_BOX_AREA,
    LOCALISATION_MIN_GEMINI_CONFIDENCE,
    LOCALISATION_MIN_YOLO_CONFIDENCE,
    MAX_IMAGE_SIZE_BYTES,
)
from app.schemas.hazard import (
    DetectionItem,
    HazardAnalysisResponse,
    HazardItem,
    HazardVisualisation,
    ImageReceipt,
    NormalizedBoundingBox,
    ProcessingTimes,
)
from app.schemas.live_analysis import (
    GeneratedHazard,
    YoloDetection,
)
from app.services.live_analysis_service import (
    analyse_live_image,
)

from app.services.live_analysis_service import (
    GeminiServiceUnavailableError,
    analyse_live_image,
)


router = APIRouter(
    prefix="/api/v1/live",
    tags=["Live hazard analysis"],
)


CLASS_TEXT_TERMS: dict[str, tuple[str, ...]] = {
    "knife": (
        "knife",
        "blade",
        "kitchen knife",
        "cutting knife",
    ),

    "scissors": (
        "scissors",
        "shears",
    ),

    "bottle": (
        "bottle",
        "container",
    ),

    "wine glass": (
        "wine glass",
        "glass",
        "drinking glass",
    ),

    "cup": (
        "cup",
        "mug",
    ),

    "fork": (
        "fork",
    ),

    "spoon": (
        "spoon",
        "utensil",
    ),

    "bowl": (
        "bowl",
        "dish",
    ),

    "microwave": (
        "microwave",
        "microwave oven",
    ),

    "oven": (
        "oven",
        "cooker",
        "stove",
    ),

    "toaster": (
        "toaster",
    ),

    "sink": (
        "sink",
        "basin",
    ),

    "refrigerator": (
        "refrigerator",
        "fridge",
    ),
}


def _normalise_text(value: str | None) -> str:
    """Normalise text for conservative class-name comparison."""

    if not value:
        return ""

    lowered = value.strip().lower()

    return re.sub(
        r"\s+",
        " ",
        lowered,
    )


def _text_contains_term(
    text: str,
    term: str,
) -> bool:
    """Check a complete word or phrase instead of a substring."""

    pattern = (
        r"\b"
        + re.escape(term)
        + r"\b"
    )

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def _hazard_mentions_class(
    hazard: GeneratedHazard,
    class_name: str,
) -> bool:
    """
    Require the displayed hazard wording to identify the
    same object class as the proposed detection.
    """

    combined_text = _normalise_text(
        " ".join(
            [
                hazard.description,
                hazard.explanation,
                hazard.recommended_action,
            ]
        )
    )

    normalised_class = _normalise_text(
        class_name
    )

    candidate_terms = CLASS_TEXT_TERMS.get(
        normalised_class,
        (normalised_class,),
    )

    return any(
        _text_contains_term(
            combined_text,
            term,
        )
        for term in candidate_terms
    )


def _warning_only(
    reason: str,
) -> HazardVisualisation:
    """Create a text-only localisation result."""

    return HazardVisualisation(
        mode="warning_label",
        bbox_normalized=None,
        source_detection_id=None,
        source_class_name=None,
        localisation_confidence=0.0,
        validation_reason=reason,
    )

def _validate_approximate_region(
    hazard: GeneratedHazard,
) -> HazardVisualisation:
    """
    Validate a Gemini-estimated approximate hazard region.

    This region is less reliable than a detector-confirmed box,
    so it is returned with a different visualisation mode.
    """

    if hazard.localisation_kind != "approximate_region":
        return _warning_only(
            "not_an_approximate_region"
        )

    bbox = hazard.estimated_bbox_normalized

    if bbox is None:
        return _warning_only(
            "missing_estimated_bbox"
        )

    if hazard.localisation_confidence < 0.65:
        return _warning_only(
            "approximate_localisation_confidence_below_threshold"
        )

    box_width = max(
        0.0,
        bbox.x_max - bbox.x_min,
    )

    box_height = max(
        0.0,
        bbox.y_max - bbox.y_min,
    )

    box_area = box_width * box_height

    if box_width <= 0 or box_height <= 0:
        return _warning_only(
            "invalid_estimated_bbox"
        )

    if box_area < 0.001:
        return _warning_only(
            "estimated_box_too_small"
        )

    if box_area > 0.65:
        return _warning_only(
            "estimated_box_too_large"
        )

    return HazardVisualisation(
        mode="approximate_box",

        bbox_normalized=NormalizedBoundingBox(
            x_min=bbox.x_min,
            y_min=bbox.y_min,
            x_max=bbox.x_max,
            y_max=bbox.y_max,
        ),

        source_detection_id=None,
        source_class_name=None,

        localisation_confidence=round(
            hazard.localisation_confidence,
            4,
        ),

        validation_reason=(
            "accepted_gemini_approximate_region"
        ),
    )

def _validate_localisation(
    *,
    hazard: GeneratedHazard,
    detection_lookup: dict[str, YoloDetection],
) -> HazardVisualisation:
    if hazard.localisation_kind == "approximate_region":
        return _validate_approximate_region(
            hazard
        )

    if hazard.localisation_kind != "exact_object":
        return _warning_only(
            "gemini_did_not_localise_hazard"
        )

    detection_id = (
        hazard.target_detection_id
        or ""
    ).strip()

    if not detection_id:
        return _warning_only(
            "missing_detection_id"
        )

    detection = detection_lookup.get(
        detection_id
    )

    if detection is None:
        return _warning_only(
            "unknown_detection_id"
        )

    detection_class = _normalise_text(
        detection.class_name
    )

    target_class = _normalise_text(
        hazard.target_class_name
    )

    if not target_class:
        return _warning_only(
            "missing_target_class"
        )

    if target_class != detection_class:
        return _warning_only(
            "target_class_does_not_match_detection"
        )

    if (
        detection_class
        in LOCALISATION_BLOCKED_BOX_CLASSES
    ):
        return _warning_only(
            "broad_detection_class_blocked"
        )

    if (
        detection.confidence
        < LOCALISATION_MIN_YOLO_CONFIDENCE
    ):
        return _warning_only(
            "yolo_confidence_below_threshold"
        )

    if (
        hazard.localisation_confidence
        < LOCALISATION_MIN_GEMINI_CONFIDENCE
    ):
        return _warning_only(
            "gemini_localisation_confidence_below_threshold"
        )

    if not _hazard_mentions_class(
        hazard,
        detection.class_name,
    ):
        return _warning_only(
            "hazard_text_does_not_name_detected_object"
        )

    bbox = detection.bbox_normalized

    box_width = max(
        0.0,
        bbox.x_max - bbox.x_min,
    )

    box_height = max(
        0.0,
        bbox.y_max - bbox.y_min,
    )

    box_area = box_width * box_height

    if box_area < LOCALISATION_MIN_BOX_AREA:
        return _warning_only(
            "box_too_small"
        )

    if box_area > LOCALISATION_MAX_BOX_AREA:
        return _warning_only(
            "box_too_large"
        )

    return HazardVisualisation(
        mode="bounding_box",

        bbox_normalized=NormalizedBoundingBox(
            x_min=bbox.x_min,
            y_min=bbox.y_min,
            x_max=bbox.x_max,
            y_max=bbox.y_max,
        ),

        source_detection_id=(
            detection.detection_id
        ),

        source_class_name=(
            detection.class_name
        ),

        localisation_confidence=round(
            min(
                detection.confidence,
                hazard.localisation_confidence,
            ),
            4,
        ),

        validation_reason=(
            "accepted_exact_object_match"
        ),
    )


@router.post(
    "/analyse-image",
    response_model=HazardAnalysisResponse,
)
async def analyse_image(
    image: Annotated[
        UploadFile,
        File(
            description=(
                "AR camera image uploaded by Unity"
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
    """Analyse one AR camera image using YOLO and Gemini."""

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

    try:
        result = await run_in_threadpool(
            analyse_live_image,
            image_bytes=image_bytes,
            mime_type=image.content_type,
        )

    except GeminiServiceUnavailableError as error:
        print(
            "All Gemini models are temporarily unavailable:",
            error,
        )

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
            "Live analysis failed:",
            type(error).__name__,
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The live hazard analysis failed because "
                "of an internal processing error."
            ),
        ) from error

    detection_lookup = {
        detection.detection_id: detection
        for detection in result.detections
    }

    generated = result.analysis

    hazards: list[HazardItem] = []

    for index, generated_hazard in enumerate(
        generated.hazards,
        start=1,
    ):
        visualisation = _validate_localisation(
            hazard=generated_hazard,
            detection_lookup=detection_lookup,
        )

        print(
            f"Hazard {index} localisation: "
            f"{visualisation.validation_reason}"
        )

        hazards.append(
            HazardItem(
                hazard_id=(
                    f"{capture_id}_H{index:03d}"
                ),

                description=(
                    generated_hazard.description
                ),

                explanation=(
                    generated_hazard.explanation
                ),

                severity=(
                    generated_hazard.severity
                ),

                confidence=(
                    generated_hazard.confidence
                ),

                recommended_action=(
                    generated_hazard
                    .recommended_action
                ),

                visualisation=visualisation,
            )
        )

    detections = [
        DetectionItem(
            detection_id=detection.detection_id,
            class_name=detection.class_name,
            confidence=detection.confidence,

            bbox_normalized=NormalizedBoundingBox(
                x_min=(
                    detection
                    .bbox_normalized
                    .x_min
                ),

                y_min=(
                    detection
                    .bbox_normalized
                    .y_min
                ),

                x_max=(
                    detection
                    .bbox_normalized
                    .x_max
                ),

                y_max=(
                    detection
                    .bbox_normalized
                    .y_max
                ),
            ),
        )
        for detection in result.detections
    ]

    return HazardAnalysisResponse(
        event_type=(
            "hazard_analysis_completed"
        ),

        mode="live_yolo_vlm",
        model_used=result.model_used,

        capture_id=capture_id,

        completed_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),

        image_received=ImageReceipt(
            filename=(
                image.filename
                or "ar_camera_frame.jpg"
            ),

            content_type=image.content_type,
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

        detections=detections,

        processing_times=ProcessingTimes(
            yolo_ms=round(
                result.yolo_ms,
                2,
            ),

            gemini_ms=round(
                result.gemini_ms,
                2,
            ),

            total_ms=round(
                result.total_ms,
                2,
            ),
        ),
    )