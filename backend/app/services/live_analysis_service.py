"""Optimised YOLO-grounded Gemini live analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.prompt_loader import load_prompt
from app.core.settings import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MEDIA_RESOLUTION,
    GEMINI_MODEL,
    GEMINI_THINKING_LEVEL,
)
from app.schemas.live_analysis import (
    CombinedLiveAnalysis,
    GeneratedHazard,
    YoloDetection,
)
from app.services.yolo_service import detect_objects


SchemaType = TypeVar(
    "SchemaType",
    bound=BaseModel,
)


client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=120_000,
        retry_options=types.HttpRetryOptions(
            attempts=3,
            initial_delay=1.0,
            max_delay=8.0,
            exp_base=2.0,
            jitter=0.5,
            http_status_codes=[
                408,
                429,
                500,
                502,
                503,
                504,
            ],
        ),
    ),
)


MEDIA_RESOLUTION_MAP = {
    "low": (
        types.MediaResolution
        .MEDIA_RESOLUTION_LOW
    ),
    "medium": (
        types.MediaResolution
        .MEDIA_RESOLUTION_MEDIUM
    ),
    "high": (
        types.MediaResolution
        .MEDIA_RESOLUTION_HIGH
    ),
}


LIVE_ANALYSIS_PROMPT = load_prompt(
    "live_hazard_analysis.txt"
)

class GeminiServiceUnavailableError(RuntimeError):
    """Raised when all configured Gemini models are unavailable."""


def _is_transient_gemini_error(
    error: Exception,
) -> bool:
    """
    Return True for temporary Gemini capacity, timeout,
    rate-limit and server-side failures.
    """

    status_code = getattr(
        error,
        "status_code",
        None,
    )

    code = getattr(
        error,
        "code",
        None,
    )

    transient_status_codes = {
        408,
        429,
        500,
        502,
        503,
        504,
    }

    if status_code in transient_status_codes:
        return True

    if code in transient_status_codes:
        return True

    error_text = str(error).upper()

    transient_terms = (
        "408",
        "429",
        "500",
        "502",
        "503",
        "504",
        "UNAVAILABLE",
        "HIGH DEMAND",
        "RESOURCE_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "TIMED OUT",
    )

    return any(
        term in error_text
        for term in transient_terms
    )


@dataclass
class LiveAnalysisResult:
    analysis: CombinedLiveAnalysis
    detections: list[YoloDetection]

    model_used: str

    yolo_ms: float
    gemini_ms: float
    total_ms: float


def _parse_response(
    response: object,
    schema: type[SchemaType],
) -> SchemaType:
    """Parse a Gemini structured-output response."""

    parsed = getattr(
        response,
        "parsed",
        None,
    )

    if isinstance(parsed, schema):
        return parsed

    if parsed is not None:
        return schema.model_validate(parsed)

    response_text = getattr(
        response,
        "text",
        "",
    )

    if not response_text:
        raise ValueError(
            "Gemini returned no structured response."
        )

    return schema.model_validate(
        json.loads(response_text)
    )


def _limit_words(
    text: str,
    maximum_words: int,
) -> str:
    """
    Enforce a final server-side word limit.

    Prompt instructions are not guaranteed, so this protects the
    mobile UI even when Gemini returns longer wording.
    """

    cleaned = " ".join(
        (text or "").split()
    ).strip()

    if not cleaned:
        return ""

    words = cleaned.split(" ")

    if len(words) <= maximum_words:
        return cleaned

    shortened = " ".join(
        words[:maximum_words]
    ).rstrip(
        " ,;:-"
    )

    return f"{shortened}…"


def _normalise_generated_hazard(
    hazard: GeneratedHazard,
) -> None:
    """Apply mobile UI length and localisation consistency rules."""

    hazard.description = _limit_words(
        hazard.description,
        maximum_words=8,
    )

    hazard.explanation = _limit_words(
        hazard.explanation,
        maximum_words=14,
    )

    hazard.recommended_action = _limit_words(
        hazard.recommended_action,
        maximum_words=12,
    )

    if hazard.localisation_kind != "exact_object":
        hazard.localisation_kind = "none"
        hazard.target_detection_id = None
        hazard.target_class_name = None
        hazard.localisation_confidence = 0.0


def _generate_analysis(
    *,
    image_bytes: bytes,
    mime_type: str,
    detections: list[YoloDetection],
) -> tuple[CombinedLiveAnalysis, str]:
    """
    Run the analysis using the primary Gemini model.

    A fallback model is used only when the previous model
    fails with a temporary capacity or server error.
    """

    detection_context = [
        detection.model_dump()
        for detection in detections
    ]

    prompt = (
        f"{LIVE_ANALYSIS_PROMPT}\n\n"
        "YOLO DETECTIONS:\n"
        f"{json.dumps(detection_context, indent=2)}"
    )

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    model_candidates = tuple(
        dict.fromkeys(
            (
                GEMINI_MODEL,
                *GEMINI_FALLBACK_MODELS,
            )
        )
    )

    last_error: Exception | None = None

    for model_name in model_candidates:
        try:
            print(
                f"Trying Gemini model: {model_name}"
            )

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    image_part,
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=1536,

                    response_mime_type=(
                        "application/json"
                    ),

                    response_schema=(
                        CombinedLiveAnalysis
                    ),

                    thinking_config=(
                        types.ThinkingConfig(
                            thinking_level=(
                                GEMINI_THINKING_LEVEL
                            )
                        )
                    ),

                    media_resolution=(
                        MEDIA_RESOLUTION_MAP[
                            GEMINI_MEDIA_RESOLUTION
                        ]
                    ),
                ),
            )

            analysis = _parse_response(
                response=response,
                schema=CombinedLiveAnalysis,
            )

            analysis.scene_summary = _limit_words(
                analysis.scene_summary,
                maximum_words=16,
            )

            analysis.current_activity = _limit_words(
                analysis.current_activity,
                maximum_words=8,
            )

            analysis.overall_safety_summary = (
                _limit_words(
                    analysis.overall_safety_summary,
                    maximum_words=10,
                )
            )

            for hazard in analysis.hazards:
                _normalise_generated_hazard(
                    hazard
                )

            analysis.hazards_present = bool(
                analysis.hazards
            )

            print(
                f"Gemini analysis succeeded with "
                f"{model_name}"
            )

            return analysis, model_name

        except Exception as error:
            last_error = error

            if not _is_transient_gemini_error(
                error
            ):
                # Invalid prompt, schema, API key or other
                # non-temporary errors should not silently
                # switch models.
                raise

            print(
                f"Gemini model temporarily unavailable: "
                f"{model_name}"
            )

            print(
                type(error).__name__,
                error,
            )

            # Continue to the next configured model.

    raise GeminiServiceUnavailableError(
        "All configured Gemini models are temporarily unavailable."
    ) from last_error


def analyse_live_image(
    *,
    image_bytes: bytes,
    mime_type: str,
) -> LiveAnalysisResult:
    """Run YOLO followed by one Gemini reasoning request."""

    total_start = time.perf_counter()

    detections, yolo_ms = detect_objects(
        image_bytes
    )

    gemini_start = time.perf_counter()

    analysis, model_used = _generate_analysis(
        image_bytes=image_bytes,
        mime_type=mime_type,
        detections=detections,
    )

    gemini_ms = (
        time.perf_counter()
        - gemini_start
    ) * 1000.0

    total_ms = (
        time.perf_counter()
        - total_start
    ) * 1000.0

    print(
        "Live analysis timing | "
        f"YOLO: {yolo_ms:.0f} ms | "
        f"Gemini: {gemini_ms:.0f} ms | "
        f"Total: {total_ms:.0f} ms"
    )

    return LiveAnalysisResult(
        analysis=analysis,
        detections=detections,
        model_used=model_used,
        yolo_ms=yolo_ms,
        gemini_ms=gemini_ms,
        total_ms=total_ms,
    )