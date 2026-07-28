"""Internal schemas for YOLO-grounded Gemini analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NormalizedBoundingBox(BaseModel):
    """Bounding-box coordinates normalised to 0–1."""

    x_min: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_min: float = Field(
        ge=0.0,
        le=1.0,
    )

    x_max: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_max: float = Field(
        ge=0.0,
        le=1.0,
    )


class YoloDetection(BaseModel):
    """One YOLO object detection."""

    detection_id: str
    class_name: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    bbox_normalized: NormalizedBoundingBox


class GeneratedHazard(BaseModel):
    description: str
    explanation: str

    severity: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    recommended_action: str

    localisation_kind: Literal[
        "exact_object",
        "approximate_region",
        "none",
    ] = "none"

    target_detection_id: str | None = None
    target_class_name: str | None = None

    localisation_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    estimated_bbox_normalized: (
        NormalizedBoundingBox | None
    ) = None


class CombinedLiveAnalysis(BaseModel):
    """Scene understanding and hazards from one Gemini call."""

    scene_summary: str
    current_activity: str

    hazards_present: bool
    overall_safety_summary: str

    hazards: list[GeneratedHazard] = Field(
        default_factory=list,
        max_length=3,
    )

    uncertainties: list[str] = Field(
        default_factory=list,
        max_length=3,
    )