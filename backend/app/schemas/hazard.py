"""Schemas returned from FastAPI to Unity."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ImageReceipt(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    width: int
    height: int


class NormalizedBoundingBox(BaseModel):
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)


class HazardVisualisation(BaseModel):
    mode: Literal[
        "warning_label",
        "bounding_box",
        "approximate_box",
    ]

    bbox_normalized: NormalizedBoundingBox | None = None

    source_detection_id: str | None = None
    source_class_name: str | None = None

    localisation_confidence: float = 0.0
    validation_reason: str = ""


class HazardItem(BaseModel):
    hazard_id: str

    canonical_name: str = ""
    hazard_status: Literal[
        "present",
        "possible",
    ] = "present"

    trigger_condition: str | None = None

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
    visualisation: HazardVisualisation


class DetectionItem(BaseModel):
    detection_id: str
    class_name: str
    confidence: float
    bbox_normalized: NormalizedBoundingBox


class ProcessingTimes(BaseModel):
    yolo_ms: float
    gemini_ms: float
    total_ms: float


class HazardAnalysisResponse(BaseModel):
    event_type: str
    mode: str
    capture_id: str
    completed_at_utc: str

    image_received: ImageReceipt

    hazards_present: bool
    overall_safety_summary: str
    hazards: list[HazardItem]
    uncertainties: list[str]

    scene_summary: str | None = None
    current_activity: str | None = None
    action_confidence: float = 0.0
    scene_observations: list[str] = Field(
        default_factory=list
    )

    model_used: str | None = None

    detections: list[DetectionItem] = Field(
        default_factory=list
    )

    processing_times: ProcessingTimes | None = None

    # These two fields are populated only by session analysis.
    session_id: str | None = None
    frame_index: int | None = None
