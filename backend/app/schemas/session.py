"""Cooking-session persistence and API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SessionState = Literal[
    "active",
    "paused",
    "stopped",
]

HazardStatus = Literal[
    "present",
    "possible",
]

Severity = Literal[
    "low",
    "medium",
    "high",
    "critical",
]


class SessionHazardObservation(BaseModel):
    canonical_name: str
    hazard_status: HazardStatus
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    trigger_condition: str | None = None


class SessionFrameRecord(BaseModel):
    frame_index: int
    capture_id: str
    analysed_at_utc: str

    image_file: str | None = None

    current_action: str = ""
    action_confidence: float = 0.0
    scene_summary: str = ""
    scene_observations: list[str] = Field(
        default_factory=list
    )

    hazards: list[SessionHazardObservation] = Field(
        default_factory=list
    )


class AggregatedSessionHazard(BaseModel):
    canonical_name: str
    hazard_status: HazardStatus
    highest_severity: Severity
    max_confidence: float

    occurrence_count: int
    first_seen_utc: str
    last_seen_utc: str


class CookingSessionRecord(BaseModel):
    session_id: str
    state: SessionState

    started_at_utc: str
    paused_at_utc: str | None = None
    stopped_at_utc: str | None = None

    frames: list[SessionFrameRecord] = Field(
        default_factory=list
    )

    aggregated_hazards: list[AggregatedSessionHazard] = Field(
        default_factory=list
    )


class SessionStartResponse(BaseModel):
    session_id: str
    state: SessionState
    started_at_utc: str


class SessionStateResponse(BaseModel):
    session_id: str
    state: SessionState
    message: str


class SessionSummaryResponse(BaseModel):
    session_id: str
    state: SessionState

    started_at_utc: str
    stopped_at_utc: str

    analysed_frame_count: int

    # Names only and already ordered by severity.
    present_hazards: list[str]
    possible_hazards: list[str]
