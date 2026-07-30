"""Persistent sampled-frame cooking-session service."""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from app.core.settings import (
    SESSION_CONTEXT_FRAME_LIMIT,
    SESSION_DATA_DIR,
    SESSION_DEDUPLICATION_THRESHOLD,
    STORE_SESSION_IMAGES,
)
from app.schemas.hazard import HazardAnalysisResponse
from app.schemas.session import (
    AggregatedSessionHazard,
    CookingSessionRecord,
    SessionFrameRecord,
    SessionHazardObservation,
    SessionStartResponse,
    SessionStateResponse,
    SessionSummaryResponse,
)


SEVERITY_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

STATUS_RANK = {
    "possible": 1,
    "present": 2,
}


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _normalise_name(value: str) -> str:
    text = value.strip().lower()

    replacements = {
        "worktop": "counter",
        "stove top": "stovetop",
        "cook top": "stovetop",
        "blade": "knife",
    }

    for old_value, new_value in replacements.items():
        text = text.replace(
            old_value,
            new_value,
        )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _name_similarity(
    first: str,
    second: str,
) -> float:
    first_normalised = _normalise_name(
        first
    )
    second_normalised = _normalise_name(
        second
    )

    if first_normalised == second_normalised:
        return 1.0

    first_tokens = set(
        first_normalised.split()
    )
    second_tokens = set(
        second_normalised.split()
    )

    union = first_tokens | second_tokens

    jaccard = (
        len(first_tokens & second_tokens)
        / len(union)
        if union
        else 0.0
    )

    sequence_score = SequenceMatcher(
        None,
        first_normalised,
        second_normalised,
    ).ratio()

    return max(
        jaccard,
        sequence_score,
    )


class SessionService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[
            str,
            CookingSessionRecord,
        ] = {}

        SESSION_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def start_session(
        self,
    ) -> SessionStartResponse:
        with self._lock:
            session_id = str(uuid4())
            started_at = _utc_now()

            record = CookingSessionRecord(
                session_id=session_id,
                state="active",
                started_at_utc=started_at,
            )

            self._sessions[
                session_id
            ] = record

            self._persist(record)

            return SessionStartResponse(
                session_id=session_id,
                state=record.state,
                started_at_utc=started_at,
            )

    def pause_session(
        self,
        session_id: str,
    ) -> SessionStateResponse:
        with self._lock:
            record = self._get_record(
                session_id
            )

            if record.state == "stopped":
                raise ValueError(
                    "A stopped session cannot be paused."
                )

            record.state = "paused"
            record.paused_at_utc = _utc_now()

            self._persist(record)

            return SessionStateResponse(
                session_id=session_id,
                state=record.state,
                message="Cooking session paused.",
            )

    def resume_session(
        self,
        session_id: str,
    ) -> SessionStateResponse:
        with self._lock:
            record = self._get_record(
                session_id
            )

            if record.state == "stopped":
                raise ValueError(
                    "A stopped session cannot be resumed."
                )

            record.state = "active"
            record.paused_at_utc = None

            self._persist(record)

            return SessionStateResponse(
                session_id=session_id,
                state=record.state,
                message="Cooking session resumed.",
            )

    def require_active(
        self,
        session_id: str,
    ) -> None:
        with self._lock:
            record = self._get_record(
                session_id
            )

            if record.state != "active":
                raise ValueError(
                    "The cooking session is not active."
                )

    def build_text_context(
        self,
        session_id: str,
    ) -> dict[str, object]:
        """
        Build compact text-only memory.

        No prior image bytes, image paths or image objects are returned.
        """

        with self._lock:
            record = self._get_record(
                session_id
            )

            recent_frames = record.frames[
                -SESSION_CONTEXT_FRAME_LIMIT:
            ]

            return {
                "session_active": (
                    record.state == "active"
                ),
                "recent_frames": [
                    {
                        "frame_index": frame.frame_index,
                        "analysed_at_utc": (
                            frame.analysed_at_utc
                        ),
                        "current_action": (
                            frame.current_action
                        ),
                        "scene_summary": (
                            frame.scene_summary
                        ),
                        "scene_observations": (
                            frame.scene_observations
                        ),
                        "hazards": [
                            {
                                "canonical_name": (
                                    hazard.canonical_name
                                ),
                                "hazard_status": (
                                    hazard.hazard_status
                                ),
                                "severity": (
                                    hazard.severity
                                ),
                                "trigger_condition": (
                                    hazard.trigger_condition
                                ),
                            }
                            for hazard in frame.hazards
                        ],
                    }
                    for frame in recent_frames
                ],
                "known_hazards": [
                    {
                        "canonical_name": (
                            hazard.canonical_name
                        ),
                        "hazard_status": (
                            hazard.hazard_status
                        ),
                        "highest_severity": (
                            hazard.highest_severity
                        ),
                        "last_seen_utc": (
                            hazard.last_seen_utc
                        ),
                    }
                    for hazard
                    in record.aggregated_hazards
                ],
            }

    def record_analysis(
        self,
        *,
        session_id: str,
        capture_id: str,
        image_bytes: bytes,
        content_type: str,
        response: HazardAnalysisResponse,
    ) -> int:
        with self._lock:
            record = self._get_record(
                session_id
            )

            frame_index = len(record.frames) + 1

            image_file = self._save_image(
                record=record,
                frame_index=frame_index,
                capture_id=capture_id,
                image_bytes=image_bytes,
                content_type=content_type,
            )

            hazard_observations = [
                SessionHazardObservation(
                    canonical_name=(
                        hazard.canonical_name
                        or hazard.description
                    ),
                    hazard_status=(
                        hazard.hazard_status
                    ),
                    severity=hazard.severity,
                    confidence=hazard.confidence,
                    trigger_condition=(
                        hazard.trigger_condition
                    ),
                )
                for hazard in response.hazards
            ]

            frame = SessionFrameRecord(
                frame_index=frame_index,
                capture_id=capture_id,
                analysed_at_utc=(
                    response.completed_at_utc
                ),
                image_file=image_file,
                current_action=(
                    response.current_activity
                    or ""
                ),
                action_confidence=(
                    response.action_confidence
                ),
                scene_summary=(
                    response.scene_summary
                    or ""
                ),
                scene_observations=(
                    response.scene_observations
                ),
                hazards=hazard_observations,
            )

            record.frames.append(frame)

            for observation in hazard_observations:
                self._merge_hazard(
                    record,
                    observation,
                    response.completed_at_utc,
                )

            self._persist(record)

            return frame_index

    def stop_session(
        self,
        session_id: str,
    ) -> SessionSummaryResponse:
        with self._lock:
            record = self._get_record(
                session_id
            )

            record.state = "stopped"
            record.stopped_at_utc = (
                record.stopped_at_utc
                or _utc_now()
            )

            self._persist(record)

            ordered = sorted(
                record.aggregated_hazards,
                key=lambda hazard: (
                    -SEVERITY_RANK[
                        hazard.highest_severity
                    ],
                    -STATUS_RANK[
                        hazard.hazard_status
                    ],
                    hazard.canonical_name.lower(),
                ),
            )

            return SessionSummaryResponse(
                session_id=record.session_id,
                state=record.state,
                started_at_utc=(
                    record.started_at_utc
                ),
                stopped_at_utc=(
                    record.stopped_at_utc
                    or _utc_now()
                ),
                analysed_frame_count=(
                    len(record.frames)
                ),
                present_hazards=[
                    hazard.canonical_name
                    for hazard in ordered
                    if hazard.hazard_status
                    == "present"
                ],
                possible_hazards=[
                    hazard.canonical_name
                    for hazard in ordered
                    if hazard.hazard_status
                    == "possible"
                ],
            )

    def get_session(
        self,
        session_id: str,
    ) -> CookingSessionRecord:
        with self._lock:
            return self._get_record(
                session_id
            ).model_copy(
                deep=True
            )

    def _merge_hazard(
        self,
        record: CookingSessionRecord,
        observation: SessionHazardObservation,
        observed_at: str,
    ) -> None:
        best_match: AggregatedSessionHazard | None = None
        best_score = 0.0

        for candidate in record.aggregated_hazards:
            score = _name_similarity(
                observation.canonical_name,
                candidate.canonical_name,
            )

            if score > best_score:
                best_match = candidate
                best_score = score

        if (
            best_match is None
            or best_score
            < SESSION_DEDUPLICATION_THRESHOLD
        ):
            record.aggregated_hazards.append(
                AggregatedSessionHazard(
                    canonical_name=(
                        observation.canonical_name
                    ),
                    hazard_status=(
                        observation.hazard_status
                    ),
                    highest_severity=(
                        observation.severity
                    ),
                    max_confidence=(
                        observation.confidence
                    ),
                    occurrence_count=1,
                    first_seen_utc=observed_at,
                    last_seen_utc=observed_at,
                )
            )
            return

        best_match.occurrence_count += 1
        best_match.last_seen_utc = observed_at
        best_match.max_confidence = max(
            best_match.max_confidence,
            observation.confidence,
        )

        if (
            SEVERITY_RANK[
                observation.severity
            ]
            > SEVERITY_RANK[
                best_match.highest_severity
            ]
        ):
            best_match.highest_severity = (
                observation.severity
            )

        if (
            STATUS_RANK[
                observation.hazard_status
            ]
            > STATUS_RANK[
                best_match.hazard_status
            ]
        ):
            best_match.hazard_status = (
                observation.hazard_status
            )

    def _get_record(
        self,
        session_id: str,
    ) -> CookingSessionRecord:
        if session_id in self._sessions:
            return self._sessions[session_id]

        session_file = (
            SESSION_DATA_DIR
            / session_id
            / "session.json"
        )

        if not session_file.exists():
            raise KeyError(
                f"Unknown session: {session_id}"
            )

        record = CookingSessionRecord.model_validate_json(
            session_file.read_text(
                encoding="utf-8"
            )
        )

        self._sessions[session_id] = record

        return record

    def _persist(
        self,
        record: CookingSessionRecord,
    ) -> None:
        session_dir = (
            SESSION_DATA_DIR
            / record.session_id
        )

        session_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            session_dir
            / "session.json.tmp"
        )

        final_path = (
            session_dir
            / "session.json"
        )

        temporary_path.write_text(
            record.model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            final_path
        )

    def _save_image(
        self,
        *,
        record: CookingSessionRecord,
        frame_index: int,
        capture_id: str,
        image_bytes: bytes,
        content_type: str,
    ) -> str | None:
        if not STORE_SESSION_IMAGES:
            return None

        extension = (
            ".png"
            if content_type == "image/png"
            else ".jpg"
        )

        safe_capture_id = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            capture_id,
        )[:80]

        frames_dir = (
            SESSION_DATA_DIR
            / record.session_id
            / "frames"
        )

        frames_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"{frame_index:04d}_"
            f"{safe_capture_id}"
            f"{extension}"
        )

        (frames_dir / filename).write_bytes(
            image_bytes
        )

        # Relative path only. This path is stored locally and is not
        # included in the Gemini session context.
        return str(
            Path("frames")
            / filename
        ).replace("\\", "/")


session_service = SessionService()
