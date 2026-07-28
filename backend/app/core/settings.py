"""Configuration for the Kitchen Safety Assistant backend."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# settings.py:
# KitchenSafetyAssistant/backend/app/core/settings.py
#
# parents[3]:
# KitchenSafetyAssistant/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = PROJECT_ROOT / ".env"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

load_dotenv(ENV_FILE)


# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
).strip()

GEMINI_FALLBACK_MODELS = tuple(
    model_name.strip()
    for model_name in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.6-flash,gemini-3.5-flash-lite",
    ).split(",")
    if model_name.strip()
)

GEMINI_THINKING_LEVEL = os.getenv(
    "GEMINI_THINKING_LEVEL",
    "low",
).strip().lower()

GEMINI_MEDIA_RESOLUTION = os.getenv(
    "GEMINI_MEDIA_RESOLUTION",
    "medium",
).strip().lower()

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from the project .env file."
    )

if GEMINI_THINKING_LEVEL not in {
    "minimal",
    "low",
    "medium",
    "high",
}:
    raise RuntimeError(
        "GEMINI_THINKING_LEVEL must be minimal, low, "
        "medium or high."
    )

if GEMINI_MEDIA_RESOLUTION not in {
    "low",
    "medium",
    "high",
}:
    raise RuntimeError(
        "GEMINI_MEDIA_RESOLUTION must be low, medium or high."
    )

if not PROMPTS_DIR.exists():
    raise RuntimeError(
        f"Prompts directory was not found: {PROMPTS_DIR}"
    )


# ---------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo11s.pt"
)

YOLO_CONFIDENCE = float(
    os.getenv(
        "YOLO_CONFIDENCE",
        "0.25",
    )
)

YOLO_IMAGE_SIZE = int(
    os.getenv(
        "YOLO_IMAGE_SIZE",
        "640",
    )
)

YOLO_DEVICE = os.getenv(
    "YOLO_DEVICE",
    "cpu",
).strip()

YOLO_RELEVANT_CLASSES = {
    "person",
    "knife",
    "scissors",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "dining table",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
}


# ---------------------------------------------------------------------
# Bounding-box validation
# ---------------------------------------------------------------------

LOCALISATION_MIN_YOLO_CONFIDENCE = float(
    os.getenv(
        "LOCALISATION_MIN_YOLO_CONFIDENCE",
        "0.45",
    )
)

LOCALISATION_MIN_GEMINI_CONFIDENCE = float(
    os.getenv(
        "LOCALISATION_MIN_GEMINI_CONFIDENCE",
        "0.78",
    )
)

LOCALISATION_MIN_BOX_AREA = float(
    os.getenv(
        "LOCALISATION_MIN_BOX_AREA",
        "0.002",
    )
)

LOCALISATION_MAX_BOX_AREA = float(
    os.getenv(
        "LOCALISATION_MAX_BOX_AREA",
        "0.50",
    )
)

# These classes are still supplied to Gemini as useful context,
# but their boxes are too broad to represent a specific kitchen
# hazard accurately.
LOCALISATION_BLOCKED_BOX_CLASSES = {
    "person",
    "chair",
    "dining table",
}


# ---------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
}