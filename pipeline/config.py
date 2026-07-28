from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Dataset
# ==========================================================

DATASET_DIR = PROJECT_ROOT / "dataset"

CLIPS_DIR = DATASET_DIR / "clips"

DEMO_DIR = DATASET_DIR / "demo"

FRAMES_DIR = DATASET_DIR / "frames"

DETECTIONS_DIR = DATASET_DIR / "detections"

SCENE_JSON_DIR = DATASET_DIR / "scene_json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

LOGS_DIR = PROJECT_ROOT / "logs"

PROMPTS_DIR = PROJECT_ROOT / "prompts"

ANNOTATED_FRAMES_DIR = DATASET_DIR / "annotated_frames"

KEYFRAMES_DIR = DATASET_DIR / "keyframes"

SCENE_UNDERSTANDING_DIR = DATASET_DIR / "scene_understanding"

HAZARD_ANALYSIS_DIR = DATASET_DIR / "hazard_analysis"

# ==========================================================
# Models
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models"

YOLO_MODEL = "yolo11n.pt"

# ==========================================================
# Frame Extraction
# ==========================================================

FRAME_INTERVAL = 1.0      # seconds

FRAME_WIDTH = 854

FRAME_HEIGHT = 480

IMAGE_FORMAT = "jpg"

# ==========================================================
# Development
# ==========================================================

USE_DEMO_DATASET = False

OVERWRITE_FRAMES = False

# ==========================================================
# Create folders automatically
# ==========================================================

for folder in [
    FRAMES_DIR,
    DETECTIONS_DIR,
    SCENE_JSON_DIR,
    ANNOTATED_FRAMES_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
    KEYFRAMES_DIR,
    SCENE_UNDERSTANDING_DIR,
    HAZARD_ANALYSIS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Frame Extraction
# ==========================================================

FRAME_INTERVAL = 1.0      # seconds

SAVE_LAST_FRAME = True

FRAME_WIDTH = 854

FRAME_HEIGHT = 480

IMAGE_FORMAT = "jpg"

# ==========================================================
# YOLO
# ==========================================================

YOLO_MODEL = "yolo11s.pt"

YOLO_CONFIDENCE = 0.25

YOLO_IOU = 0.45

SAVE_ANNOTATED_IMAGES = True

# ==========================================================
# Keyframe Selection
# ==========================================================

KEEP_FIRST_FRAME = True
KEEP_LAST_FRAME = True
KEEP_OBJECT_CHANGE = True

# ==========================================================
# Gemini
# ==========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please set it in your .env file."
    )

GEMINI_MODEL = "gemini-3.5-flash"

MAX_OUTPUT_TOKENS = 4096

TEMPERATURE = 0.2

# ==========================================================
# Prompts
# ==========================================================

SCENE_PROMPT = PROMPTS_DIR / "scene_understanding_prompt.txt"

HAZARD_PROMPT = PROMPTS_DIR / "hazard_analysis_prompt.txt"