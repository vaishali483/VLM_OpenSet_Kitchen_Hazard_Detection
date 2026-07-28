import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types

from config import *


# ==========================================================
# Gemini Client
# ==========================================================

client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================================
# Hazard Analysis Output Schema
# ==========================================================

HAZARD_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "hazards_present": {
            "type": "boolean",
            "description": "True if one or more meaningful hazards are present."
        },
        "overall_safety_summary": {
            "type": "string",
            "description": "Brief overall safety assessment of the scene."
        },
        "hazards": {
            "type": "array",
            "description": "List of detected open-set hazards.",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Clear description of the hazardous situation."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Why this situation may be unsafe."
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Estimated severity of the hazard."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence from 0.0 to 1.0 that the hazard is present."
                    },
                    "evidence_frames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Frame filenames that support the hazard."
                    },
                    "older_adult_relevance": {
                        "type": "string",
                        "description": "Why this may be especially relevant for an older adult."
                    },
                    "recommended_action": {
                        "type": "string",
                        "description": "Practical safety action to reduce the hazard."
                    }
                },
                "required": [
                    "description",
                    "explanation",
                    "severity",
                    "confidence",
                    "evidence_frames",
                    "older_adult_relevance",
                    "recommended_action"
                ]
            }
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Unclear or uncertain observations."
        }
    },
    "required": [
        "hazards_present",
        "overall_safety_summary",
        "hazards",
        "uncertainties"
    ]
}


# ==========================================================
# Helper Functions
# ==========================================================

def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def parse_gemini_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()

    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    return json.loads(text)


def load_keyframe_images_with_names(
    keyframe_dir: Path
) -> list[tuple[str, Image.Image]]:
    keyframe_paths = sorted(
        keyframe_dir.glob(f"*.{IMAGE_FORMAT}")
    )

    images = []

    for image_path in keyframe_paths:
        with Image.open(image_path) as img:
            images.append((image_path.name, img.copy()))

    return images


def build_contents(
    prompt_text: str,
    scene_understanding: dict[str, Any],
    keyframe_images: list[tuple[str, Image.Image]]
) -> list[Any]:
    contents = []

    contents.append(prompt_text)

    contents.append(
        "SCENE UNDERSTANDING JSON:\n\n"
        + json.dumps(scene_understanding, indent=2, ensure_ascii=False)
    )

    contents.append(
        "Now analyse the selected keyframes below. "
        "Each image is preceded by its frame filename. "
        "If you cite evidence, use exactly these frame filenames."
    )

    for frame_name, image in keyframe_images:
        contents.append(f"FRAME_FILENAME: {frame_name}")
        contents.append(image)

    return contents


def add_hazard_ids(
    result: dict[str, Any],
    clip_name: str
) -> dict[str, Any]:
    hazards = result.get("hazards", [])

    for index, hazard in enumerate(hazards, start=1):
        hazard["hazard_id"] = f"{clip_name}_H{index:03d}"

    return result


def validate_basic_output(result: dict[str, Any]) -> dict[str, Any]:
    """
    Basic cleanup in case the model returns small inconsistencies.
    """

    if "hazards" not in result or result["hazards"] is None:
        result["hazards"] = []

    if not result["hazards"]:
        result["hazards_present"] = False

    else:
        result["hazards_present"] = True

    if "uncertainties" not in result or result["uncertainties"] is None:
        result["uncertainties"] = []

    if "overall_safety_summary" not in result:
        result["overall_safety_summary"] = ""

    return result


def is_retryable_error(error: Exception) -> bool:
    error_text = str(error)

    retryable_keywords = [
        "503",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "429",
        "high demand",
        "temporarily"
    ]

    return any(keyword in error_text for keyword in retryable_keywords)


def call_gemini_with_retries(
    contents: list[Any],
    max_retries: int = 3,
    initial_wait_seconds: int = 60
):
    """
    Handles temporary Gemini failures such as 503 high-demand errors.
    """

    wait_seconds = initial_wait_seconds

    for attempt in range(1, max_retries + 1):

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    response_schema=HAZARD_ANALYSIS_SCHEMA,
                ),
            )

            return response

        except Exception as error:

            if attempt == max_retries or not is_retryable_error(error):
                raise error

            print(
                f"\nTemporary Gemini error. "
                f"Retry {attempt}/{max_retries - 1} after {wait_seconds} seconds."
            )

            time.sleep(wait_seconds)

            wait_seconds *= 2


# ==========================================================
# Main Processing Function
# ==========================================================

def process_hazard_analysis(
    subset: str,
    overwrite: bool = False,
    delay_seconds: int = 90,
    max_retries: int = 3
) -> None:

    scene_understanding_root = SCENE_UNDERSTANDING_DIR / subset
    keyframe_root = KEYFRAMES_DIR / subset

    # If you added HAZARD_ANALYSIS_DIR to config.py, this will use it.
    # Otherwise, it safely creates dataset/hazard_analysis.
    hazard_output_root = globals().get(
        "HAZARD_ANALYSIS_DIR",
        DATASET_DIR / "hazard_analysis"
    ) / subset

    if not scene_understanding_root.exists():
        raise FileNotFoundError(
            f"Scene understanding folder not found: {scene_understanding_root}"
        )

    if not keyframe_root.exists():
        raise FileNotFoundError(
            f"Keyframe folder not found: {keyframe_root}"
        )

    prompt_path = globals().get(
        "HAZARD_PROMPT",
        PROMPTS_DIR / "hazard_analysis_prompt.txt"
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Hazard prompt not found: {prompt_path}"
        )

    prompt_text = prompt_path.read_text(encoding="utf-8")

    scene_files = sorted(
        scene_understanding_root.rglob("scene_understanding.json")
    )

    print("\n" + "=" * 60)
    print("OPEN-SET HAZARD ANALYSIS")
    print("=" * 60)
    print(f"Subset                  : {subset}")
    print(f"Scene understanding root: {scene_understanding_root}")
    print(f"Keyframe root           : {keyframe_root}")
    print(f"Output root             : {hazard_output_root}")
    print(f"Found clips             : {len(scene_files)}")
    print("=" * 60 + "\n")

    if not scene_files:
        print("No scene_understanding.json files found.")
        return

    processed = 0
    skipped = 0
    failed = 0

    for scene_file in tqdm(scene_files):

        clip_name = scene_file.parent.name
        participant = clip_name.split("_")[0]

        output_dir = hazard_output_root / clip_name
        output_json = output_dir / "hazard_analysis.json"

        if output_json.exists() and not overwrite:
            skipped += 1
            continue

        try:
            scene_understanding = load_json(scene_file)

            keyframe_dir = keyframe_root / clip_name

            if not keyframe_dir.exists():
                print(f"\nMissing keyframe directory: {keyframe_dir}")
                failed += 1
                continue

            keyframe_images = load_keyframe_images_with_names(keyframe_dir)

            if not keyframe_images:
                print(f"\nNo keyframe images found for: {clip_name}")
                failed += 1
                continue

            contents = build_contents(
                prompt_text=prompt_text,
                scene_understanding=scene_understanding,
                keyframe_images=keyframe_images
            )

            response = call_gemini_with_retries(
                contents=contents,
                max_retries=max_retries,
                initial_wait_seconds=delay_seconds
            )

            raw_text = response.text.strip()

            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "gemini_raw.txt", "w", encoding="utf-8") as file:
                file.write(raw_text)

            parsed = parse_gemini_json(raw_text)

            parsed = validate_basic_output(parsed)

            parsed = add_hazard_ids(
                result=parsed,
                clip_name=clip_name
            )

            parsed["clip_id"] = clip_name
            parsed["participant"] = participant

            parsed["metadata"] = {
                "model": GEMINI_MODEL,
                "temperature": TEMPERATURE,
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "subset": subset,
                "task": "open_set_hazard_analysis",
                "keyframes_used": len(keyframe_images),
                "predefined_hazard_categories_used": False,
            }

            save_json(parsed, output_json)

            processed += 1

        except Exception as error:

            print(f"\nFailed clip: {clip_name}")
            print(type(error).__name__)
            print(error)

            failed += 1

            if "RESOURCE_EXHAUSTED" in str(error):
                print("\nGemini quota reached.")
                print("Stopping inference.")
                break

        time.sleep(delay_seconds)

    print("\n" + "=" * 60)
    print("HAZARD ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Subset    : {subset}")
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Output    : {hazard_output_root}")
    print("=" * 60)


# ==========================================================
# Command Line
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-set VLM hazard analysis on selected keyframes."
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="P07",
        help="Dataset subset folder. Example: P07, demo, P02."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing hazard_analysis.json files."
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=90,
        help="Delay in seconds between Gemini API calls."
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum Gemini retries for temporary 503/429 errors."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_hazard_analysis(
        subset=args.subset,
        overwrite=args.overwrite,
        delay_seconds=args.delay,
        max_retries=args.max_retries
    )