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
# Scene Understanding Output Schema
# ==========================================================

SCENE_UNDERSTANDING_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A short description of the visible kitchen scene."
        },
        "current_activity": {
            "type": "string",
            "description": "The main activity currently visible in the keyframes."
        },
        "activity_confidence": {
            "type": "number",
            "description": "Confidence in the current activity, from 0.0 to 1.0."
        },
        "objects_in_use": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Objects directly being used by the person."
        },
        "important_objects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important visible objects in the scene."
        },
        "environment": {
            "type": "object",
            "properties": {
                "stove_visible": {"type": "boolean"},
                "sink_visible": {"type": "boolean"},
                "counter_visible": {"type": "boolean"},
                "other_context": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Other useful environmental observations."
                }
            },
            "required": [
                "stove_visible",
                "sink_visible",
                "counter_visible",
                "other_context"
            ]
        },
        "observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Neutral visual observations without hazard judgement."
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Anything unclear or not confidently visible."
        }
    },
    "required": [
        "summary",
        "current_activity",
        "activity_confidence",
        "objects_in_use",
        "important_objects",
        "environment",
        "observations",
        "uncertainties"
    ]
}


# ==========================================================
# Extra instruction to prevent hazard analysis in this stage
# ==========================================================

SCENE_ONLY_INSTRUCTION = """
IMPORTANT TASK BOUNDARY:

You are only performing scene understanding.

Your job is to describe:
- what is visible,
- what activity is happening,
- which objects are being used,
- which objects are important,
- what the kitchen environment looks like.

Do NOT perform hazard detection in this step.

Do NOT output:
- potential_hazards,
- hazard descriptions,
- severity,
- safety recommendations,
- next_probable_action,
- detailed reasoning chain.

Do not classify anything as dangerous here.
The next pipeline stage will perform hazard analysis separately.

Return only valid JSON matching the required schema.
"""


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


def clean_scene_json_for_vlm(scene_json: dict[str, Any]) -> dict[str, Any]:
    """
    Remove human annotation leakage before sending metadata to Gemini.

    The EPIC-KITCHENS fields such as verb, noun, category and narration
    are human-provided labels. If we send them to Gemini, the model may
    copy the activity label rather than infer the activity from the image.
    """

    cleaned = dict(scene_json)

    # Remove human annotation labels.
    cleaned.pop("annotation", None)

    # Keep useful non-semantic metadata if available.
    # video_id and clip timestamps are acceptable because they do not
    # directly tell Gemini what the activity is.
    return cleaned


def remove_unwanted_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Make sure old prompt fields do not survive in the output.
    """

    unwanted_keys = [
        "potential_hazards",
        "hazards",
        "hazards_present",
        "next_probable_action",
        "reasoning",
        "safety_recommendations",
        "severity",
        "confidence"
    ]

    for key in unwanted_keys:
        parsed.pop(key, None)

    return parsed


def parse_gemini_json(text: str) -> dict[str, Any]:
    """
    Parse Gemini JSON safely, including cases where the model wraps
    JSON inside markdown fences.
    """

    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()

    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    return json.loads(text)


def get_participant_from_subset_and_clip(subset: str, clip_name: str) -> str:
    """
    For demo clips:
        subset = demo
        clip_name = P02_03_131
        participant = P02

    For participant folders:
        subset = P02
        clip_name = P02_03_131
        participant = P02
    """

    if subset.lower() == "demo":
        return clip_name.split("_")[0]

    return subset


def load_keyframe_images(keyframe_dir: Path) -> list[Image.Image]:
    keyframe_paths = sorted(
        keyframe_dir.glob(f"*.{IMAGE_FORMAT}")
    )

    images = []

    for image_path in keyframe_paths:
        with Image.open(image_path) as img:
            images.append(img.copy())

    return images


def build_contents(
    prompt_text: str,
    cleaned_scene_json: dict[str, Any],
    keyframe_images: list[Image.Image]
) -> list[Any]:
    contents = []

    contents.append(prompt_text)

    contents.append(SCENE_ONLY_INSTRUCTION)

    contents.append(
        "CLEANED YOLO SCENE JSON METADATA:\n\n"
        + json.dumps(cleaned_scene_json, indent=2, ensure_ascii=False)
    )

    contents.append(
        "Now analyse the following selected keyframes visually. "
        "Use the metadata only as support. Infer the current activity "
        "from the images and metadata, but do not perform hazard analysis."
    )

    for img in keyframe_images:
        contents.append(img)

    return contents


# ==========================================================
# Main Processing Function
# ==========================================================

def process_scene_understanding(
    subset: str,
    overwrite: bool = False,
    delay_seconds: int = 20
) -> None:

    scene_root = SCENE_JSON_DIR / subset
    keyframe_root = KEYFRAMES_DIR / subset
    output_root = SCENE_UNDERSTANDING_DIR / subset

    if not scene_root.exists():
        raise FileNotFoundError(f"Scene JSON folder not found: {scene_root}")

    if not keyframe_root.exists():
        raise FileNotFoundError(f"Keyframe folder not found: {keyframe_root}")

    scene_files = sorted(scene_root.rglob("scene.json"))

    print("\n" + "=" * 60)
    print("SCENE UNDERSTANDING")
    print("=" * 60)
    print(f"Subset       : {subset}")
    print(f"Scene root   : {scene_root}")
    print(f"Keyframe root: {keyframe_root}")
    print(f"Found clips  : {len(scene_files)}")
    print("=" * 60 + "\n")

    if not scene_files:
        print("No scene.json files found.")
        return

    prompt_text = SCENE_PROMPT.read_text(encoding="utf-8")

    processed = 0
    skipped = 0
    failed = 0

    for scene_file in tqdm(scene_files):

        clip_name = scene_file.parent.name
        participant = get_participant_from_subset_and_clip(
            subset=subset,
            clip_name=clip_name
        )

        output_dir = output_root / clip_name
        output_json = output_dir / "scene_understanding.json"

        if output_json.exists() and not overwrite:
            skipped += 1
            continue

        try:
            scene_json = load_json(scene_file)

            cleaned_scene_json = clean_scene_json_for_vlm(scene_json)

            keyframe_dir = keyframe_root / clip_name

            if not keyframe_dir.exists():
                print(f"\nMissing keyframe directory: {keyframe_dir}")
                failed += 1
                continue

            keyframe_images = load_keyframe_images(keyframe_dir)

            if not keyframe_images:
                print(f"\nNo keyframe images found for: {clip_name}")
                failed += 1
                continue

            contents = build_contents(
                prompt_text=prompt_text,
                cleaned_scene_json=cleaned_scene_json,
                keyframe_images=keyframe_images
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    response_schema=SCENE_UNDERSTANDING_SCHEMA,
                ),
            )

            raw_text = response.text.strip()

            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "gemini_raw.txt", "w", encoding="utf-8") as file:
                file.write(raw_text)

            parsed = parse_gemini_json(raw_text)
            parsed = remove_unwanted_fields(parsed)

            parsed["clip_id"] = clip_name
            parsed["participant"] = participant

            parsed["metadata"] = {
                "model": GEMINI_MODEL,
                "temperature": TEMPERATURE,
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "subset": subset,
                "keyframes_used": len(keyframe_images),
                "human_epic_annotation_removed": True,
                "task": "scene_understanding_only",
            }

            save_json(parsed, output_json)

            save_json(
                {
                    "model": GEMINI_MODEL,
                    "temperature": TEMPERATURE,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "subset": subset,
                    "clip_id": clip_name,
                    "keyframes_used": len(keyframe_images),
                    "human_epic_annotation_removed": True,
                    "output_json": str(output_json)
                },
                output_dir / "metadata.json"
            )

            processed += 1

        except Exception as error:
            print(f"\nFailed clip: {clip_name}")
            print(type(error).__name__)
            print(error)

            failed += 1

            if "RESOURCE_EXHAUSTED" in str(error):
                print("\nDaily Gemini quota reached.")
                print("Stopping inference.")
                break

        time.sleep(delay_seconds)

    print("\n" + "=" * 60)
    print("SCENE UNDERSTANDING COMPLETE")
    print("=" * 60)
    print(f"Subset    : {subset}")
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Output    : {output_root}")
    print("=" * 60)


# ==========================================================
# Command Line
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini scene understanding on selected keyframes."
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="demo",
        help=(
            "Dataset subset folder under dataset/scene_json and "
            "dataset/keyframes. Examples: demo, P01, P02."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene_understanding.json files."
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=20,
        help="Delay in seconds between Gemini API calls."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_scene_understanding(
        subset=args.subset,
        overwrite=args.overwrite,
        delay_seconds=args.delay
    )