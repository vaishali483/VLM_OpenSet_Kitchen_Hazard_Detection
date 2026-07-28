import json
import time
from pathlib import Path

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

scene_files = sorted(
    (SCENE_JSON_DIR / "demo").rglob("scene.json")
)

print(f"\nFound {len(scene_files)} demo clips.\n")

processed = 0
failed = 0

# ==========================================================

prompt_text = SCENE_PROMPT.read_text(encoding="utf-8")

# ==========================================================

for scene_file in tqdm(scene_files):

    clip_name = scene_file.parent.name
    participant = clip_name.split("_")[0]

    output_dir = (SCENE_UNDERSTANDING_DIR / "demo" / clip_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / "scene_understanding.json"

    if output_json.exists():
        processed += 1
        continue

    try:

        with open(scene_file) as f:
            scene_json = json.load(f)

        keyframe_dir = KEYFRAMES_DIR / "demo" / clip_name

        keyframe_images = sorted(
            keyframe_dir.glob(f"*.{IMAGE_FORMAT}")
        )

        contents = []

        contents.append(prompt_text)

        contents.append(
            "YOLO Scene JSON:\n\n"
            + json.dumps(scene_json, indent=2)
        )

        for image_path in keyframe_images:

            img = Image.open(image_path)

            contents.append(img)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )

        text = response.text.strip()

        metadata = {
            "model": GEMINI_MODEL,
            "temperature": TEMPERATURE,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        with open(output_dir / "gemini_raw.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # Remove markdown if Gemini wraps JSON
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()

        elif text.startswith("```"):
            text = text.replace("```", "").strip()

        parsed = json.loads(text)

        parsed["clip_id"] = clip_name
        parsed["participant"] = participant

        with open(output_json, "w") as f:
            json.dump(parsed, f, indent=4)

        processed += 1

    except Exception as e:

        print(f"\nFailed clip : {clip_name}")
        print(type(e).__name__)
        print(e)

        failed += 1

        if "RESOURCE_EXHAUSTED" in str(e):
            print("\nDaily Gemini quota reached.")
            print("Stopping inference.")
            break
    
    time.sleep(20)

# ==========================================================

print("\n" + "=" * 60)
print("SCENE UNDERSTANDING COMPLETE")
print("=" * 60)
print(f"Processed : {processed}")
print(f"Failed    : {failed}")
print(f"Output    : {SCENE_UNDERSTANDING_DIR}")
print("=" * 60)