import json
import shutil
from pathlib import Path

from tqdm import tqdm

from config import *

# ==========================================================
# Find Scene JSON files
# ==========================================================

scene_files = sorted(SCENE_JSON_DIR.rglob("scene.json"))

print(f"Found {len(scene_files)} scene files.\n")

processed = 0
failed = 0

# ==========================================================

for scene_file in tqdm(scene_files):

    participant = scene_file.parent.parent.name
    clip_name = scene_file.parent.name

    try:

        with open(scene_file, "r") as f:
            scene = json.load(f)

        output_dir = KEYFRAMES_DIR / participant / clip_name
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_source = FRAMES_DIR / participant / clip_name

        selected = []

        previous_objects = None

        total_frames = len(scene["frames"])

        for i, frame in enumerate(scene["frames"]):

            current_objects = set(frame["detected_classes"])

            keep = False
            reason = []
            change_score = 0

            # --------------------------------------------
            # First frame
            # --------------------------------------------

            if i == 0 and KEEP_FIRST_FRAME:
                keep = True
                reason.append("first_frame")

            # --------------------------------------------
            # Last frame
            # --------------------------------------------

            if i == total_frames - 1 and KEEP_LAST_FRAME:
                keep = True
                reason.append("last_frame")

            # --------------------------------------------
            # Object change
            # --------------------------------------------

            if previous_objects is not None and KEEP_OBJECT_CHANGE:

                added = current_objects - previous_objects
                removed = previous_objects - current_objects

                change_score = len(added) + len(removed)

                if change_score > 0:
                    keep = True
                    reason.append("object_change")

            previous_objects = current_objects

            if keep:

                src = frame_source / frame["frame"]
                dst = output_dir / frame["frame"]

                if src.exists():
                    shutil.copy2(src, dst)

                selected.append({

                    "frame": frame["frame"],

                    "timestamp": frame["timestamp"],

                    "reason": reason,

                    "change_score": change_score,

                    "detected_classes": frame["detected_classes"],

                    "num_objects": frame["num_objects"]

                })

        with open(output_dir / "keyframes.json", "w") as f:
            json.dump(
                {
                    "clip_id": clip_name,
                    "participant": participant,
                    "total_frames": total_frames,
                    "selected_frames": len(selected),
                    "frames": selected
                },
                f,
                indent=4
            )

        processed += 1

    except Exception as e:

        print(f"\nFailed: {clip_name}")
        print(e)

        failed += 1

print("\n" + "=" * 60)
print("KEYFRAME SELECTION COMPLETE")
print("=" * 60)
print(f"Processed : {processed}")
print(f"Failed    : {failed}")
print(f"Output    : {KEYFRAMES_DIR}")
print("=" * 60)