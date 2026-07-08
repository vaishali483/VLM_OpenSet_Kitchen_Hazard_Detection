import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import *

# ==========================================================
# Load annotations
# ==========================================================

print("Loading annotation CSV...")

annotations = pd.read_csv(DATASET_DIR / "selected_clips/final_selected_clips.csv")

annotation_lookup = {}

for _, row in annotations.iterrows():
    annotation_lookup[row["narration_id"]] = row.to_dict()

print(f"Loaded {len(annotation_lookup)} annotations.\n")

# ==========================================================

clip_folders = sorted(
    [
        p
        for p in DETECTIONS_DIR.rglob("*")
        if (p / "detections.json").exists()
    ]
)

print(f"Found {len(clip_folders)} detection folders.\n")

processed = 0
failed = 0

# ==========================================================

for clip_folder in tqdm(clip_folders):

    participant = clip_folder.parent.name
    clip_name = clip_folder.name

    output_dir = SCENE_JSON_DIR / participant / clip_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / "scene.json"

    if output_json.exists():
        processed += 1
        continue

    try:

        with open(clip_folder / "detections.json", "r") as f:
            detections = json.load(f)

        annotation = annotation_lookup.get(clip_name)

        if annotation is None:
            print(f"Missing annotation: {clip_name}")
            failed += 1
            continue

        # ==================================================
        # Statistics
        # ==================================================

        object_counter = Counter()

        unique_objects = set()

        first_seen = {}

        last_seen = {}

        total_objects = 0

        for frame in detections["frames"]:

            timestamp = frame["timestamp"]

            for obj in frame["objects"]:

                cls = obj["class_name"]

                object_counter[cls] += 1

                unique_objects.add(cls)

                total_objects += 1

                if cls not in first_seen:
                    first_seen[cls] = timestamp

                last_seen[cls] = timestamp

        total_frames = len(detections["frames"])

        avg_objects = (
            round(total_objects / total_frames, 2)
            if total_frames > 0 else 0
        )

        # ==================================================
        # Build Scene JSON
        # ==================================================

        scene = {

            "clip_id": clip_name,

            "participant": participant,

            "video_id": annotation["video_id"],

            "annotation": {

                "verb": annotation["verb"],

                "noun": annotation["noun"],

                "category": annotation["category"],

                "narration": annotation["narration"]

            },

            "clip": {

                "start_timestamp": annotation["start_timestamp"],

                "stop_timestamp": annotation["stop_timestamp"]

            },

            "frame_summary": {

                "total_frames": total_frames,

                "average_objects_per_frame": avg_objects

            },

            "object_summary": dict(object_counter),

            "unique_objects": sorted(list(unique_objects)),

            "first_seen": first_seen,

            "last_seen": last_seen,

            "frames": detections["frames"]

        }

        with open(output_json, "w") as f:
            json.dump(scene, f, indent=4)

        processed += 1

    except Exception as e:

        print(f"\nFailed: {clip_name}")

        print(e)

        failed += 1

# ==========================================================

print("\n")

print("=" * 60)
print("SCENE JSON BUILD COMPLETE")
print("=" * 60)

print(f"Processed : {processed}")
print(f"Failed    : {failed}")
print(f"Output    : {SCENE_JSON_DIR}")

print("=" * 60)