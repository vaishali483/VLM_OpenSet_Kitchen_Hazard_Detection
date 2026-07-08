import json
import time
import cv2
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from config import *

# ==========================================================
# Load YOLO
# ==========================================================

print("\nLoading YOLO model...")

model = YOLO(YOLO_MODEL)

print("Model loaded.")

# ==========================================================
# Find all clip folders
# ==========================================================

clip_folders = sorted(
    [p for p in FRAMES_DIR.rglob("*") if p.is_dir()]
)

clip_folders = [
    p for p in clip_folders
    if (p / "metadata.json").exists()
]

print(f"\nFound {len(clip_folders)} clips.\n")

processed = 0
failed = 0

# ==========================================================

for clip_folder in tqdm(clip_folders):

    participant = clip_folder.parent.name
    clip_name = clip_folder.name

    output_dir = DETECTIONS_DIR / participant / clip_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / "detections.json"

    if output_json.exists():
        processed += 1
        continue

    metadata_file = clip_folder / "metadata.json"

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    detections = {
        "clip_id": clip_name,
        "participant": participant,
        "model": YOLO_MODEL,
        "frames": []
    }

    frame_files = sorted(
        clip_folder.glob(f"*.{IMAGE_FORMAT}")
    )

    try:

        for frame in frame_files:

            timestamp = None

            for item in metadata["frames"]:
                if item["file"] == frame.name:
                    timestamp = item["timestamp"]
                    break

            start = time.time()

            results = model.predict(
                source=str(frame),
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                verbose=False
            )

            inference_time = round(
                (time.time() - start) * 1000,
                2
            )

            result = results[0]

            # ----------------------------------------------------------
            # Save annotated image
            # ----------------------------------------------------------

            if SAVE_ANNOTATED_IMAGES:

                annotated_dir = (
                    ANNOTATED_FRAMES_DIR
                    / participant
                    / clip_name
                )

                annotated_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                annotated_image = result.plot()

                cv2.imwrite(
                    str(annotated_dir / frame.name),
                    annotated_image
                )

            frame_data = {
                "frame": frame.name,
                "timestamp": timestamp,
                "image_size": {
                    "width": result.orig_shape[1],
                    "height": result.orig_shape[0]
                },
                "inference_time_ms": inference_time,
                "objects": []
            }

            for box in result.boxes:

                cls = int(box.cls[0])

                frame_data["objects"].append({

                    "class_id": cls,

                    "class_name": model.names[cls],

                    "confidence": round(float(box.conf[0]), 4),

                    "bbox": [
                        round(float(x), 2)
                        for x in box.xyxy[0].tolist()
                    ]

                })

            # ----------------------------------------------------------
            # Frame summary
            # ----------------------------------------------------------

            frame_data["num_objects"] = len(frame_data["objects"])

            frame_data["detected_classes"] = sorted(
                list({
                    obj["class_name"]
                    for obj in frame_data["objects"]
                })
            )

            detections["frames"].append(frame_data)

        with open(output_json, "w") as f:
            json.dump(detections, f, indent=4)

        processed += 1

    except Exception as e:

        print(f"\nFailed: {clip_name}")
        print(e)

        failed += 1

# ==========================================================

print("\n")
print("=" * 60)
print("YOLO DETECTION COMPLETE")
print("=" * 60)
print(f"Processed : {processed}")
print(f"Failed    : {failed}")
print(f"Output    : {DETECTIONS_DIR}")
print("=" * 60)