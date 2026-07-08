import json
import cv2
from pathlib import Path
from tqdm import tqdm

from config import *

# ==========================================================
# Select Dataset
# ==========================================================

SOURCE_DIR = DEMO_DIR if USE_DEMO_DATASET else CLIPS_DIR

print("=" * 60)
print(f"Using dataset: {SOURCE_DIR.name}")
print("=" * 60)

# ==========================================================
# Find all clips recursively
# ==========================================================

clips = sorted(SOURCE_DIR.rglob("*.mp4"))

print(f"Found {len(clips)} clips.\n")

# ==========================================================

processed = 0
skipped = 0
failed = 0

# ==========================================================

for clip in tqdm(clips):

    participant = clip.parent.name
    clip_name = clip.stem

    output_folder = FRAMES_DIR / participant / clip_name

    metadata_file = output_folder / "metadata.json"

    if output_folder.exists() and metadata_file.exists() and not OVERWRITE_FRAMES:
        skipped += 1
        continue

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip))

    if not cap.isOpened():
        print(f"\nCould not open: {clip}")
        failed += 1
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"\nInvalid FPS: {clip}")
        cap.release()
        failed += 1
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps

    frame_interval = max(1, int(round(fps * FRAME_INTERVAL)))

    metadata = {
        "clip_id": clip_name,
        "participant": participant,
        "video_file": clip.name,
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 2),
        "frame_interval_seconds": FRAME_INTERVAL,
        "total_video_frames": total_frames,
        "frames": []
    }

    frame_number = 0
    saved = 0

    last_frame = None
    last_frame_number = None

    # ======================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        last_frame = frame.copy()
        last_frame_number = frame_number

        if frame_number % frame_interval == 0:

            resized = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

            filename = f"frame_{saved:04d}.{IMAGE_FORMAT}"

            cv2.imwrite(
                str(output_folder / filename),
                resized
            )

            metadata["frames"].append({
                "frame_id": saved,
                "file": filename,
                "timestamp": round(frame_number / fps, 2)
            })

            saved += 1

        frame_number += 1

    cap.release()

    # ======================================================
    # Save last frame if requested
    # ======================================================

    if SAVE_LAST_FRAME and last_frame is not None:

        last_timestamp = round(last_frame_number / fps, 2)

        already_saved = False

        if metadata["frames"]:

            previous_timestamp = metadata["frames"][-1]["timestamp"]

            if abs(previous_timestamp - last_timestamp) < 0.05:
                already_saved = True

        if not already_saved:

            resized = cv2.resize(
                last_frame,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

            filename = f"frame_{saved:04d}.{IMAGE_FORMAT}"

            cv2.imwrite(
                str(output_folder / filename),
                resized
            )

            metadata["frames"].append({
                "frame_id": saved,
                "file": filename,
                "timestamp": last_timestamp,
                "last_frame": True
            })

            saved += 1

    metadata["total_frames_saved"] = saved

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    processed += 1

# ==========================================================

print("\n")
print("=" * 60)
print("FRAME EXTRACTION COMPLETE")
print("=" * 60)

print(f"Processed : {processed}")
print(f"Skipped   : {skipped}")
print(f"Failed    : {failed}")
print(f"Output    : {FRAMES_DIR}")
print("=" * 60)