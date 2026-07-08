from pathlib import Path
import pandas as pd
import subprocess
from tqdm import tqdm

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV = (
    PROJECT_ROOT
    / "dataset"
    / "selected_clips"
    / "missing_clips.csv"
)

RAW = (
    PROJECT_ROOT
    / "dataset"
    / "raw_videos"
    / "EPIC-KITCHENS"
)

OUTPUT = (
    PROJECT_ROOT
    / "dataset"
    / "clips"
)

OUTPUT.mkdir(exist_ok=True)

# ==========================================================

df = pd.read_csv(CSV)

print(f"Extracting {len(df)} clips...\n")

success = 0
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df)):

    participant = row["participant_id"]
    video_id = row["video_id"]
    narration_id = row["narration_id"]

    input_video = (
        RAW
        / participant
        / "videos"
        / f"{video_id}.MP4"
    )

    output_dir = OUTPUT / participant
    output_dir.mkdir(exist_ok=True)

    output_video = output_dir / f"{narration_id}.mp4"

    if output_video.exists():
        success += 1
        continue

    cmd = [
        "ffmpeg",
        "-y",

        "-ss",
        str(row["start_timestamp"]),

        "-to",
        str(row["stop_timestamp"]),

        "-i",
        str(input_video),

        "-vf",
        "scale=854:480",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "23",

        "-c:a",
        "aac",

        str(output_video)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode == 0:
        success += 1
    else:
        failed += 1

print("\n")
print("=" * 50)
print(f"Success : {success}")
print(f"Failed  : {failed}")
print("=" * 50)