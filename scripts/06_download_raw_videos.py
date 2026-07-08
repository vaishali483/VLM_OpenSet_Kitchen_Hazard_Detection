from pathlib import Path
import pandas as pd
import subprocess
import sys

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOWNLOAD_SCRIPT = (
    PROJECT_ROOT
    / "epic-kitchens-download-scripts"
    / "epic_video_downloader_verify.py"
)

DATA_FOLDER = (
    PROJECT_ROOT
    / "epic-kitchens-download-scripts"
    / "data"
)

VIDEO_LIST = (
    PROJECT_ROOT
    / "dataset"
    / "selected_clips"
    / "missing_video_list.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "dataset"
    / "raw_videos"
)

LOG_FOLDER = (
    PROJECT_ROOT
    / "logs"
)

LOG_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ============================================================

videos = pd.read_csv(VIDEO_LIST)

print("=" * 60)
print(f"Videos to download : {len(videos)}")
print("=" * 60)

downloaded = []
failed = []

for i, row in videos.iterrows():

    video_id = row["video_id"]

    print("\n")
    print("=" * 60)
    print(f"[{i+1}/{len(videos)}] {video_id}")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(DOWNLOAD_SCRIPT),

        "--specific-videos",
        video_id,

        "--output-path",
        str(OUTPUT_FOLDER),

        "--epic55-splits",
        str(DATA_FOLDER / "epic_55_splits.csv"),

        "--epic100-splits",
        str(DATA_FOLDER / "epic_100_splits.csv"),

        "--md5",
        str(DATA_FOLDER / "md5.csv"),

        "--errata",
        str(DATA_FOLDER / "errata.csv"),
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        downloaded.append(video_id)
    else:
        failed.append(video_id)

# ============================================================

pd.DataFrame(
    downloaded,
    columns=["video_id"]
).to_csv(
    LOG_FOLDER / "raw_downloaded.csv",
    index=False
)

pd.DataFrame(
    failed,
    columns=["video_id"]
).to_csv(
    LOG_FOLDER / "raw_failed.csv",
    index=False
)

print("\n")
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(f"Downloaded : {len(downloaded)}")
print(f"Failed     : {len(failed)}")