from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

missing = pd.read_csv(
    PROJECT_ROOT
    / "dataset"
    / "selected_clips"
    / "missing_clips.csv"
)

print("=" * 50)
print(f"Missing clips : {len(missing)}")
print("=" * 50)

video_counts = (
    missing["video_id"]
    .value_counts()
    .sort_values(ascending=False)
)

print(f"\nUnique videos needed: {len(video_counts)}\n")

print(video_counts)

video_counts.to_csv(
    PROJECT_ROOT
    / "dataset"
    / "selected_clips"
    / "missing_video_counts.csv"
)

video_list = pd.DataFrame({
    "video_id": video_counts.index,
    "num_clips": video_counts.values
})

video_list.to_csv(
    PROJECT_ROOT
    / "dataset"
    / "selected_clips"
    / "missing_video_list.csv",
    index=False
)

print("\nSaved:")
print("missing_video_counts.csv")
print("missing_video_list.csv")