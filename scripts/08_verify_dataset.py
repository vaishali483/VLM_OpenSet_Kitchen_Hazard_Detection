from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV = (
    PROJECT_ROOT
    / "dataset"
    / "selected_clips"
    / "final_selected_clips.csv"
)

CLIPS = (
    PROJECT_ROOT
    / "dataset"
    / "clips"
)

df = pd.read_csv(CSV)

missing = []

for narration in df["narration_id"]:

    participant = narration.split("_")[0]

    clip = (
        CLIPS
        / participant
        / f"{narration}.mp4"
    )

    if not clip.exists():
        missing.append(narration)

print("=" * 50)

print("Expected clips :", len(df))

print("Found clips    :", len(df) - len(missing))

print("Missing clips  :", len(missing))

print("=" * 50)

if missing:

    print("\nMissing files:\n")

    for m in missing:
        print(m)