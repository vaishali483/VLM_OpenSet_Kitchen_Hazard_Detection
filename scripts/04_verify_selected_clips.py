from pathlib import Path
import pandas as pd
from huggingface_hub import list_repo_files

# --------------------------------------------------
# Configuration
# --------------------------------------------------

REPO_ID = "lightly-ai/epic-kitchens-100-clips"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "selected_clips"
    / "final_selected_clips.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "dataset"
    / "selected_clips"
)

# --------------------------------------------------
# Load selected clips
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

print(f"Selected clips : {len(df)}")

# --------------------------------------------------
# Fetch HuggingFace repository file list
# --------------------------------------------------

print("\nFetching repository file list...")
print("This may take 1-2 minutes...\n")

repo_files = list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

print(f"Repository contains {len(repo_files)} files.")

repo_set = set(repo_files)

# --------------------------------------------------
# Verify clips
# --------------------------------------------------

exists = []
paths = []

for narration_id in df["narration_id"]:

    participant = narration_id.split("_")[0]

    path = f"clips/{participant}/{narration_id}.mp4"

    paths.append(path)

    exists.append(path in repo_set)

df["hf_path"] = paths
df["available"] = exists

# --------------------------------------------------
# Save reports
# --------------------------------------------------

available = df[df["available"]].copy()
missing = df[~df["available"]].copy()

available.to_csv(
    OUTPUT_FOLDER / "available_clips.csv",
    index=False
)

missing.to_csv(
    OUTPUT_FOLDER / "missing_clips.csv",
    index=False
)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n==============================")
print(f"Available : {len(available)}")
print(f"Missing   : {len(missing)}")
print("==============================")

if len(missing):

    print("\nFirst missing clips:\n")

    print(
        missing[
            [
                "narration_id",
                "verb",
                "noun"
            ]
        ].head(20)
    )

print("\nDone.")