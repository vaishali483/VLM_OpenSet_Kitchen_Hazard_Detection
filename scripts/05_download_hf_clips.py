from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ======================================================
# Configuration
# ======================================================

REPO_ID = "lightly-ai/epic-kitchens-100-clips"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "selected_clips"
    / "available_clips.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "clips"
)

# ======================================================

df = pd.read_csv(CSV_PATH)

print(f"\nDownloading {len(df)} clips...\n")

success = []
failed = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    hf_path = row["hf_path"]

    try:

        local_file = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=hf_path,
            local_dir=OUTPUT_ROOT,
            local_dir_use_symlinks=False
        )

        success.append(row["narration_id"])

    except Exception as e:

        print(f"\nFAILED: {hf_path}")
        print(e)

        failed.append(row["narration_id"])

print("\n==========================")
print(f"Downloaded : {len(success)}")
print(f"Failed     : {len(failed)}")
print("==========================")

pd.DataFrame(success, columns=["narration_id"]).to_csv(
    PROJECT_ROOT / "logs" / "hf_downloaded.csv",
    index=False
)

pd.DataFrame(failed, columns=["narration_id"]).to_csv(
    PROJECT_ROOT / "logs" / "hf_failed.csv",
    index=False
)

print("\nDone.")