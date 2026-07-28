from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import DATASET_DIR, IMAGE_FORMAT, KEYFRAMES_DIR


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save JSON using a Git-friendly, platform-independent format."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def create_ground_truth_templates(
    subset: str,
    overwrite: bool = False,
) -> None:
    """Create one ground-truth template for every keyframe folder."""

    keyframe_root = KEYFRAMES_DIR / subset
    output_root = DATASET_DIR / "ground_truth" / subset

    if not keyframe_root.exists():
        raise FileNotFoundError(
            f"Keyframe subset folder not found: {keyframe_root}"
        )

    clip_folders = sorted(
        folder
        for folder in keyframe_root.iterdir()
        if folder.is_dir()
    )

    print("\n" + "=" * 60)
    print("CREATE GROUND-TRUTH TEMPLATES")
    print("=" * 60)
    print(f"Subset : {subset}")
    print(f"Found  : {len(clip_folders)} clip folders")
    print(f"Output : dataset/ground_truth/{subset}")
    print("=" * 60 + "\n")

    created = 0
    skipped = 0
    failed = 0

    for clip_folder in clip_folders:
        clip_name = clip_folder.name
        participant = clip_name.split("_")[0]

        output_json = (
            output_root
            / clip_name
            / "ground_truth.json"
        )

        if output_json.exists() and not overwrite:
            skipped += 1
            continue

        try:
            keyframes = sorted(
                path.name
                for path in clip_folder.glob(
                    f"*.{IMAGE_FORMAT}"
                )
            )

            if not keyframes:
                print(
                    f"Skipped {clip_name}: "
                    "no keyframe images found."
                )
                skipped += 1
                continue

            template = {
                "clip_id": clip_name,
                "participant": participant,
                "keyframes": keyframes,
                "clip_contains_hazard": None,
                "ground_truth_hazards": [],
            }

            save_json(template, output_json)
            created += 1

        except Exception as error:
            print(f"Failed: {clip_name}")
            print(type(error).__name__)
            print(error)
            failed += 1

    print("\n" + "=" * 60)
    print("GROUND-TRUTH TEMPLATE CREATION COMPLETE")
    print("=" * 60)
    print(f"Created : {created}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(f"Output  : dataset/ground_truth/{subset}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create minimal manual ground-truth templates."
        )
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="P07",
        help="Dataset subset, for example P07 or demo.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing ground-truth templates.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    create_ground_truth_templates(
        subset=arguments.subset,
        overwrite=arguments.overwrite,
    )