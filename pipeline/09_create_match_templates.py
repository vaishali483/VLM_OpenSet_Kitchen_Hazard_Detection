"""Create minimal manual prediction-to-ground-truth matching templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import DATASET_DIR, PROJECT_ROOT


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a JSON file without storing machine-specific paths."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def simplify_ground_truth_hazard(
    hazard: dict[str, Any],
) -> dict[str, Any]:
    """Keep only fields useful during manual matching."""

    return {
        "id": hazard.get("ground_truth_id", ""),
        "description": hazard.get("description", ""),
        "severity": hazard.get("severity", ""),
        "evidence_frames": hazard.get(
            "evidence_frames",
            [],
        ),
        "is_novel_relative_to_rule_baseline":
            hazard.get(
                "is_novel_relative_to_rule_baseline"
            ),
    }


def simplify_prediction(
    hazard: dict[str, Any],
) -> dict[str, Any]:
    """Keep only fields useful during manual matching."""

    simplified = {
        "id": hazard.get("hazard_id", ""),
        "description": hazard.get("description", ""),
        "severity": hazard.get("severity", ""),
        "evidence_frames": hazard.get(
            "evidence_frames",
            [],
        ),
    }

    # Include the rule ID only for baseline predictions.
    if hazard.get("triggered_rule_id"):
        simplified["triggered_rule_id"] = hazard[
            "triggered_rule_id"
        ]

    if hazard.get("rule_name"):
        simplified["rule_name"] = hazard["rule_name"]

    return simplified


def validate_unique_ids(
    hazards: list[dict[str, Any]],
    id_key: str,
    clip_name: str,
    source_name: str,
) -> None:
    """Raise an error when duplicate hazard IDs are found."""

    ids = [
        hazard.get(id_key)
        for hazard in hazards
        if hazard.get(id_key)
    ]

    duplicate_ids = {
        hazard_id
        for hazard_id in ids
        if ids.count(hazard_id) > 1
    }

    if duplicate_ids:
        raise ValueError(
            f"Duplicate IDs in {source_name} for "
            f"{clip_name}: {sorted(duplicate_ids)}"
        )


def create_match_templates(
    subset: str,
    overwrite: bool = False,
) -> None:
    """Create one minimal matches.json file per clip."""

    ground_truth_root = (
        DATASET_DIR
        / "ground_truth"
        / subset
    )

    vlm_root = (
        DATASET_DIR
        / "hazard_analysis"
        / subset
    )

    baseline_root = (
        DATASET_DIR
        / "baseline_hazards"
        / subset
    )

    output_root = (
        PROJECT_ROOT
        / "evaluation"
        / "matching"
        / subset
    )

    if not ground_truth_root.exists():
        raise FileNotFoundError(
            "Ground-truth subset folder not found: "
            f"{ground_truth_root}"
        )

    ground_truth_files = sorted(
        ground_truth_root.rglob("ground_truth.json")
    )

    print("\n" + "=" * 60)
    print("CREATE MATCHING TEMPLATES")
    print("=" * 60)
    print(f"Subset : {subset}")
    print(
        f"Found  : {len(ground_truth_files)} "
        "ground-truth files"
    )
    print(
        f"Output : evaluation/matching/{subset}"
    )
    print("=" * 60 + "\n")

    created = 0
    skipped = 0
    failed = 0

    for ground_truth_file in ground_truth_files:
        clip_name = ground_truth_file.parent.name

        output_json = (
            output_root
            / clip_name
            / "matches.json"
        )

        if output_json.exists() and not overwrite:
            skipped += 1
            continue

        try:
            ground_truth = load_json(
                ground_truth_file
            )

            if ground_truth.get(
                "clip_contains_hazard"
            ) is None:
                print(
                    f"Skipped {clip_name}: "
                    "ground truth is not completed."
                )
                skipped += 1
                continue

            ground_truth_hazards = ground_truth.get(
                "ground_truth_hazards",
                [],
            )

            validate_unique_ids(
                hazards=ground_truth_hazards,
                id_key="ground_truth_id",
                clip_name=clip_name,
                source_name="ground truth",
            )

            vlm_file = (
                vlm_root
                / clip_name
                / "hazard_analysis.json"
            )

            baseline_file = (
                baseline_root
                / clip_name
                / "baseline_hazards.json"
            )

            vlm_data: dict[str, Any] = {
                "hazards": []
            }

            baseline_data: dict[str, Any] = {
                "hazards": []
            }

            if vlm_file.exists():
                vlm_data = load_json(vlm_file)
            else:
                print(
                    f"Warning: missing VLM output for "
                    f"{clip_name}"
                )

            if baseline_file.exists():
                baseline_data = load_json(
                    baseline_file
                )
            else:
                print(
                    f"Warning: missing baseline output "
                    f"for {clip_name}"
                )

            vlm_hazards = vlm_data.get(
                "hazards",
                [],
            )

            baseline_hazards = baseline_data.get(
                "hazards",
                [],
            )

            validate_unique_ids(
                hazards=vlm_hazards,
                id_key="hazard_id",
                clip_name=clip_name,
                source_name="VLM predictions",
            )

            validate_unique_ids(
                hazards=baseline_hazards,
                id_key="hazard_id",
                clip_name=clip_name,
                source_name="baseline predictions",
            )

            template = {
                "clip_id": clip_name,
                "ground_truth_hazards": [
                    simplify_ground_truth_hazard(
                        hazard
                    )
                    for hazard
                    in ground_truth_hazards
                ],
                "vlm_predictions": [
                    simplify_prediction(hazard)
                    for hazard in vlm_hazards
                ],
                "baseline_predictions": [
                    simplify_prediction(hazard)
                    for hazard in baseline_hazards
                ],
                "vlm_matches": [],
                "baseline_matches": [],
            }

            save_json(template, output_json)
            created += 1

        except Exception as error:
            print(f"Failed: {clip_name}")
            print(type(error).__name__)
            print(error)
            failed += 1

    print("\n" + "=" * 60)
    print("MATCHING TEMPLATE CREATION COMPLETE")
    print("=" * 60)
    print(f"Created : {created}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(
        f"Output  : evaluation/matching/{subset}"
    )
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create minimal manual hazard "
            "matching templates."
        )
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="P07",
        help="Dataset subset, for example P07.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing matches.json files. "
            "This removes any manually entered matches."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    create_match_templates(
        subset=arguments.subset,
        overwrite=arguments.overwrite,
    )