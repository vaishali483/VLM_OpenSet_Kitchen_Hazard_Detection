from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import DATASET_DIR


FIELDS_TO_REMOVE = {
    "input_scene_json",
    "output_json",
    "scene_understanding_input",
    "rules_file",
}


def remove_fields(value: Any) -> bool:
    """Recursively remove local-path metadata fields."""

    changed = False

    if isinstance(value, dict):
        for key in list(value.keys()):
            if key in FIELDS_TO_REMOVE:
                del value[key]
                changed = True
                continue

            if remove_fields(value[key]):
                changed = True

    elif isinstance(value, list):
        for item in value:
            if remove_fields(item):
                changed = True

    return changed


def clean_json_file(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        changed = remove_fields(data)

        if changed:
            with path.open("w", encoding="utf-8") as file:
                json.dump(
                    data,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

        return changed

    except json.JSONDecodeError:
        print(f"Skipped invalid JSON: {path}")
        return False


def main() -> None:
    roots = [
        DATASET_DIR / "scene_understanding",
        DATASET_DIR / "hazard_analysis",
        DATASET_DIR / "baseline_hazards",
    ]

    checked = 0
    cleaned = 0

    for root in roots:
        if not root.exists():
            continue

        for json_file in root.rglob("*.json"):
            checked += 1

            if clean_json_file(json_file):
                cleaned += 1
                print(
                    f"Cleaned: "
                    f"{json_file.relative_to(DATASET_DIR.parent)}"
                )

    print("\n" + "=" * 60)
    print("LOCAL PATH CLEANUP COMPLETE")
    print("=" * 60)
    print(f"Files checked : {checked}")
    print(f"Files cleaned : {cleaned}")
    print("=" * 60)


if __name__ == "__main__":
    main()