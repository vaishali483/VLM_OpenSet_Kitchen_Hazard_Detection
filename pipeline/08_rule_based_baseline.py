from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from config import (
    DATASET_DIR,
    IMAGE_FORMAT,
    KEYFRAMES_DIR,
    PROJECT_ROOT,
    SCENE_JSON_DIR,
)


RULES_PATH = PROJECT_ROOT / "rules" / "kitchen_rules.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def get_keyframe_names(
    subset: str,
    clip_name: str,
) -> set[str]:
    keyframe_dir = KEYFRAMES_DIR / subset / clip_name

    if not keyframe_dir.exists():
        return set()

    return {
        path.name
        for path in keyframe_dir.glob(
            f"*.{IMAGE_FORMAT}"
        )
    }


def filter_to_keyframes(
    frames: list[dict[str, Any]],
    keyframe_names: set[str],
) -> list[dict[str, Any]]:
    if not keyframe_names:
        return frames

    return [
        frame
        for frame in frames
        if frame.get("frame") in keyframe_names
    ]


def object_matches(
    detected_object: dict[str, Any],
    object_classes: list[str],
    min_confidence: float,
) -> bool:
    return (
        detected_object.get("class_name")
        in object_classes
        and float(
            detected_object.get("confidence", 0.0)
        )
        >= min_confidence
    )


def bbox_gap_ratio(
    bbox_1: list[float],
    bbox_2: list[float],
    image_width: int,
    image_height: int,
) -> float:
    """
    Calculate the shortest gap between two bounding boxes,
    normalised by the image diagonal.

    A result of 0 means the boxes overlap or touch.
    """

    x1_a, y1_a, x2_a, y2_a = bbox_1
    x1_b, y1_b, x2_b, y2_b = bbox_2

    horizontal_gap = max(
        x1_a - x2_b,
        x1_b - x2_a,
        0.0,
    )

    vertical_gap = max(
        y1_a - y2_b,
        y1_b - y2_a,
        0.0,
    )

    pixel_gap = math.sqrt(
        horizontal_gap**2 + vertical_gap**2
    )

    image_diagonal = math.sqrt(
        image_width**2 + image_height**2
    )

    if image_diagonal == 0:
        return float("inf")

    return pixel_gap / image_diagonal


def bbox_near_edge(
    bbox: list[float],
    image_width: int,
    image_height: int,
    edge_margin_ratio: float,
) -> bool:
    x1, y1, x2, y2 = bbox

    margin_x = image_width * edge_margin_ratio
    margin_y = image_height * edge_margin_ratio

    return (
        x1 <= margin_x
        or x2 >= image_width - margin_x
        or y1 <= margin_y
        or y2 >= image_height - margin_y
    )


def apply_object_proximity_rule(
    rule: dict[str, Any],
    frames: list[dict[str, Any]],
) -> tuple[bool, list[str], float]:
    evidence_frames: list[str] = []
    supporting_confidences: list[float] = []

    for frame in frames:
        image_size = frame.get("image_size", {})
        image_width = int(image_size.get("width", 0))
        image_height = int(image_size.get("height", 0))

        if image_width <= 0 or image_height <= 0:
            continue

        objects = frame.get("objects", [])

        group_1_objects = [
            obj
            for obj in objects
            if object_matches(
                detected_object=obj,
                object_classes=rule["object_group_1"],
                min_confidence=rule["min_confidence"],
            )
        ]

        group_2_objects = [
            obj
            for obj in objects
            if object_matches(
                detected_object=obj,
                object_classes=rule["object_group_2"],
                min_confidence=rule["min_confidence"],
            )
        ]

        frame_triggered = False

        for object_1 in group_1_objects:
            for object_2 in group_2_objects:
                bbox_1 = object_1.get("bbox")
                bbox_2 = object_2.get("bbox")

                if not bbox_1 or not bbox_2:
                    continue

                gap_ratio = bbox_gap_ratio(
                    bbox_1=bbox_1,
                    bbox_2=bbox_2,
                    image_width=image_width,
                    image_height=image_height,
                )

                if gap_ratio <= rule["max_gap_ratio"]:
                    confidence = min(
                        float(
                            object_1.get(
                                "confidence",
                                0.0,
                            )
                        ),
                        float(
                            object_2.get(
                                "confidence",
                                0.0,
                            )
                        ),
                    )

                    supporting_confidences.append(
                        confidence
                    )
                    frame_triggered = True

        if frame_triggered:
            evidence_frames.append(frame["frame"])

    if not evidence_frames:
        return False, [], 0.0

    return (
        True,
        sorted(set(evidence_frames)),
        max(supporting_confidences),
    )


def apply_near_image_edge_rule(
    rule: dict[str, Any],
    frames: list[dict[str, Any]],
) -> tuple[bool, list[str], float]:
    evidence_frames: list[str] = []
    supporting_confidences: list[float] = []

    for frame in frames:
        image_size = frame.get("image_size", {})
        image_width = int(image_size.get("width", 0))
        image_height = int(image_size.get("height", 0))

        if image_width <= 0 or image_height <= 0:
            continue

        for detected_object in frame.get(
            "objects",
            [],
        ):
            if not object_matches(
                detected_object=detected_object,
                object_classes=rule["object_classes"],
                min_confidence=rule["min_confidence"],
            ):
                continue

            bbox = detected_object.get("bbox")

            if not bbox:
                continue

            if bbox_near_edge(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
                edge_margin_ratio=rule[
                    "edge_margin_ratio"
                ],
            ):
                evidence_frames.append(
                    frame["frame"]
                )

                supporting_confidences.append(
                    float(
                        detected_object.get(
                            "confidence",
                            0.0,
                        )
                    )
                )

    if not evidence_frames:
        return False, [], 0.0

    return (
        True,
        sorted(set(evidence_frames)),
        max(supporting_confidences),
    )


def apply_rule(
    rule: dict[str, Any],
    frames: list[dict[str, Any]],
) -> tuple[bool, list[str], float]:
    rule_type = rule.get("type")

    if rule_type == "object_proximity":
        return apply_object_proximity_rule(
            rule,
            frames,
        )

    if rule_type == "near_image_edge":
        return apply_near_image_edge_rule(
            rule,
            frames,
        )

    raise ValueError(
        f"Unsupported rule type: {rule_type}"
    )


def build_hazard(
    rule: dict[str, Any],
    clip_name: str,
    hazard_index: int,
    evidence_frames: list[str],
    confidence: float,
) -> dict[str, Any]:
    return {
        "hazard_id": (
            f"{clip_name}_B{hazard_index:03d}"
        ),
        "triggered_rule_id": rule["rule_id"],
        "rule_name": rule["name"],
        "description": rule["description"],
        "explanation": rule["explanation"],
        "severity": rule["severity"],
        "confidence": round(confidence, 4),
        "evidence_frames": evidence_frames,
        "older_adult_relevance": rule[
            "older_adult_relevance"
        ],
        "recommended_action": rule[
            "recommended_action"
        ],
    }


def run_rule_based_baseline(
    subset: str,
    overwrite: bool = False,
) -> None:
    if not RULES_PATH.exists():
        raise FileNotFoundError(
            "rules/kitchen_rules.json was not found."
        )

    rules_data = load_json(RULES_PATH)
    rules = rules_data.get("rules", [])

    scene_root = SCENE_JSON_DIR / subset
    output_root = (
        DATASET_DIR
        / "baseline_hazards"
        / subset
    )

    if not scene_root.exists():
        raise FileNotFoundError(
            f"Scene JSON subset not found: {scene_root}"
        )

    scene_files = sorted(
        scene_root.rglob("scene.json")
    )

    print("\n" + "=" * 60)
    print("RULE-BASED BASELINE")
    print("=" * 60)
    print(f"Subset : {subset}")
    print(f"Rules  : {len(rules)}")
    print(f"Found  : {len(scene_files)} clips")
    print(
        f"Output : dataset/baseline_hazards/{subset}"
    )
    print("=" * 60 + "\n")

    processed = 0
    skipped = 0
    failed = 0

    for scene_file in scene_files:
        clip_name = scene_file.parent.name
        participant = clip_name.split("_")[0]

        output_json = (
            output_root
            / clip_name
            / "baseline_hazards.json"
        )

        if output_json.exists() and not overwrite:
            skipped += 1
            continue

        try:
            scene = load_json(scene_file)

            frames = scene.get("frames", [])

            keyframe_names = get_keyframe_names(
                subset=subset,
                clip_name=clip_name,
            )

            frames = filter_to_keyframes(
                frames=frames,
                keyframe_names=keyframe_names,
            )

            hazards: list[dict[str, Any]] = []

            for rule in rules:
                (
                    triggered,
                    evidence_frames,
                    confidence,
                ) = apply_rule(
                    rule=rule,
                    frames=frames,
                )

                if not triggered:
                    continue

                hazards.append(
                    build_hazard(
                        rule=rule,
                        clip_name=clip_name,
                        hazard_index=len(hazards) + 1,
                        evidence_frames=evidence_frames,
                        confidence=confidence,
                    )
                )

            result = {
                "clip_id": clip_name,
                "participant": participant,
                "system": "rule_based_baseline",
                "ruleset": {
                    "name": rules_data.get(
                        "name",
                        "kitchen_rules",
                    ),
                    "version": rules_data.get(
                        "version",
                        "unknown",
                    ),
                    "frozen_before_final_evaluation":
                        rules_data.get(
                            "frozen_before_final_evaluation",
                            False,
                        ),
                },
                "predefined_hazard_categories_used":
                    True,
                "hazards_present": bool(hazards),
                "overall_safety_summary": (
                    "One or more predefined rules "
                    "were triggered."
                    if hazards
                    else
                    "No predefined rule was triggered."
                ),
                "hazards": hazards,
                "uncertainties": [
                    (
                        "The baseline is limited to "
                        "predefined rules and YOLO "
                        "detections."
                    ),
                    (
                        "It cannot reason about hazards "
                        "outside its fixed rule set."
                    ),
                ],
            }

            save_json(result, output_json)
            processed += 1

        except Exception as error:
            print(f"Failed: {clip_name}")
            print(type(error).__name__)
            print(error)
            failed += 1

    print("\n" + "=" * 60)
    print("RULE-BASED BASELINE COMPLETE")
    print("=" * 60)
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(
        f"Output    : "
        f"dataset/baseline_hazards/{subset}"
    )
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the closed-set rule-based baseline."
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
        help="Replace existing baseline outputs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    run_rule_based_baseline(
        subset=arguments.subset,
        overwrite=arguments.overwrite,
    )