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


DEFAULT_RULES_FILENAME = "kitchen_rules_evaluation.json"

SUPPORTED_RULE_TYPES = {
    "object_proximity",
    "object_overlap",
    "surface_edge_proximity",
}


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


def resolve_rules_path(rules_argument: str) -> Path:
    candidate = Path(rules_argument)

    if candidate.is_absolute():
        return candidate

    if candidate.parent != Path("."):
        return (PROJECT_ROOT / candidate).resolve()

    return (
        PROJECT_ROOT
        / "rules"
        / candidate.name
    ).resolve()


def validate_non_empty_class_list(
    rule: dict[str, Any],
    key: str,
) -> None:
    classes = rule.get(key)

    if (
        not isinstance(classes, list)
        or not classes
        or not all(
            isinstance(item, str) and item.strip()
            for item in classes
        )
    ):
        raise ValueError(
            f"{rule['rule_id']} must contain a "
            f"non-empty string list: {key}"
        )


def validate_ratio(
    rule: dict[str, Any],
    key: str,
) -> None:
    try:
        value = float(rule[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{rule['rule_id']} must contain a "
            f"numeric {key}."
        ) from error

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{rule['rule_id']} {key} must be "
            "between 0 and 1."
        )


def validate_ruleset(
    rules_data: dict[str, Any],
) -> list[dict[str, Any]]:
    rules = rules_data.get("rules", [])

    if not isinstance(rules, list) or not rules:
        raise ValueError(
            "The ruleset must contain a non-empty rules list."
        )

    seen_ids: set[str] = set()

    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError(
                "Every ruleset entry must be a JSON object."
            )

        rule_id = str(
            rule.get("rule_id", "")
        ).strip()

        if not rule_id:
            raise ValueError(
                "Every rule must contain a non-empty rule_id."
            )

        if rule_id in seen_ids:
            raise ValueError(
                f"Duplicate rule_id: {rule_id}"
            )

        seen_ids.add(rule_id)

        rule_type = str(
            rule.get("type", "")
        ).strip()

        if rule_type not in SUPPORTED_RULE_TYPES:
            raise ValueError(
                f"Unsupported rule type in {rule_id}: "
                f"{rule_type}"
            )

        for required_key in (
            "name",
            "severity",
            "description",
            "explanation",
            "older_adult_relevance",
            "recommended_action",
        ):
            if not rule.get(required_key):
                raise ValueError(
                    f"{rule_id} is missing required field: "
                    f"{required_key}"
                )

        if rule_type in {
            "object_proximity",
            "object_overlap",
        }:
            validate_non_empty_class_list(
                rule,
                "object_group_1",
            )
            validate_non_empty_class_list(
                rule,
                "object_group_2",
            )
            validate_ratio(
                rule,
                "min_confidence",
            )

        if rule_type == "object_proximity":
            validate_ratio(
                rule,
                "max_gap_ratio",
            )

        if rule_type == "object_overlap":
            validate_ratio(
                rule,
                "minimum_subject_overlap_ratio",
            )

        if rule_type == "surface_edge_proximity":
            validate_non_empty_class_list(
                rule,
                "object_classes",
            )
            validate_non_empty_class_list(
                rule,
                "surface_classes",
            )
            validate_ratio(
                rule,
                "min_confidence",
            )
            validate_ratio(
                rule,
                "min_surface_confidence",
            )
            validate_ratio(
                rule,
                "minimum_support_overlap_ratio",
            )
            validate_ratio(
                rule,
                "max_edge_gap_ratio",
            )

    return rules


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


def bbox_area(
    bbox: list[float],
) -> float:
    x1, y1, x2, y2 = bbox

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def bbox_intersection_area(
    bbox_1: list[float],
    bbox_2: list[float],
) -> float:
    x1_a, y1_a, x2_a, y2_a = bbox_1
    x1_b, y1_b, x2_b, y2_b = bbox_2

    intersection_width = max(
        0.0,
        min(x2_a, x2_b) - max(x1_a, x1_b),
    )

    intersection_height = max(
        0.0,
        min(y2_a, y2_b) - max(y1_a, y1_b),
    )

    return (
        intersection_width
        * intersection_height
    )


def bbox_subject_overlap_ratio(
    subject_bbox: list[float],
    other_bbox: list[float],
) -> float:
    subject_area = bbox_area(subject_bbox)

    if subject_area <= 0:
        return 0.0

    return (
        bbox_intersection_area(
            subject_bbox,
            other_bbox,
        )
        / subject_area
    )


def bbox_gap_ratio(
    bbox_1: list[float],
    bbox_2: list[float],
    image_width: int,
    image_height: int,
) -> float:
    """Shortest box gap normalised by image diagonal."""

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
        horizontal_gap**2
        + vertical_gap**2
    )

    image_diagonal = math.sqrt(
        image_width**2
        + image_height**2
    )

    if image_diagonal == 0:
        return float("inf")

    return pixel_gap / image_diagonal


def bbox_surface_edge_gap_ratio(
    subject_bbox: list[float],
    surface_bbox: list[float],
    image_width: int,
    image_height: int,
) -> float:
    """
    Shortest distance from the subject box to an edge of the
    detected supporting surface, normalised by image diagonal.
    """

    sx1, sy1, sx2, sy2 = subject_bbox
    tx1, ty1, tx2, ty2 = surface_bbox

    image_diagonal = math.sqrt(
        image_width**2
        + image_height**2
    )

    if image_diagonal == 0:
        return float("inf")

    if (
        sx1 <= tx1 <= sx2
        or sx1 <= tx2 <= sx2
        or sy1 <= ty1 <= sy2
        or sy1 <= ty2 <= sy2
    ):
        return 0.0

    edge_gap_pixels = min(
        abs(sx1 - tx1),
        abs(tx2 - sx2),
        abs(sy1 - ty1),
        abs(ty2 - sy2),
    )

    return (
        max(0.0, edge_gap_pixels)
        / image_diagonal
    )


def apply_object_proximity_rule(
    rule: dict[str, Any],
    frames: list[dict[str, Any]],
) -> tuple[bool, list[str], float]:
    evidence_frames: list[str] = []
    supporting_confidences: list[float] = []

    for frame in frames:
        image_size = frame.get("image_size", {})
        image_width = int(
            image_size.get("width", 0)
        )
        image_height = int(
            image_size.get("height", 0)
        )

        if image_width <= 0 or image_height <= 0:
            continue

        objects = frame.get("objects", [])

        group_1_objects = [
            obj
            for obj in objects
            if object_matches(
                detected_object=obj,
                object_classes=rule[
                    "object_group_1"
                ],
                min_confidence=float(
                    rule["min_confidence"]
                ),
            )
        ]

        group_2_objects = [
            obj
            for obj in objects
            if object_matches(
                detected_object=obj,
                object_classes=rule[
                    "object_group_2"
                ],
                min_confidence=float(
                    rule["min_confidence"]
                ),
            )
        ]

        frame_triggered = False

        for object_1 in group_1_objects:
            for object_2 in group_2_objects:
                if object_1 is object_2:
                    continue

                bbox_1 = object_1.get("bbox")
                bbox_2 = object_2.get("bbox")

                if not bbox_1 or not bbox_2:
                    continue

                if (
                    bbox_gap_ratio(
                        bbox_1=bbox_1,
                        bbox_2=bbox_2,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    > float(rule["max_gap_ratio"])
                ):
                    continue

                supporting_confidences.append(
                    min(
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
                )

                frame_triggered = True

        if frame_triggered:
            evidence_frames.append(
                frame["frame"]
            )

    if not evidence_frames:
        return False, [], 0.0

    return (
        True,
        sorted(set(evidence_frames)),
        max(supporting_confidences),
    )


def apply_object_overlap_rule(
    rule: dict[str, Any],
    frames: list[dict[str, Any]],
) -> tuple[bool, list[str], float]:
    evidence_frames: list[str] = []
    supporting_confidences: list[float] = []

    for frame in frames:
        objects = frame.get("objects", [])

        subjects = [
            obj
            for obj in objects
            if object_matches(
                detected_object=obj,
                object_classes=rule[
                    "object_group_1"
                ],
                min_confidence=float(
                    rule["min_confidence"]
                ),
            )
        ]

        targets = [
            obj
            for obj in objects
            if object_matches(
                detected_object=obj,
                object_classes=rule[
                    "object_group_2"
                ],
                min_confidence=float(
                    rule["min_confidence"]
                ),
            )
        ]

        frame_triggered = False

        for subject in subjects:
            for target in targets:
                if subject is target:
                    continue

                subject_bbox = subject.get("bbox")
                target_bbox = target.get("bbox")

                if not subject_bbox or not target_bbox:
                    continue

                if (
                    bbox_subject_overlap_ratio(
                        subject_bbox=subject_bbox,
                        other_bbox=target_bbox,
                    )
                    < float(
                        rule[
                            "minimum_subject_overlap_ratio"
                        ]
                    )
                ):
                    continue

                supporting_confidences.append(
                    min(
                        float(
                            subject.get(
                                "confidence",
                                0.0,
                            )
                        ),
                        float(
                            target.get(
                                "confidence",
                                0.0,
                            )
                        ),
                    )
                )

                frame_triggered = True

        if frame_triggered:
            evidence_frames.append(
                frame["frame"]
            )

    if not evidence_frames:
        return False, [], 0.0

    return (
        True,
        sorted(set(evidence_frames)),
        max(supporting_confidences),
    )


def apply_surface_edge_proximity_rule(
    rule: dict[str, Any],
    frames: list[dict[str, Any]],
) -> tuple[bool, list[str], float]:
    evidence_frames: list[str] = []
    supporting_confidences: list[float] = []

    for frame in frames:
        image_size = frame.get("image_size", {})
        image_width = int(
            image_size.get("width", 0)
        )
        image_height = int(
            image_size.get("height", 0)
        )

        if image_width <= 0 or image_height <= 0:
            continue

        objects = frame.get("objects", [])

        subjects = [
            obj
            for obj in objects
            if object_matches(
                detected_object=obj,
                object_classes=rule[
                    "object_classes"
                ],
                min_confidence=float(
                    rule["min_confidence"]
                ),
            )
        ]

        surfaces = [
            obj
            for obj in objects
            if object_matches(
                detected_object=obj,
                object_classes=rule[
                    "surface_classes"
                ],
                min_confidence=float(
                    rule[
                        "min_surface_confidence"
                    ]
                ),
            )
        ]

        frame_triggered = False

        for subject in subjects:
            for surface in surfaces:
                if subject is surface:
                    continue

                subject_bbox = subject.get("bbox")
                surface_bbox = surface.get("bbox")

                if not subject_bbox or not surface_bbox:
                    continue

                if (
                    bbox_subject_overlap_ratio(
                        subject_bbox=subject_bbox,
                        other_bbox=surface_bbox,
                    )
                    < float(
                        rule[
                            "minimum_support_overlap_ratio"
                        ]
                    )
                ):
                    continue

                if (
                    bbox_surface_edge_gap_ratio(
                        subject_bbox=subject_bbox,
                        surface_bbox=surface_bbox,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    > float(
                        rule["max_edge_gap_ratio"]
                    )
                ):
                    continue

                supporting_confidences.append(
                    min(
                        float(
                            subject.get(
                                "confidence",
                                0.0,
                            )
                        ),
                        float(
                            surface.get(
                                "confidence",
                                0.0,
                            )
                        ),
                    )
                )

                frame_triggered = True

        if frame_triggered:
            evidence_frames.append(
                frame["frame"]
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
    rule_type = rule["type"]

    if rule_type == "object_proximity":
        return apply_object_proximity_rule(
            rule=rule,
            frames=frames,
        )

    if rule_type == "object_overlap":
        return apply_object_overlap_rule(
            rule=rule,
            frames=frames,
        )

    if rule_type == "surface_edge_proximity":
        return apply_surface_edge_proximity_rule(
            rule=rule,
            frames=frames,
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
        "confidence": round(
            confidence,
            4,
        ),
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
    rules_path: Path,
    overwrite: bool = False,
) -> None:
    if not rules_path.exists():
        raise FileNotFoundError(
            f"Rules file not found: {rules_path}"
        )

    rules_data = load_json(rules_path)
    rules = validate_ruleset(rules_data)

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
    print(f"File   : {rules_path}")
    print(
        "Version: "
        f"{rules_data.get('version', 'unknown')}"
    )
    print(
        "Frozen : "
        f"{rules_data.get('frozen_before_final_evaluation', False)}"
    )
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

        if output_json.exists():
            if not overwrite:
                skipped += 1
                continue

            # Remove stale output before reprocessing this clip.
            output_json.unlink()

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
                        hazard_index=(
                            len(hazards) + 1
                        ),
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
                        "kitchen_rules_evaluation",
                    ),
                    "version": rules_data.get(
                        "version",
                        "unknown",
                    ),
                    "filename": rules_path.name,
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
                        "The baseline is limited to the "
                        "fixed frozen ruleset and the "
                        "available YOLO detections."
                    ),
                    (
                        "It cannot reason about heat, "
                        "appliance state, hidden conditions "
                        "or hazards outside its predefined "
                        "rules."
                    ),
                ],
            }

            save_json(
                result,
                output_json,
            )

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
        "Output    : "
        f"dataset/baseline_hazards/{subset}"
    )
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen closed-set "
            "rule-based kitchen hazard baseline."
        )
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="P07",
        help=(
            "Dataset subset, for example "
            "P07 or test."
        ),
    )

    parser.add_argument(
        "--rules",
        type=str,
        default=DEFAULT_RULES_FILENAME,
        help=(
            "Rules JSON filename inside the project "
            "rules folder, a project-relative path, "
            "or an absolute path."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing baseline outputs. "
            "Stale files are removed before each "
            "clip is reprocessed."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    run_rule_based_baseline(
        subset=arguments.subset,
        rules_path=resolve_rules_path(
            arguments.rules
        ),
        overwrite=arguments.overwrite,
    )
