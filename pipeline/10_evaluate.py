from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from config import *


SEVERITY_LEVELS = (
    "low",
    "medium",
    "high",
    "critical",
)


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


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def rounded_ratio(numerator: float, denominator: float) -> float:
    return round(safe_divide(numerator, denominator), 4)


def calculate_scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def get_prediction_ids(predictions: dict[str, Any]) -> set[str]:
    return {
        hazard.get("hazard_id", "")
        for hazard in predictions.get("hazards", [])
        if hazard.get("hazard_id")
    }


def get_ground_truth_ids(ground_truth: dict[str, Any]) -> set[str]:
    return {
        hazard.get("ground_truth_id", "")
        for hazard in ground_truth.get("ground_truth_hazards", [])
        if hazard.get("ground_truth_id")
    }


def get_ground_truth_hazards_by_id(
    ground_truth: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    hazards_by_id: dict[str, dict[str, Any]] = {}

    for hazard in ground_truth.get("ground_truth_hazards", []):
        ground_truth_id = hazard.get("ground_truth_id")

        if not ground_truth_id:
            continue

        if ground_truth_id in hazards_by_id:
            raise ValueError(
                "Duplicate ground-truth ID in one file: "
                f"{ground_truth_id}"
            )

        hazards_by_id[ground_truth_id] = hazard

    return hazards_by_id


def validate_global_ids(
    *,
    new_ids: set[str],
    existing_ids: set[str],
    source_name: str,
    clip_name: str,
) -> None:
    duplicates = new_ids & existing_ids

    if duplicates:
        raise ValueError(
            f"Duplicate {source_name} IDs across clips. "
            f"Clip: {clip_name}. IDs: {sorted(duplicates)}"
        )


def validate_required_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}"
        )


def get_true_matches(
    *,
    match_items: list[dict[str, Any]],
    prediction_ids: set[str],
    ground_truth_ids: set[str],
    clip_name: str,
    system_name: str,
) -> tuple[set[str], set[str]]:
    matched_predictions: set[str] = set()
    matched_ground_truth: set[str] = set()

    for index, item in enumerate(match_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"{clip_name} {system_name} match #{index} "
                "is not a JSON object."
            )

        if not item.get("is_match", False):
            continue

        prediction_id = item.get("prediction_id")
        ground_truth_id = item.get("ground_truth_id")

        if not prediction_id:
            raise ValueError(
                f"{clip_name} {system_name} match #{index} "
                "has no prediction_id."
            )

        if not ground_truth_id:
            raise ValueError(
                f"{clip_name} {system_name} match #{index} "
                "has no ground_truth_id."
            )

        if prediction_id not in prediction_ids:
            raise ValueError(
                f"{clip_name} {system_name} match references "
                f"an unknown prediction_id: {prediction_id}"
            )

        if ground_truth_id not in ground_truth_ids:
            raise ValueError(
                f"{clip_name} {system_name} match references "
                f"an unknown ground_truth_id: {ground_truth_id}"
            )

        if prediction_id in matched_predictions:
            raise ValueError(
                f"{clip_name} {system_name} matches the same "
                f"prediction more than once: {prediction_id}"
            )

        if ground_truth_id in matched_ground_truth:
            raise ValueError(
                f"{clip_name} {system_name} matches the same "
                f"ground truth more than once: {ground_truth_id}"
            )

        matched_predictions.add(prediction_id)
        matched_ground_truth.add(ground_truth_id)

    return matched_predictions, matched_ground_truth


def evaluate_system(
    *,
    system_name: str,
    prediction_ids: set[str],
    ground_truth_ids: set[str],
    matched_predictions: set[str],
    matched_ground_truth: set[str],
) -> dict[str, Any]:
    tp = len(matched_predictions)
    fp = len(prediction_ids - matched_predictions)
    fn = len(ground_truth_ids - matched_ground_truth)

    scores = calculate_scores(tp, fp, fn)

    return {
        "system": system_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **scores,
    }


def build_severity_recall(
    *,
    severity_gt_ids: dict[str, set[str]],
    matched_ground_truth: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for severity in SEVERITY_LEVELS:
        ground_truth_ids = severity_gt_ids[severity]
        detected_ids = matched_ground_truth & ground_truth_ids

        result[severity] = {
            "detected": len(detected_ids),
            "total": len(ground_truth_ids),
            "recall": rounded_ratio(
                len(detected_ids),
                len(ground_truth_ids),
            ),
        }

    return result


def build_grouped_severity_metric(
    *,
    severities: tuple[str, ...],
    severity_gt_ids: dict[str, set[str]],
    matched_ground_truth: set[str],
) -> dict[str, Any]:
    relevant_gt_ids: set[str] = set()

    for severity in severities:
        relevant_gt_ids.update(severity_gt_ids[severity])

    detected_ids = matched_ground_truth & relevant_gt_ids

    return {
        "detected": len(detected_ids),
        "total": len(relevant_gt_ids),
        "recall": rounded_ratio(
            len(detected_ids),
            len(relevant_gt_ids),
        ),
    }


def print_metric(label: str, value: Any) -> None:
    print(f"{label:<38}: {value}")


def print_system_report(
    *,
    title: str,
    result: dict[str, Any],
) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)

    print_metric("True Positives", result["tp"])
    print_metric("False Positives", result["fp"])
    print_metric("False Negatives", result["fn"])
    print_metric("Precision", f"{result['precision']:.4f}")
    print_metric("Recall", f"{result['recall']:.4f}")
    print_metric("F1 Score", f"{result['f1_score']:.4f}")

    print_metric(
        "Hazardous Clip Detection Rate",
        (
            f"{result['hazardous_clip_detection_rate']:.4f} "
            f"({result['hazardous_clips_detected']}/"
            f"{result['total_hazardous_clips']})"
        ),
    )

    print_metric(
        "Clip-Level False Positive Rate",
        (
            f"{result['clip_level_false_positive_rate']:.4f} "
            f"({result['false_positive_safe_clips']}/"
            f"{result['total_safe_clips']})"
        ),
    )

    print_metric(
        "Mean False Warnings per Clip",
        f"{result['mean_false_positives_per_clip']:.4f}",
    )

    print_metric(
        "Novel Hazard Detection Rate",
        (
            f"{result['novel_hazard_detection_rate']:.4f} "
            f"({result['detected_novel_hazards']}/"
            f"{result['total_novel_hazards']})"
        ),
    )

    print_metric(
        "Known Hazard Recall",
        (
            f"{result['known_hazard_recall']:.4f} "
            f"({result['detected_known_hazards']}/"
            f"{result['total_known_hazards']})"
        ),
    )

    high_critical = result["high_critical_hazard_recall"]
    medium_low = result["medium_low_hazard_recall"]

    print_metric(
        "High/Critical Hazard Recall",
        (
            f"{high_critical['recall']:.4f} "
            f"({high_critical['detected']}/{high_critical['total']})"
        ),
    )

    print_metric(
        "Medium/Low Hazard Recall",
        (
            f"{medium_low['recall']:.4f} "
            f"({medium_low['detected']}/{medium_low['total']})"
        ),
    )

    print("\nSeverity Recall")

    for severity in SEVERITY_LEVELS:
        metric = result["severity_recall"][severity]

        print_metric(
            f"  {severity.capitalize()}",
            (
                f"{metric['recall']:.4f} "
                f"({metric['detected']}/{metric['total']})"
            ),
        )


def evaluate_subset(subset: str) -> None:
    gt_root = DATASET_DIR / "ground_truth" / subset
    vlm_root = DATASET_DIR / "hazard_analysis" / subset
    baseline_root = DATASET_DIR / "baseline_hazards" / subset
    match_root = PROJECT_ROOT / "evaluation" / "matching" / subset
    result_root = PROJECT_ROOT / "evaluation" / "results" / subset

    if not gt_root.exists():
        raise FileNotFoundError(
            f"Ground-truth subset folder not found: {gt_root}"
        )

    if not vlm_root.exists():
        raise FileNotFoundError(
            f"VLM hazard-analysis subset folder not found: {vlm_root}"
        )

    if not baseline_root.exists():
        raise FileNotFoundError(
            f"Baseline subset folder not found: {baseline_root}"
        )

    if not match_root.exists():
        raise FileNotFoundError(
            f"Matching subset folder not found: {match_root}"
        )

    gt_files = sorted(gt_root.rglob("ground_truth.json"))

    if not gt_files:
        raise RuntimeError(
            f"No ground-truth files found for subset: {subset}"
        )

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Subset : {subset}")
    print(f"Found  : {len(gt_files)} ground-truth files")
    print("=" * 60 + "\n")

    total_gt_ids: set[str] = set()

    vlm_prediction_ids: set[str] = set()
    baseline_prediction_ids: set[str] = set()

    vlm_matched_predictions: set[str] = set()
    vlm_matched_ground_truth: set[str] = set()

    baseline_matched_predictions: set[str] = set()
    baseline_matched_ground_truth: set[str] = set()

    novel_gt_ids_all: set[str] = set()
    known_gt_ids_all: set[str] = set()

    severity_gt_ids: dict[str, set[str]] = {
        severity: set()
        for severity in SEVERITY_LEVELS
    }

    safe_clips = 0
    hazardous_clips = 0

    vlm_false_positive_safe_clips = 0
    baseline_false_positive_safe_clips = 0

    vlm_hazardous_clips_detected = 0
    baseline_hazardous_clips_detected = 0

    suspicious_unreviewed_templates: list[str] = []
    per_clip_rows: list[dict[str, Any]] = []

    for gt_file in gt_files:
        clip_name = gt_file.parent.name
        gt = load_json(gt_file)

        gt_hazards_by_id = get_ground_truth_hazards_by_id(gt)
        gt_ids = set(gt_hazards_by_id.keys())

        validate_global_ids(
            new_ids=gt_ids,
            existing_ids=total_gt_ids,
            source_name="ground-truth",
            clip_name=clip_name,
        )

        total_gt_ids.update(gt_ids)

        clip_novel_gt_ids: set[str] = set()
        clip_known_gt_ids: set[str] = set()

        for ground_truth_id, hazard in gt_hazards_by_id.items():
            is_novel = bool(
                hazard.get(
                    "is_novel_relative_to_rule_baseline",
                    False,
                )
            )

            if is_novel:
                clip_novel_gt_ids.add(ground_truth_id)
                novel_gt_ids_all.add(ground_truth_id)
            else:
                clip_known_gt_ids.add(ground_truth_id)
                known_gt_ids_all.add(ground_truth_id)

            severity = str(
                hazard.get("severity", "")
            ).strip().lower()

            if severity not in SEVERITY_LEVELS:
                raise ValueError(
                    f"{clip_name} ground truth {ground_truth_id} "
                    f"has unsupported severity: {severity!r}. "
                    f"Expected one of: {', '.join(SEVERITY_LEVELS)}"
                )

            severity_gt_ids[severity].add(ground_truth_id)

        vlm_path = vlm_root / clip_name / "hazard_analysis.json"
        baseline_path = baseline_root / clip_name / "baseline_hazards.json"
        match_path = match_root / clip_name / "matches.json"

        validate_required_file(
            vlm_path,
            f"VLM output for {clip_name}",
        )
        validate_required_file(
            baseline_path,
            f"baseline output for {clip_name}",
        )
        validate_required_file(
            match_path,
            f"matching file for {clip_name}",
        )

        vlm = load_json(vlm_path)
        baseline = load_json(baseline_path)
        matches = load_json(match_path)

        vlm_ids = get_prediction_ids(vlm)
        baseline_ids = get_prediction_ids(baseline)

        validate_global_ids(
            new_ids=vlm_ids,
            existing_ids=vlm_prediction_ids,
            source_name="VLM prediction",
            clip_name=clip_name,
        )
        validate_global_ids(
            new_ids=baseline_ids,
            existing_ids=baseline_prediction_ids,
            source_name="baseline prediction",
            clip_name=clip_name,
        )

        vlm_prediction_ids.update(vlm_ids)
        baseline_prediction_ids.update(baseline_ids)

        vlm_match_items = matches.get("vlm_matches", [])
        baseline_match_items = matches.get("baseline_matches", [])

        if (
            gt_ids
            and (vlm_ids or baseline_ids)
            and not vlm_match_items
            and not baseline_match_items
        ):
            suspicious_unreviewed_templates.append(clip_name)

        (
            vlm_clip_matched_pred,
            vlm_clip_matched_gt,
        ) = get_true_matches(
            match_items=vlm_match_items,
            prediction_ids=vlm_ids,
            ground_truth_ids=gt_ids,
            clip_name=clip_name,
            system_name="VLM",
        )

        (
            baseline_clip_matched_pred,
            baseline_clip_matched_gt,
        ) = get_true_matches(
            match_items=baseline_match_items,
            prediction_ids=baseline_ids,
            ground_truth_ids=gt_ids,
            clip_name=clip_name,
            system_name="baseline",
        )

        vlm_matched_predictions.update(vlm_clip_matched_pred)
        vlm_matched_ground_truth.update(vlm_clip_matched_gt)

        baseline_matched_predictions.update(baseline_clip_matched_pred)
        baseline_matched_ground_truth.update(baseline_clip_matched_gt)

        is_safe_clip = len(gt_ids) == 0
        is_hazardous_clip = not is_safe_clip

        if is_safe_clip:
            safe_clips += 1

            if vlm_ids:
                vlm_false_positive_safe_clips += 1

            if baseline_ids:
                baseline_false_positive_safe_clips += 1
        else:
            hazardous_clips += 1

            if vlm_clip_matched_gt:
                vlm_hazardous_clips_detected += 1

            if baseline_clip_matched_gt:
                baseline_hazardous_clips_detected += 1

        vlm_clip_tp = len(vlm_clip_matched_pred)
        vlm_clip_fp = len(vlm_ids - vlm_clip_matched_pred)
        vlm_clip_fn = len(gt_ids - vlm_clip_matched_gt)

        baseline_clip_tp = len(baseline_clip_matched_pred)
        baseline_clip_fp = len(baseline_ids - baseline_clip_matched_pred)
        baseline_clip_fn = len(gt_ids - baseline_clip_matched_gt)

        per_clip_rows.append(
            {
                "clip_id": clip_name,
                "is_safe_clip": is_safe_clip,
                "ground_truth_hazards": len(gt_ids),
                "novel_ground_truth_hazards": len(clip_novel_gt_ids),
                "known_ground_truth_hazards": len(clip_known_gt_ids),
                "vlm_predictions": len(vlm_ids),
                "vlm_tp": vlm_clip_tp,
                "vlm_fp": vlm_clip_fp,
                "vlm_fn": vlm_clip_fn,
                "vlm_detected_hazardous_clip": (
                    bool(vlm_clip_matched_gt)
                    if is_hazardous_clip
                    else False
                ),
                "baseline_predictions": len(baseline_ids),
                "baseline_tp": baseline_clip_tp,
                "baseline_fp": baseline_clip_fp,
                "baseline_fn": baseline_clip_fn,
                "baseline_detected_hazardous_clip": (
                    bool(baseline_clip_matched_gt)
                    if is_hazardous_clip
                    else False
                ),
            }
        )

    vlm_core_result = evaluate_system(
        system_name="vlm_open_set",
        prediction_ids=vlm_prediction_ids,
        ground_truth_ids=total_gt_ids,
        matched_predictions=vlm_matched_predictions,
        matched_ground_truth=vlm_matched_ground_truth,
    )

    baseline_core_result = evaluate_system(
        system_name="rule_based_baseline",
        prediction_ids=baseline_prediction_ids,
        ground_truth_ids=total_gt_ids,
        matched_predictions=baseline_matched_predictions,
        matched_ground_truth=baseline_matched_ground_truth,
    )

    vlm_detected_novel_gt = vlm_matched_ground_truth & novel_gt_ids_all
    baseline_detected_novel_gt = (
        baseline_matched_ground_truth & novel_gt_ids_all
    )

    vlm_detected_known_gt = vlm_matched_ground_truth & known_gt_ids_all
    baseline_detected_known_gt = (
        baseline_matched_ground_truth & known_gt_ids_all
    )

    vlm_severity_recall = build_severity_recall(
        severity_gt_ids=severity_gt_ids,
        matched_ground_truth=vlm_matched_ground_truth,
    )

    baseline_severity_recall = build_severity_recall(
        severity_gt_ids=severity_gt_ids,
        matched_ground_truth=baseline_matched_ground_truth,
    )

    vlm_high_critical = build_grouped_severity_metric(
        severities=("high", "critical"),
        severity_gt_ids=severity_gt_ids,
        matched_ground_truth=vlm_matched_ground_truth,
    )

    baseline_high_critical = build_grouped_severity_metric(
        severities=("high", "critical"),
        severity_gt_ids=severity_gt_ids,
        matched_ground_truth=baseline_matched_ground_truth,
    )

    vlm_medium_low = build_grouped_severity_metric(
        severities=("medium", "low"),
        severity_gt_ids=severity_gt_ids,
        matched_ground_truth=vlm_matched_ground_truth,
    )

    baseline_medium_low = build_grouped_severity_metric(
        severities=("medium", "low"),
        severity_gt_ids=severity_gt_ids,
        matched_ground_truth=baseline_matched_ground_truth,
    )

    vlm_result = {
        **vlm_core_result,
        "hazardous_clip_detection_rate": rounded_ratio(
            vlm_hazardous_clips_detected,
            hazardous_clips,
        ),
        "hazardous_clips_detected": vlm_hazardous_clips_detected,
        "total_hazardous_clips": hazardous_clips,
        "clip_level_false_positive_rate": rounded_ratio(
            vlm_false_positive_safe_clips,
            safe_clips,
        ),
        "false_positive_safe_clips": vlm_false_positive_safe_clips,
        "total_safe_clips": safe_clips,
        "mean_false_positives_per_clip": rounded_ratio(
            vlm_core_result["fp"],
            len(gt_files),
        ),
        "novel_hazard_detection_rate": rounded_ratio(
            len(vlm_detected_novel_gt),
            len(novel_gt_ids_all),
        ),
        "detected_novel_hazards": len(vlm_detected_novel_gt),
        "total_novel_hazards": len(novel_gt_ids_all),
        "known_hazard_recall": rounded_ratio(
            len(vlm_detected_known_gt),
            len(known_gt_ids_all),
        ),
        "detected_known_hazards": len(vlm_detected_known_gt),
        "total_known_hazards": len(known_gt_ids_all),
        "severity_recall": vlm_severity_recall,
        "high_critical_hazard_recall": vlm_high_critical,
        "medium_low_hazard_recall": vlm_medium_low,
    }

    baseline_result = {
        **baseline_core_result,
        "hazardous_clip_detection_rate": rounded_ratio(
            baseline_hazardous_clips_detected,
            hazardous_clips,
        ),
        "hazardous_clips_detected": baseline_hazardous_clips_detected,
        "total_hazardous_clips": hazardous_clips,
        "clip_level_false_positive_rate": rounded_ratio(
            baseline_false_positive_safe_clips,
            safe_clips,
        ),
        "false_positive_safe_clips": baseline_false_positive_safe_clips,
        "total_safe_clips": safe_clips,
        "mean_false_positives_per_clip": rounded_ratio(
            baseline_core_result["fp"],
            len(gt_files),
        ),
        "novel_hazard_detection_rate": rounded_ratio(
            len(baseline_detected_novel_gt),
            len(novel_gt_ids_all),
        ),
        "detected_novel_hazards": len(baseline_detected_novel_gt),
        "total_novel_hazards": len(novel_gt_ids_all),
        "known_hazard_recall": rounded_ratio(
            len(baseline_detected_known_gt),
            len(known_gt_ids_all),
        ),
        "detected_known_hazards": len(baseline_detected_known_gt),
        "total_known_hazards": len(known_gt_ids_all),
        "severity_recall": baseline_severity_recall,
        "high_critical_hazard_recall": baseline_high_critical,
        "medium_low_hazard_recall": baseline_medium_low,
    }

    summary = {
        "subset": subset,
        "num_clips": len(gt_files),
        "num_ground_truth_hazards": len(total_gt_ids),
        "num_safe_clips": safe_clips,
        "num_hazardous_clips": hazardous_clips,
        "num_novel_ground_truth_hazards": len(novel_gt_ids_all),
        "num_known_ground_truth_hazards": len(known_gt_ids_all),
        "ground_truth_hazards_by_severity": {
            severity: len(severity_gt_ids[severity])
            for severity in SEVERITY_LEVELS
        },
        "vlm_open_set": vlm_result,
        "rule_based_baseline": baseline_result,
        "matching_warnings": {
            "templates_with_ground_truth_and_predictions_but_no_matches": (
                suspicious_unreviewed_templates
            ),
            "note": (
                "An empty match list can be legitimate. "
                "This warning only identifies files worth "
                "double-checking before final reporting."
            ),
        },
        "per_clip": per_clip_rows,
    }

    result_root.mkdir(parents=True, exist_ok=True)

    summary_path = result_root / "evaluation_summary.json"
    save_json(summary, summary_path)

    per_clip_csv_path = result_root / "per_clip_summary.csv"

    per_clip_fieldnames = [
        "clip_id",
        "is_safe_clip",
        "ground_truth_hazards",
        "novel_ground_truth_hazards",
        "known_ground_truth_hazards",
        "vlm_predictions",
        "vlm_tp",
        "vlm_fp",
        "vlm_fn",
        "vlm_detected_hazardous_clip",
        "baseline_predictions",
        "baseline_tp",
        "baseline_fp",
        "baseline_fn",
        "baseline_detected_hazardous_clip",
    ]

    with per_clip_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=per_clip_fieldnames,
        )
        writer.writeheader()
        writer.writerows(per_clip_rows)

    comparison_csv_path = result_root / "system_comparison.csv"

    comparison_rows = []

    for result in (vlm_result, baseline_result):
        comparison_rows.append(
            {
                "system": result["system"],
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "hazardous_clip_detection_rate": result[
                    "hazardous_clip_detection_rate"
                ],
                "clip_level_false_positive_rate": result[
                    "clip_level_false_positive_rate"
                ],
                "mean_false_positives_per_clip": result[
                    "mean_false_positives_per_clip"
                ],
                "novel_hazard_detection_rate": result[
                    "novel_hazard_detection_rate"
                ],
                "known_hazard_recall": result[
                    "known_hazard_recall"
                ],
                "high_critical_hazard_recall": result[
                    "high_critical_hazard_recall"
                ]["recall"],
                "medium_low_hazard_recall": result[
                    "medium_low_hazard_recall"
                ]["recall"],
                "low_recall": result["severity_recall"]["low"]["recall"],
                "medium_recall": result["severity_recall"]["medium"]["recall"],
                "high_recall": result["severity_recall"]["high"]["recall"],
                "critical_recall": result[
                    "severity_recall"
                ]["critical"]["recall"],
            }
        )

    comparison_fieldnames = list(comparison_rows[0].keys())

    with comparison_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=comparison_fieldnames,
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results JSON  : {summary_path}")
    print(f"Per-clip CSV  : {per_clip_csv_path}")
    print(f"Comparison CSV: {comparison_csv_path}")
    print("=" * 60)

    print_system_report(
        title="VLM OPEN-SET",
        result=vlm_result,
    )

    print_system_report(
        title="RULE-BASED BASELINE",
        result=baseline_result,
    )

    if suspicious_unreviewed_templates:
        print("\n" + "!" * 60)
        print("MATCHING REVIEW WARNING")
        print("!" * 60)
        print(
            "The following clips have ground-truth hazards and "
            "predictions, but both match lists are empty:"
        )

        for clip_name in suspicious_unreviewed_templates:
            print(f"  - {clip_name}")

        print(
            "\nAn empty match list can be correct, but verify these "
            "files before using the numbers in the dissertation."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the VLM open-set system and rule-based "
            "baseline using manual one-to-one hazard matches."
        )
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="P07",
        help="Dataset subset, for example P07 or test.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    evaluate_subset(subset=arguments.subset)
