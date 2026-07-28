import argparse
import csv
import json
from pathlib import Path
from typing import Any

from config import *


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }


def get_prediction_ids(predictions: dict[str, Any]) -> set[str]:
    return {
        hazard.get("hazard_id", "")
        for hazard in predictions.get("hazards", [])
        if hazard.get("hazard_id")
    }


def get_ground_truth_ids(gt: dict[str, Any]) -> set[str]:
    return {
        hazard.get("ground_truth_id", "")
        for hazard in gt.get("ground_truth_hazards", [])
        if hazard.get("ground_truth_id")
    }


def get_true_matches(
    match_items: list[dict[str, Any]],
    prediction_ids: set[str],
    ground_truth_ids: set[str]
) -> tuple[set[str], set[str]]:

    matched_predictions = set()
    matched_ground_truth = set()

    for item in match_items:
        if not item.get("is_match", False):
            continue

        prediction_id = item.get("prediction_id")
        ground_truth_id = item.get("ground_truth_id")

        if prediction_id not in prediction_ids:
            continue

        if ground_truth_id not in ground_truth_ids:
            continue

        if prediction_id in matched_predictions:
            continue

        if ground_truth_id in matched_ground_truth:
            continue

        matched_predictions.add(prediction_id)
        matched_ground_truth.add(ground_truth_id)

    return matched_predictions, matched_ground_truth


def evaluate_system(
    system_name: str,
    prediction_ids: set[str],
    ground_truth_ids: set[str],
    matched_predictions: set[str],
    matched_ground_truth: set[str]
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
        **scores
    }


def evaluate_subset(subset: str) -> None:

    gt_root = DATASET_DIR / "ground_truth" / subset
    vlm_root = DATASET_DIR / "hazard_analysis" / subset
    baseline_root = DATASET_DIR / "baseline_hazards" / subset
    match_root = PROJECT_ROOT / "evaluation" / "matching" / subset
    result_root = PROJECT_ROOT / "evaluation" / "results" / subset

    gt_files = sorted(gt_root.rglob("ground_truth.json"))

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Subset : {subset}")
    print(f"Found  : {len(gt_files)} ground-truth files")
    print("=" * 60 + "\n")

    total_gt_ids = set()

    vlm_prediction_ids = set()
    baseline_prediction_ids = set()

    vlm_matched_predictions = set()
    vlm_matched_ground_truth = set()

    baseline_matched_predictions = set()
    baseline_matched_ground_truth = set()

    safe_clips = 0
    vlm_false_positive_safe_clips = 0
    baseline_false_positive_safe_clips = 0

    total_novel_gt = 0
    vlm_detected_novel_gt = set()
    baseline_detected_novel_gt = set()

    per_clip_rows = []

    for gt_file in gt_files:

        clip_name = gt_file.parent.name

        gt = load_json(gt_file)

        gt_ids = get_ground_truth_ids(gt)

        total_gt_ids.update(gt_ids)

        novel_gt_ids = {
            hazard["ground_truth_id"]
            for hazard in gt.get("ground_truth_hazards", [])
            if hazard.get("is_novel_relative_to_rule_baseline", False)
        }

        total_novel_gt += len(novel_gt_ids)

        vlm_path = vlm_root / clip_name / "hazard_analysis.json"
        baseline_path = baseline_root / clip_name / "baseline_hazards.json"
        match_path = match_root / clip_name / "matches.json"

        vlm = {"hazards": []}
        baseline = {"hazards": []}
        matches = {
            "vlm_matches": [],
            "baseline_matches": []
        }

        if vlm_path.exists():
            vlm = load_json(vlm_path)

        if baseline_path.exists():
            baseline = load_json(baseline_path)

        if match_path.exists():
            matches = load_json(match_path)

        vlm_ids = get_prediction_ids(vlm)
        baseline_ids = get_prediction_ids(baseline)

        vlm_prediction_ids.update(vlm_ids)
        baseline_prediction_ids.update(baseline_ids)

        vlm_clip_matched_pred, vlm_clip_matched_gt = get_true_matches(
            match_items=matches.get("vlm_matches", []),
            prediction_ids=vlm_ids,
            ground_truth_ids=gt_ids
        )

        baseline_clip_matched_pred, baseline_clip_matched_gt = get_true_matches(
            match_items=matches.get("baseline_matches", []),
            prediction_ids=baseline_ids,
            ground_truth_ids=gt_ids
        )

        vlm_matched_predictions.update(vlm_clip_matched_pred)
        vlm_matched_ground_truth.update(vlm_clip_matched_gt)

        baseline_matched_predictions.update(baseline_clip_matched_pred)
        baseline_matched_ground_truth.update(baseline_clip_matched_gt)

        vlm_detected_novel_gt.update(vlm_clip_matched_gt & novel_gt_ids)
        baseline_detected_novel_gt.update(baseline_clip_matched_gt & novel_gt_ids)

        is_safe_clip = len(gt_ids) == 0

        if is_safe_clip:
            safe_clips += 1

            if len(vlm_ids) > 0:
                vlm_false_positive_safe_clips += 1

            if len(baseline_ids) > 0:
                baseline_false_positive_safe_clips += 1

        per_clip_rows.append({
            "clip_id": clip_name,
            "ground_truth_hazards": len(gt_ids),
            "vlm_predictions": len(vlm_ids),
            "baseline_predictions": len(baseline_ids),
            "vlm_matches": len(vlm_clip_matched_pred),
            "baseline_matches": len(baseline_clip_matched_pred)
        })

    vlm_result = evaluate_system(
        system_name="vlm_open_set",
        prediction_ids=vlm_prediction_ids,
        ground_truth_ids=total_gt_ids,
        matched_predictions=vlm_matched_predictions,
        matched_ground_truth=vlm_matched_ground_truth
    )

    baseline_result = evaluate_system(
        system_name="rule_based_baseline",
        prediction_ids=baseline_prediction_ids,
        ground_truth_ids=total_gt_ids,
        matched_predictions=baseline_matched_predictions,
        matched_ground_truth=baseline_matched_ground_truth
    )

    vlm_clip_fpr = safe_divide(
        vlm_false_positive_safe_clips,
        safe_clips
    )

    baseline_clip_fpr = safe_divide(
        baseline_false_positive_safe_clips,
        safe_clips
    )

    vlm_novel_detection_rate = safe_divide(
        len(vlm_detected_novel_gt),
        total_novel_gt
    )

    baseline_novel_detection_rate = safe_divide(
        len(baseline_detected_novel_gt),
        total_novel_gt
    )

    summary = {
        "subset": subset,
        "num_clips": len(gt_files),
        "num_ground_truth_hazards": len(total_gt_ids),
        "num_safe_clips": safe_clips,
        "vlm_open_set": {
            **vlm_result,
            "clip_level_false_positive_rate": round(vlm_clip_fpr, 4),
            "novel_hazard_detection_rate": round(vlm_novel_detection_rate, 4),
            "detected_novel_hazards": len(vlm_detected_novel_gt),
            "total_novel_hazards": total_novel_gt
        },
        "rule_based_baseline": {
            **baseline_result,
            "clip_level_false_positive_rate": round(baseline_clip_fpr, 4),
            "novel_hazard_detection_rate": round(baseline_novel_detection_rate, 4),
            "detected_novel_hazards": len(baseline_detected_novel_gt),
            "total_novel_hazards": total_novel_gt
        },
        "per_clip": per_clip_rows
    }

    result_root.mkdir(parents=True, exist_ok=True)

    save_json(summary, result_root / "evaluation_summary.json")

    csv_path = result_root / "per_clip_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "clip_id",
            "ground_truth_hazards",
            "vlm_predictions",
            "baseline_predictions",
            "vlm_matches",
            "baseline_matches"
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in per_clip_rows:
            writer.writerow(row)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results JSON : {result_root / 'evaluation_summary.json'}")
    print(f"Per-clip CSV : {csv_path}")
    print("=" * 60)

    print("\nVLM Open-Set:")
    print(vlm_result)
    print(f"Novel Hazard Detection Rate: {round(vlm_novel_detection_rate, 4)}")
    print(f"Clip-level FPR: {round(vlm_clip_fpr, 4)}")

    print("\nRule-Based Baseline:")
    print(baseline_result)
    print(f"Novel Hazard Detection Rate: {round(baseline_novel_detection_rate, 4)}")
    print(f"Clip-level FPR: {round(baseline_clip_fpr, 4)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        type=str,
        default="P07"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_subset(subset=args.subset)