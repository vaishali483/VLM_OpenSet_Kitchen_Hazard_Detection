from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "test"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "evaluation_summary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "thesis_outputs"
)

FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURE_DIR
        / f"{filename}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{filename}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )


def create_grouped_bar_chart(
    *,
    categories: list[str],
    vlm_values: list[float],
    baseline_values: list[float],
    title: str,
    ylabel: str,
    filename: str,
    percentage: bool = False,
) -> None:

    positions = list(
        range(
            len(categories)
        )
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    vlm_positions = [
        position - width / 2
        for position
        in positions
    ]

    baseline_positions = [
        position + width / 2
        for position
        in positions
    ]

    axis.bar(
        vlm_positions,
        vlm_values,
        width=width,
        label="VLM Open-Set Agent",
    )

    axis.bar(
        baseline_positions,
        baseline_values,
        width=width,
        label="Rule-Based Baseline",
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        categories
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    axis.legend()

    if percentage:
        axis.set_ylim(
            0,
            1.0,
        )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    save_figure(
        figure,
        filename,
    )


def create_core_metrics(
    vlm: dict,
    baseline: dict,
) -> None:

    categories = [
        "Precision",
        "Recall",
        "F1 Score",
    ]

    vlm_values = [
        vlm["precision"],
        vlm["recall"],
        vlm["f1_score"],
    ]

    baseline_values = [
        baseline["precision"],
        baseline["recall"],
        baseline["f1_score"],
    ]

    create_grouped_bar_chart(
        categories=categories,
        vlm_values=vlm_values,
        baseline_values=baseline_values,
        title=(
            "Core Hazard Detection Performance"
        ),
        ylabel="Score",
        filename="core_metrics",
        percentage=True,
    )


def create_detection_rates(
    vlm: dict,
    baseline: dict,
) -> None:

    categories = [
        "Hazardous Clip\nDetection",
        "Novel Hazard\nDetection",
        "Known Hazard\nRecall",
    ]

    vlm_values = [
        vlm[
            "hazardous_clip_detection_rate"
        ],
        vlm[
            "novel_hazard_detection_rate"
        ],
        vlm[
            "known_hazard_recall"
        ],
    ]

    baseline_values = [
        baseline[
            "hazardous_clip_detection_rate"
        ],
        baseline[
            "novel_hazard_detection_rate"
        ],
        baseline[
            "known_hazard_recall"
        ],
    ]

    create_grouped_bar_chart(
        categories=categories,
        vlm_values=vlm_values,
        baseline_values=baseline_values,
        title=(
            "Hazard Detection Rates"
        ),
        ylabel="Recall / detection rate",
        filename="detection_rates",
        percentage=True,
    )


def create_severity_recall(
    vlm: dict,
    baseline: dict,
) -> None:

    categories = [
        "Low",
        "Medium",
        "High",
    ]

    vlm_values = [
        vlm[
            "severity_recall"
        ][severity][
            "recall"
        ]
        for severity in (
            "low",
            "medium",
            "high",
        )
    ]

    baseline_values = [
        baseline[
            "severity_recall"
        ][severity][
            "recall"
        ]
        for severity in (
            "low",
            "medium",
            "high",
        )
    ]

    create_grouped_bar_chart(
        categories=categories,
        vlm_values=vlm_values,
        baseline_values=baseline_values,
        title=(
            "Recall by Ground-Truth "
            "Hazard Severity"
        ),
        ylabel="Recall",
        filename="severity_recall",
        percentage=True,
    )


def create_error_counts(
    vlm: dict,
    baseline: dict,
) -> None:

    categories = [
        "True Positives",
        "False Positives",
        "False Negatives",
    ]

    vlm_values = [
        vlm["tp"],
        vlm["fp"],
        vlm["fn"],
    ]

    baseline_values = [
        baseline["tp"],
        baseline["fp"],
        baseline["fn"],
    ]

    create_grouped_bar_chart(
        categories=categories,
        vlm_values=vlm_values,
        baseline_values=baseline_values,
        title=(
            "Hazard-Level Evaluation Counts"
        ),
        ylabel="Number of hazards",
        filename="tp_fp_fn_counts",
    )


def create_false_warning_chart(
    vlm: dict,
    baseline: dict,
) -> None:

    categories = [
        "VLM Open-Set Agent",
        "Rule-Based Baseline",
    ]

    values = [
        vlm[
            "mean_false_positives_per_clip"
        ],
        baseline[
            "mean_false_positives_per_clip"
        ],
    ]

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.bar(
        categories,
        values,
    )

    axis.set_title(
        "Mean False Warnings per Clip"
    )

    axis.set_ylabel(
        "False-positive predictions per clip"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "false_warnings_per_clip",
    )


def save_dataset_table(
    summary: dict,
) -> None:

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLE_DIR
        / "dataset_summary.csv"
    )

    severity = summary[
        "ground_truth_hazards_by_severity"
    ]

    rows = [
        (
            "Total clips",
            summary["num_clips"],
        ),
        (
            "Hazardous clips",
            summary[
                "num_hazardous_clips"
            ],
        ),
        (
            "Safe clips",
            summary[
                "num_safe_clips"
            ],
        ),
        (
            "Ground-truth hazards",
            summary[
                "num_ground_truth_hazards"
            ],
        ),
        (
            "Novel hazards",
            summary[
                "num_novel_ground_truth_hazards"
            ],
        ),
        (
            "Known hazards",
            summary[
                "num_known_ground_truth_hazards"
            ],
        ),
        (
            "Low-severity hazards",
            severity["low"],
        ),
        (
            "Medium-severity hazards",
            severity["medium"],
        ),
        (
            "High-severity hazards",
            severity["high"],
        ),
        (
            "Critical hazards",
            severity["critical"],
        ),
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Property",
                "Value",
            ]
        )

        writer.writerows(
            rows
        )

    print(
        f"Saved: {output_path}"
    )


def save_main_results_table(
    vlm: dict,
    baseline: dict,
) -> None:

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLE_DIR
        / "main_results.csv"
    )

    rows = [
        (
            "True Positives",
            vlm["tp"],
            baseline["tp"],
        ),
        (
            "False Positives",
            vlm["fp"],
            baseline["fp"],
        ),
        (
            "False Negatives",
            vlm["fn"],
            baseline["fn"],
        ),
        (
            "Precision",
            vlm["precision"],
            baseline["precision"],
        ),
        (
            "Recall",
            vlm["recall"],
            baseline["recall"],
        ),
        (
            "F1 Score",
            vlm["f1_score"],
            baseline["f1_score"],
        ),
        (
            "Hazardous Clip Detection Rate",
            vlm[
                "hazardous_clip_detection_rate"
            ],
            baseline[
                "hazardous_clip_detection_rate"
            ],
        ),
        (
            "Clip-Level False Positive Rate",
            vlm[
                "clip_level_false_positive_rate"
            ],
            baseline[
                "clip_level_false_positive_rate"
            ],
        ),
        (
            "Mean False Warnings per Clip",
            vlm[
                "mean_false_positives_per_clip"
            ],
            baseline[
                "mean_false_positives_per_clip"
            ],
        ),
        (
            "Novel Hazard Detection Rate",
            vlm[
                "novel_hazard_detection_rate"
            ],
            baseline[
                "novel_hazard_detection_rate"
            ],
        ),
        (
            "Known Hazard Recall",
            vlm[
                "known_hazard_recall"
            ],
            baseline[
                "known_hazard_recall"
            ],
        ),
        (
            "High/Critical Hazard Recall",
            vlm[
                "high_critical_hazard_recall"
            ]["recall"],
            baseline[
                "high_critical_hazard_recall"
            ]["recall"],
        ),
        (
            "Medium/Low Hazard Recall",
            vlm[
                "medium_low_hazard_recall"
            ]["recall"],
            baseline[
                "medium_low_hazard_recall"
            ]["recall"],
        ),
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Metric",
                "VLM Open-Set Agent",
                "Rule-Based Baseline",
            ]
        )

        writer.writerows(
            rows
        )

    print(
        f"Saved: {output_path}"
    )


def main() -> None:

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation summary not found: "
            f"{SUMMARY_PATH}"
        )

    summary = load_json(
        SUMMARY_PATH
    )

    vlm = summary[
        "vlm_open_set"
    ]

    baseline = summary[
        "rule_based_baseline"
    ]

    create_core_metrics(
        vlm,
        baseline,
    )

    create_detection_rates(
        vlm,
        baseline,
    )

    create_severity_recall(
        vlm,
        baseline,
    )

    create_error_counts(
        vlm,
        baseline,
    )

    create_false_warning_chart(
        vlm,
        baseline,
    )

    save_dataset_table(
        summary
    )

    save_main_results_table(
        vlm,
        baseline,
    )

    print(
        "\nThesis outputs created successfully."
    )

    print(
        f"Figures: {FIGURE_DIR}"
    )

    print(
        f"Tables : {TABLE_DIR}"
    )


if __name__ == "__main__":
    main()