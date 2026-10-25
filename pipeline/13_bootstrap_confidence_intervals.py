from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "test"
    / "per_clip_summary.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "thesis_outputs"
    / "tables"
    / "bootstrap_confidence_intervals.csv"
)

N_BOOTSTRAPS = 10000
RANDOM_SEED = 42


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_metrics(
    sample: pd.DataFrame,
    prefix: str,
) -> dict[str, float]:

    tp = sample[
        f"{prefix}_tp"
    ].sum()

    fp = sample[
        f"{prefix}_fp"
    ].sum()

    fn = sample[
        f"{prefix}_fn"
    ].sum()

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    hazardous = sample[
        sample[
            "is_safe_clip"
        ] == False
    ]

    safe = sample[
        sample[
            "is_safe_clip"
        ] == True
    ]

    hazardous_detection_rate = (
        hazardous[
            f"{prefix}_detected_hazardous_clip"
        ].mean()
        if len(hazardous) > 0
        else 0.0
    )

    clip_fpr = (
        (
            safe[
                f"{prefix}_fp"
            ] > 0
        ).mean()
        if len(safe) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "hazardous_clip_detection_rate":
            hazardous_detection_rate,
        "clip_level_false_positive_rate":
            clip_fpr,
    }


def percentile_interval(
    values: list[float],
) -> tuple[float, float]:

    lower = np.percentile(
        values,
        2.5,
    )

    upper = np.percentile(
        values,
        97.5,
    )

    return (
        float(lower),
        float(upper),
    )


def main() -> None:

    data = pd.read_csv(
        INPUT_PATH
    )

    data[
        "is_safe_clip"
    ] = data[
        "is_safe_clip"
    ].astype(bool)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = {
        "vlm": {},
        "baseline": {},
    }

    metrics = [
        "precision",
        "recall",
        "f1_score",
        "hazardous_clip_detection_rate",
        "clip_level_false_positive_rate",
    ]

    for system in results:
        for metric in metrics:
            results[
                system
            ][metric] = []

    for _ in range(
        N_BOOTSTRAPS
    ):

        indices = rng.integers(
            low=0,
            high=len(data),
            size=len(data),
        )

        sample = data.iloc[
            indices
        ]

        vlm_metrics = (
            calculate_metrics(
                sample,
                "vlm",
            )
        )

        baseline_metrics = (
            calculate_metrics(
                sample,
                "baseline",
            )
        )

        for metric in metrics:

            results[
                "vlm"
            ][metric].append(
                vlm_metrics[
                    metric
                ]
            )

            results[
                "baseline"
            ][metric].append(
                baseline_metrics[
                    metric
                ]
            )

    rows = []

    for system in (
        "vlm",
        "baseline",
    ):

        for metric in metrics:

            values = results[
                system
            ][metric]

            lower, upper = (
                percentile_interval(
                    values
                )
            )

            rows.append(
                {
                    "system": system,
                    "metric": metric,
                    "bootstrap_mean":
                        round(
                            float(
                                np.mean(
                                    values
                                )
                            ),
                            4,
                        ),
                    "ci_95_lower":
                        round(
                            lower,
                            4,
                        ),
                    "ci_95_upper":
                        round(
                            upper,
                            4,
                        ),
                }
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "system",
                "metric",
                "bootstrap_mean",
                "ci_95_lower",
                "ci_95_upper",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()