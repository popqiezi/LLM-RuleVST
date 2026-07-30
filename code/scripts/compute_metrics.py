#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from scripts.common.geo_utils import CoordinateTransformer


def trajectory_metrics(
    true_lonlat: np.ndarray,
    pred_lonlat: np.ndarray,
    metric_crs: str = "EPSG:32650",
) -> dict:
    true_values = np.asarray(
        true_lonlat, dtype=np.float64
    )
    pred_values = np.asarray(
        pred_lonlat, dtype=np.float64
    )
    if true_values.shape != pred_values.shape:
        raise ValueError("Trajectory arrays have different shapes")
    if true_values.ndim != 3 or true_values.shape[-1] != 2:
        raise ValueError("Expected shape (N, T, 2)")

    transformer = CoordinateTransformer(
        "EPSG:4326", metric_crs
    )
    true_x, true_y = transformer.lonlat_to_xy(
        true_values[..., 0].reshape(-1),
        true_values[..., 1].reshape(-1),
    )
    pred_x, pred_y = transformer.lonlat_to_xy(
        pred_values[..., 0].reshape(-1),
        pred_values[..., 1].reshape(-1),
    )
    distances = np.hypot(
        pred_x - true_x, pred_y - true_y
    ).reshape(true_values.shape[:2])

    return {
        "ADE_m": float(distances.mean()),
        "FDE_m": float(distances[:, -1].mean()),
        "per_sample_ADE_m": distances.mean(axis=1).tolist(),
        "per_sample_FDE_m": distances[:, -1].tolist(),
    }


def cri_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    true_values = np.asarray(
        y_true, dtype=np.float64
    ).reshape(-1)
    pred_values = np.asarray(
        y_pred, dtype=np.float64
    ).reshape(-1)
    error = pred_values - true_values
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(np.mean(error ** 2)),
    }


def high_risk_metrics(
    y_true_cri: np.ndarray,
    y_pred_cri: np.ndarray,
    threshold: float = 0.87,
) -> dict:
    true_labels = (
        np.asarray(y_true_cri).reshape(-1) >= threshold
    ).astype(np.int64)
    pred_labels = (
        np.asarray(y_pred_cri).reshape(-1) >= threshold
    ).astype(np.int64)
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average="binary",
            zero_division=0,
        )
    )
    tn, fp, fn, tp = confusion_matrix(
        true_labels, pred_labels, labels=[0, 1]
    ).ravel()
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "high_risk_count": int(true_labels.sum()),
        "non_high_risk_count": int(
            len(true_labels) - true_labels.sum()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.87)
    args = parser.parse_args()

    with np.load(args.input_npz, allow_pickle=False) as data:
        results = {}
        if {
            "true_trajectory_lonlat",
            "pred_trajectory_lonlat",
        }.issubset(data.files):
            results["trajectory"] = trajectory_metrics(
                data["true_trajectory_lonlat"],
                data["pred_trajectory_lonlat"],
            )
        if {"true_cri", "pred_cri"}.issubset(data.files):
            results["cri_regression"] = (
                cri_regression_metrics(
                    data["true_cri"], data["pred_cri"]
                )
            )
            results["high_risk_classification"] = (
                high_risk_metrics(
                    data["true_cri"],
                    data["pred_cri"],
                    args.threshold,
                )
            )

    args.output_json.parent.mkdir(
        parents=True, exist_ok=True
    )
    args.output_json.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
