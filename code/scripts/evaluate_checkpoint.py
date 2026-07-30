#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.baseline_models import build_baseline
from scripts.compute_metrics import (
    cri_regression_metrics,
    high_risk_metrics,
    trajectory_metrics,
)
from scripts.train_baseline import FinalSampleDataset


def load_statistics(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray([
        payload["mean"]["lon"],
        payload["mean"]["lat"],
        payload["mean"]["sog"],
        payload["mean"]["cog"],
    ])
    std = np.asarray([
        payload["std"]["lon"],
        payload["std"]["lat"],
        payload["std"]["sog"],
        payload["std"]["cog"],
    ])
    return mean, std


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.87)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu"
    )
    model = build_baseline(
        checkpoint["model_name"],
        **checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device).eval()

    index = pd.read_csv(
        args.sample_dir / "final_sample_index.csv"
    )
    subset = index[
        (index["dataset"] == checkpoint["dataset"])
        & (index["split"] == "test")
    ]
    loader = DataLoader(
        FinalSampleDataset(subset, args.sample_dir),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    mean, std = load_statistics(args.statistics)

    true_trajectory = []
    pred_trajectory = []
    true_cri = []
    pred_cri = []

    with torch.inference_mode():
        for motion, trajectory, cri in loader:
            prediction, risk = model(
                motion.to(device)
            )
            true_standardized = trajectory.numpy()
            pred_standardized = prediction.cpu().numpy()

            true_lonlat = (
                true_standardized * std[:2]
                + mean[:2]
            )
            pred_lonlat = (
                pred_standardized * std[:2]
                + mean[:2]
            )
            true_trajectory.append(true_lonlat)
            pred_trajectory.append(pred_lonlat)
            true_cri.append(cri.numpy())
            pred_cri.append(risk.cpu().numpy())

    true_trajectory = np.concatenate(
        true_trajectory, axis=0
    )
    pred_trajectory = np.concatenate(
        pred_trajectory, axis=0
    )
    true_cri = np.concatenate(true_cri, axis=0)
    pred_cri = np.concatenate(pred_cri, axis=0)

    results = {
        "trajectory": trajectory_metrics(
            true_trajectory, pred_trajectory
        ),
        "cri_regression": cri_regression_metrics(
            true_cri, pred_cri
        ),
        "high_risk_classification": high_risk_metrics(
            true_cri, pred_cri, args.threshold
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        true_trajectory_lonlat=true_trajectory,
        pred_trajectory_lonlat=pred_trajectory,
        true_cri=true_cri,
        pred_cri=pred_cri,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
