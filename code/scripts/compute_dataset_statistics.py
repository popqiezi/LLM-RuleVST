#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.io_utils import load_npz_dict, write_json


FEATURE_ORDER = ["lon", "lat", "sog", "cog"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    index = pd.read_csv(args.dataset_output / "processed_feature_index.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(index["dataset"].unique()):
        subset = index[
            (index["dataset"] == dataset)
            & (index["split"] == "train")
        ]
        arrays = []
        for row in subset.itertuples(index=False):
            data = load_npz_dict(args.dataset_output / row.npz_file)
            arrays.append(data["raw_features"].astype(np.float64))

        if not arrays:
            raise RuntimeError(f"No training trajectories for {dataset}")

        matrix = np.concatenate(arrays, axis=0)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=0)

        if np.any(std <= 0):
            raise RuntimeError(f"Zero standard deviation in {dataset}")

        payload = {
            "dataset": dataset,
            "method": "z_score",
            "statistics_source": f"{dataset} training set only",
            "ddof": 0,
            "feature_order": FEATURE_ORDER,
            "number_of_training_time_steps": int(matrix.shape[0]),
            "mean": {
                name: float(value)
                for name, value in zip(FEATURE_ORDER, mean)
            },
            "std": {
                name: float(value)
                for name, value in zip(FEATURE_ORDER, std)
            },
        }

        filename = dataset.lower().replace("-", "_") + "_training_statistics.json"
        write_json(args.output_dir / filename, payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
