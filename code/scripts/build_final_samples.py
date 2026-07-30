#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_ORDER = ["lon", "lat", "sog", "cog"]


def load_statistics(
    statistics_dir: Path,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    filename = (
        dataset.lower().replace("-", "_")
        + "_training_statistics.json"
    )
    payload = json.loads(
        (statistics_dir / filename).read_text(
            encoding="utf-8"
        )
    )
    if payload["feature_order"] != FEATURE_ORDER:
        raise ValueError(
            f"Unexpected feature order for {dataset}"
        )
    mean = np.asarray(
        [payload["mean"][name] for name in FEATURE_ORDER],
        dtype=np.float64,
    )
    std = np.asarray(
        [payload["std"][name] for name in FEATURE_ORDER],
        dtype=np.float64,
    )
    return mean, std


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encounter-dir", type=Path, required=True)
    parser.add_argument(
        "--rule-sequence-dir", type=Path, required=True
    )
    parser.add_argument("--cri-dir", type=Path, required=True)
    parser.add_argument(
        "--statistics-dir", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    encounter_index = pd.read_csv(
        args.encounter_dir / "encounter_pair_manifest.csv"
    )
    rule_index = pd.read_csv(
        args.rule_sequence_dir / "rule_sequence_index.csv"
    ).set_index("encounter_id")
    cri_index = pd.read_csv(
        args.cri_dir / "cri_target_index.csv"
    ).set_index("encounter_id")

    sample_dir = args.output_dir / "sample_files"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    statistics_cache = {}

    for encounter in encounter_index.itertuples(index=False):
        if encounter.encounter_id not in rule_index.index:
            continue
        if encounter.encounter_id not in cri_index.index:
            continue

        if encounter.dataset not in statistics_cache:
            statistics_cache[encounter.dataset] = (
                load_statistics(
                    args.statistics_dir,
                    encounter.dataset,
                )
            )
        mean, std = statistics_cache[encounter.dataset]

        with np.load(
            args.encounter_dir / encounter.encounter_file,
            allow_pickle=False,
        ) as data:
            own = data["own_states_raw"].astype(np.float64)
            target = data["target_states_raw"].astype(
                np.float64
            )
            timestamps = data["timestamps_ns"].astype(
                np.int64
            )

        rule_path = (
            args.rule_sequence_dir
            / rule_index.loc[
                encounter.encounter_id,
                "rule_sequence_file",
            ]
        )
        with np.load(rule_path, allow_pickle=False) as data:
            rule_ids = data["rule_feature_ids"].astype(
                np.int64
            )
            rule_valid = data["valid_flags"].astype(
                np.bool_
            )

        cri_path = (
            args.cri_dir
            / cri_index.loc[
                encounter.encounter_id, "cri_file"
            ]
        )
        with np.load(cri_path, allow_pickle=False) as data:
            future_cri = data["future_cri"].astype(
                np.float32
            )
            future_labels = data[
                "future_high_risk_labels"
            ].astype(np.int8)

        history_length = 40
        future_length = 20
        target_standardized = (target - mean) / std

        sample_id = encounter.encounter_id
        path = (
            sample_dir
            / encounter.split
            / f"{sample_id}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            sample_id=np.asarray(sample_id),
            encounter_id=np.asarray(encounter.encounter_id),
            dataset=np.asarray(encounter.dataset),
            split=np.asarray(encounter.split),
            timestamps_ns=timestamps,
            historical_motion_raw=target[:history_length],
            historical_motion_standardized=target_standardized[
                :history_length
            ].astype(np.float32),
            historical_rule_feature_ids=rule_ids,
            historical_rule_valid_flags=rule_valid,
            future_trajectory_raw=target[
                history_length:history_length + future_length,
                :2,
            ],
            future_trajectory_standardized=target_standardized[
                history_length:history_length + future_length,
                :2,
            ].astype(np.float32),
            future_cri=future_cri.reshape(future_length, 1),
            future_high_risk_labels=future_labels.reshape(
                future_length, 1
            ),
            own_states_raw=own,
            target_states_raw=target,
            feature_order=np.asarray(FEATURE_ORDER),
        )
        rows.append({
            "sample_id": sample_id,
            "dataset": encounter.dataset,
            "split": encounter.split,
            "encounter_id": encounter.encounter_id,
            "sample_file": str(
                path.relative_to(args.output_dir)
            ),
            "motion_shape": "40x4",
            "rule_shape": "40x5",
            "trajectory_target_shape": "20x2",
            "cri_target_shape": "20x1",
            "high_risk_target_shape": "20x1",
        })

    pd.DataFrame(rows).to_csv(
        args.output_dir / "final_sample_index.csv",
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
