#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encounter-dir", type=Path, required=True)
    parser.add_argument(
        "--rule-sequence-dir", type=Path, required=True
    )
    parser.add_argument("--cri-dir", type=Path, required=True)
    parser.add_argument(
        "--final-sample-dir", type=Path, required=True
    )
    args = parser.parse_args()

    encounters = pd.read_csv(
        args.encounter_dir / "encounter_pair_manifest.csv"
    )
    rules = pd.read_csv(
        args.rule_sequence_dir / "rule_sequence_index.csv"
    )
    cri = pd.read_csv(
        args.cri_dir / "cri_target_index.csv"
    )
    samples = pd.read_csv(
        args.final_sample_dir / "final_sample_index.csv"
    )

    if encounters["encounter_id"].duplicated().any():
        raise SystemExit("Duplicate encounter_id")
    if (
        encounters["duplicate_key"].duplicated().any()
    ):
        raise SystemExit("Duplicate ordered pair/window")
    if (encounters["final_history_distance_m"] > 1000).any():
        raise SystemExit("Encounter distance exceeds 1000 m")

    encounter_ids = set(encounters["encounter_id"])
    if not set(rules["encounter_id"]).issubset(encounter_ids):
        raise SystemExit("Rule index contains unknown encounters")
    if not set(cri["encounter_id"]).issubset(encounter_ids):
        raise SystemExit("CRI index contains unknown encounters")
    if not set(samples["encounter_id"]).issubset(encounter_ids):
        raise SystemExit("Sample index contains unknown encounters")

    checked = 0
    for row in samples.itertuples(index=False):
        path = args.final_sample_dir / row.sample_file
        if not path.exists():
            raise SystemExit(f"Missing sample file: {path}")
        with np.load(path, allow_pickle=False) as data:
            if data["historical_motion_standardized"].shape != (40, 4):
                raise SystemExit(f"Motion shape error: {row.sample_id}")
            if data["historical_rule_feature_ids"].shape != (40, 5):
                raise SystemExit(f"Rule shape error: {row.sample_id}")
            if data["future_trajectory_raw"].shape != (20, 2):
                raise SystemExit(f"Trajectory shape error: {row.sample_id}")
            if data["future_cri"].shape != (20, 1):
                raise SystemExit(f"CRI shape error: {row.sample_id}")
            if data["future_high_risk_labels"].shape != (20, 1):
                raise SystemExit(f"Label shape error: {row.sample_id}")
        checked += 1

    print(json.dumps({
        "encounter_count": len(encounters),
        "rule_sequence_count": len(rules),
        "cri_target_count": len(cri),
        "final_sample_count": len(samples),
        "sample_files_checked": checked,
        "status": "passed",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
