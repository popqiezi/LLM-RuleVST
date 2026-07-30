#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def read_csv(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    manifest = read_csv(args.output_dir / "trajectory_split_manifest.csv")
    windows = read_csv(args.output_dir / "sliding_window_index.csv.gz")
    processed = read_csv(args.output_dir / "processed_feature_index.csv")

    ids = [r["trajectory_id"] for r in manifest]
    if len(ids) != len(set(ids)):
        raise SystemExit("trajectory_id is not unique")

    split_by_id = {r["trajectory_id"]: r["split"] for r in manifest}
    length_by_id = {r["trajectory_id"]: int(r["num_time_steps"]) for r in manifest}

    for row in manifest:
        if int(row["num_time_steps"]) < 60:
            raise SystemExit(f"trajectory shorter than 60: {row['trajectory_id']}")
        if row["split_before_window_generation"] != "True":
            raise SystemExit(f"partition order error: {row['trajectory_id']}")
        if row["native_sampling_retained"] != "True":
            raise SystemExit(f"sampling flag error: {row['trajectory_id']}")

    for row in windows:
        tid = row["trajectory_id"]
        if tid not in split_by_id:
            raise SystemExit(f"window has unknown trajectory: {tid}")
        if row["split"] != split_by_id[tid]:
            raise SystemExit(f"window split mismatch: {row['window_id']}")
        h0 = int(row["history_start_index"])
        h1 = int(row["history_end_index_exclusive"])
        t0 = int(row["target_start_index"])
        t1 = int(row["target_end_index_exclusive"])
        if h1 - h0 != 40 or t1 - t0 != 20 or h1 != t0:
            raise SystemExit(f"window geometry error: {row['window_id']}")
        if t1 > length_by_id[tid]:
            raise SystemExit(f"window exceeds trajectory: {row['window_id']}")

    for row in processed:
        for key in ["npz_file", "cleaned_csv_file"]:
            value = row[key]
            if value and not (args.output_dir / value).exists():
                raise SystemExit(f"missing processed file: {value}")

    stats = json.loads(
        (args.output_dir / "training_standardization_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    if stats["statistics_source"] != "train_only":
        raise SystemExit("normalization source is not train_only")
    if stats["feature_order"] != ["lon", "lat", "sog", "cog"]:
        raise SystemExit("feature order mismatch")

    print(json.dumps({
        "trajectory_count": len(manifest),
        "window_count": len(windows),
        "processed_file_count": len(processed),
        "status": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
