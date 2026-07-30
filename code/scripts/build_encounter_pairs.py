#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.common.geo_utils import CoordinateTransformer
from scripts.common.io_utils import load_npz_dict, load_yaml


def interpolate_target_ais(
    target_states: np.ndarray,
    target_time_ns: np.ndarray,
    query_time_ns: np.ndarray,
    transformer: CoordinateTransformer,
    maximum_gap_seconds: float,
) -> Optional[np.ndarray]:
    order = np.argsort(target_time_ns)
    target_time_ns = target_time_ns[order]
    target_states = target_states[order]

    lon = target_states[:, 0]
    lat = target_states[:, 1]
    x, y = transformer.lonlat_to_xy(lon, lat)

    speed = target_states[:, 2] * 0.514444
    angle = np.radians(target_states[:, 3])
    ve = speed * np.sin(angle)
    vn = speed * np.cos(angle)

    output = np.empty((len(query_time_ns), 4), dtype=np.float64)
    for i, timestamp in enumerate(query_time_ns):
        exact = np.searchsorted(target_time_ns, timestamp)
        if exact < len(target_time_ns) and target_time_ns[exact] == timestamp:
            output[i] = target_states[exact]
            continue

        right = np.searchsorted(target_time_ns, timestamp, side="right")
        left = right - 1
        if left < 0 or right >= len(target_time_ns):
            return None

        gap_seconds = (target_time_ns[right] - target_time_ns[left]) / 1.0e9
        if gap_seconds <= 0 or gap_seconds > maximum_gap_seconds:
            return None

        alpha = (timestamp - target_time_ns[left]) / (
            target_time_ns[right] - target_time_ns[left]
        )
        xi = x[left] + alpha * (x[right] - x[left])
        yi = y[left] + alpha * (y[right] - y[left])
        vei = ve[left] + alpha * (ve[right] - ve[left])
        vni = vn[left] + alpha * (vn[right] - vn[left])
        loni, lati = transformer.xy_to_lonlat(
            np.asarray([xi]), np.asarray([yi])
        )
        sog = np.hypot(vei, vni) / 0.514444
        cog = np.mod(np.degrees(np.arctan2(vei, vni)), 360.0)
        output[i] = [loni[0], lati[0], sog, cog]
    return output


def align_target_uav(
    target_states: np.ndarray,
    target_time_ns: np.ndarray,
    query_time_ns: np.ndarray,
) -> Optional[np.ndarray]:
    mapping = {int(t): i for i, t in enumerate(target_time_ns)}
    indices = []
    for timestamp in query_time_ns:
        idx = mapping.get(int(timestamp))
        if idx is None:
            return None
        indices.append(idx)
    return target_states[np.asarray(indices, dtype=np.int64)]


def encounter_identifier(
    dataset: str,
    split: str,
    own_trajectory_id: str,
    target_trajectory_id: str,
    own_window_id: str,
) -> str:
    raw = "|".join([
        dataset, split, own_trajectory_id,
        target_trajectory_id, own_window_id,
    ])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"ENC_{digest}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/encounter_construction.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    transformer = CoordinateTransformer(
        cfg["coordinate_system"]["geographic_crs"],
        cfg["coordinate_system"]["metric_crs"],
    )

    manifest = pd.read_csv(args.dataset_output / "trajectory_split_manifest.csv")
    windows = pd.read_csv(
        args.dataset_output / "sliding_window_index.csv.gz",
        compression="gzip",
    )
    index = pd.read_csv(args.dataset_output / "processed_feature_index.csv")

    trajectory_meta = manifest.set_index("trajectory_id").to_dict("index")
    file_by_trajectory = {
        row.trajectory_id: args.dataset_output / row.npz_file
        for row in index.itertuples(index=False)
    }

    cache = {}
    for trajectory_id, path in file_by_trajectory.items():
        data = load_npz_dict(path)
        cache[trajectory_id] = {
            "states": data["raw_features"].astype(np.float64),
            "time_ns": data["timestamp_ns"].astype(np.int64),
        }

    total_length = int(cfg["sequence"]["total_length"])
    history_length = int(cfg["sequence"]["historical_length"])
    maximum_distance = float(cfg["pairing"]["maximum_distance_m"])
    distance_index = int(cfg["pairing"]["distance_evaluation_index"])
    maximum_gap = float(
        cfg["ais_alignment"]["maximum_bracketing_gap_seconds"]
    )
    require_same_source = bool(
        cfg["uav_tsd_alignment"]["require_same_source_file"]
    )

    encounter_directory = args.output_dir / cfg["storage"]["encounter_subdirectory"]
    encounter_directory.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()

    for own_window in windows.itertuples(index=False):
        own_id = own_window.trajectory_id
        own_meta = trajectory_meta[own_id]
        own_dataset = own_meta["dataset"]
        own_split = own_meta["split"]
        own_data = cache[own_id]

        start = int(own_window.window_start_index)
        end = start + total_length
        if end > len(own_data["time_ns"]):
            continue

        own_states = own_data["states"][start:end]
        query_time_ns = own_data["time_ns"][start:end]

        candidates = manifest[
            (manifest["dataset"] == own_dataset)
            & (manifest["split"] == own_split)
            & (manifest["trajectory_id"] != own_id)
        ]

        if own_dataset == "UAV-TSD" and require_same_source:
            candidates = candidates[
                candidates["source_file"] == own_meta["source_file"]
            ]

        for target_row in candidates.itertuples(index=False):
            target_id = target_row.trajectory_id
            target_data = cache[target_id]

            if own_dataset == "AIS":
                target_states = interpolate_target_ais(
                    target_data["states"],
                    target_data["time_ns"],
                    query_time_ns,
                    transformer,
                    maximum_gap,
                )
                alignment_method = "linear_interpolation_utm"
            else:
                target_states = align_target_uav(
                    target_data["states"],
                    target_data["time_ns"],
                    query_time_ns,
                )
                alignment_method = "exact_common_timestamps"

            if target_states is None or len(target_states) != total_length:
                continue

            own_x, own_y = transformer.lonlat_to_xy(
                own_states[:, 0], own_states[:, 1]
            )
            target_x, target_y = transformer.lonlat_to_xy(
                target_states[:, 0], target_states[:, 1]
            )
            distance = np.hypot(target_x - own_x, target_y - own_y)
            final_history_distance = float(distance[distance_index])

            if final_history_distance > maximum_distance:
                continue

            duplicate_key = "|".join([
                own_id, target_id, own_window.window_id
            ])
            if duplicate_key in seen:
                continue
            seen.add(duplicate_key)

            encounter_id = encounter_identifier(
                own_dataset,
                own_split,
                own_id,
                target_id,
                own_window.window_id,
            )
            encounter_file = (
                encounter_directory
                / own_split
                / f"{encounter_id}.npz"
            )
            encounter_file.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                encounter_file,
                encounter_id=np.asarray(encounter_id),
                dataset=np.asarray(own_dataset),
                split=np.asarray(own_split),
                own_trajectory_id=np.asarray(own_id),
                target_trajectory_id=np.asarray(target_id),
                own_window_id=np.asarray(own_window.window_id),
                timestamps_ns=query_time_ns,
                own_states_raw=own_states,
                target_states_raw=target_states,
                relative_distance_m=distance,
                history_length=np.asarray(history_length),
                prediction_horizon=np.asarray(
                    int(cfg["sequence"]["prediction_horizon"])
                ),
            )

            rows.append({
                "dataset": own_dataset,
                "split": own_split,
                "encounter_id": encounter_id,
                "own_trajectory_id": own_id,
                "target_trajectory_id": target_id,
                "own_window_id": own_window.window_id,
                "target_window_id": "",
                "alignment_method": alignment_method,
                "final_history_distance_m": final_history_distance,
                "spatial_threshold_m": maximum_distance,
                "ordered_pair": True,
                "duplicate_key": duplicate_key,
                "encounter_file": str(
                    encounter_file.relative_to(args.output_dir)
                ),
            })

    columns = [
        "dataset", "split", "encounter_id",
        "own_trajectory_id", "target_trajectory_id",
        "own_window_id", "target_window_id",
        "alignment_method", "final_history_distance_m",
        "spatial_threshold_m", "ordered_pair",
        "duplicate_key", "encounter_file",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        args.output_dir / cfg["storage"]["manifest_name"],
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
