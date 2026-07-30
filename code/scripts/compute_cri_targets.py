#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.geo_utils import CoordinateTransformer
from scripts.common.io_utils import load_yaml
from scripts.common.kinematics import (
    angle_between_vectors,
    dcpa_tcpa,
    sog_cog_to_velocity,
)


def calculate_cri(
    distance_m: np.ndarray,
    dcpa_m: np.ndarray,
    tcpa_s: np.ndarray,
    relative_speed_mps: np.ndarray,
    theta_rad: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1 = float(cfg["d1_m"])
    d2 = float(cfg["d2_m"])
    w1 = float(cfg["w1_temporal"])
    w2 = float(cfg["w2_spatial"])
    tref = float(cfg["reference_time_s"])
    min_speed = float(cfg["minimum_relative_speed_mps"])
    min_tan = float(cfg["minimum_tangent_magnitude"])

    temporal = np.zeros_like(distance_m, dtype=np.float64)
    temporal[dcpa_m <= d1] = 1.0
    middle = (dcpa_m > d1) & (dcpa_m < d2)

    tangent = np.tan(theta_rad)
    denominator = relative_speed_mps * tangent
    safe_denominator = denominator.copy()
    small = np.abs(safe_denominator) < (
        min_speed * min_tan
    )
    safe_denominator[small] = np.where(
        safe_denominator[small] < 0,
        -(min_speed * min_tan),
        min_speed * min_tan,
    )
    correction_time = d1 / safe_denominator
    exponent = (
        -np.abs(tcpa_s) + correction_time
    ) / tref
    exponent = np.clip(
        exponent,
        float(cfg["exponent_minimum"]),
        float(cfg["exponent_maximum"]),
    )
    temporal[middle] = np.exp(exponent[middle])

    spatial = np.zeros_like(distance_m, dtype=np.float64)
    spatial[distance_m <= d1] = 1.0
    spatial_middle = (
        (distance_m > d1) & (distance_m < d2)
    )
    spatial[spatial_middle] = (
        d2 - distance_m[spatial_middle]
    ) / (d2 - d1)

    if bool(cfg["clip_risk_factors_to_unit_interval"]):
        temporal = np.clip(temporal, 0.0, 1.0)
        spatial = np.clip(spatial, 0.0, 1.0)

    cri = np.clip(w1 * temporal + w2 * spatial, 0.0, 1.0)
    return cri, temporal, spatial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encounter-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cri_parameters.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    ccfg = config["cri"]
    transformer = CoordinateTransformer(
        "EPSG:4326",
        config["coordinate_system"]["metric_crs"],
    )
    manifest = pd.read_csv(
        args.encounter_dir / "encounter_pair_manifest.csv"
    )

    output_files = args.output_dir / "cri_files"
    output_files.mkdir(parents=True, exist_ok=True)
    rows = []

    for row in manifest.itertuples(index=False):
        with np.load(
            args.encounter_dir / row.encounter_file,
            allow_pickle=False,
        ) as data:
            own = data["own_states_raw"].astype(np.float64)
            target = data["target_states_raw"].astype(
                np.float64
            )

        own_x, own_y = transformer.lonlat_to_xy(
            own[:, 0], own[:, 1]
        )
        target_x, target_y = transformer.lonlat_to_xy(
            target[:, 0], target[:, 1]
        )
        own_ve, own_vn = sog_cog_to_velocity(
            own[:, 2], own[:, 3]
        )
        target_ve, target_vn = sog_cog_to_velocity(
            target[:, 2], target[:, 3]
        )
        relative_position = np.column_stack([
            target_x - own_x,
            target_y - own_y,
        ])
        relative_velocity = np.column_stack([
            target_ve - own_ve,
            target_vn - own_vn,
        ])

        distance = np.linalg.norm(
            relative_position, axis=1
        )
        relative_speed = np.linalg.norm(
            relative_velocity, axis=1
        )
        dcpa, tcpa = dcpa_tcpa(
            relative_position, relative_velocity
        )
        theta = angle_between_vectors(
            relative_position, relative_velocity
        )

        cri, temporal, spatial = calculate_cri(
            distance,
            dcpa,
            tcpa,
            relative_speed,
            theta,
            ccfg,
        )
        threshold = float(ccfg["high_risk_threshold"])
        labels = (cri >= threshold).astype(np.int8)

        history_length = 40
        future_slice = slice(history_length, history_length + 20)
        path = (
            output_files / row.split / f"{row.encounter_id}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            encounter_id=np.asarray(row.encounter_id),
            cri_all=cri,
            temporal_risk_all=temporal,
            spatial_risk_all=spatial,
            dcpa_m_all=dcpa,
            tcpa_s_all=tcpa,
            distance_m_all=distance,
            future_cri=cri[future_slice],
            future_high_risk_labels=labels[future_slice],
            threshold=np.asarray(threshold),
        )
        rows.append({
            "dataset": row.dataset,
            "split": row.split,
            "encounter_id": row.encounter_id,
            "cri_file": str(path.relative_to(args.output_dir)),
            "future_length": 20,
            "threshold": threshold,
            "future_high_risk_count": int(
                np.sum(labels[future_slice])
            ),
        })

    pd.DataFrame(rows).to_csv(
        args.output_dir / "cri_target_index.csv",
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
