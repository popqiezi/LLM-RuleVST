#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.geo_utils import CoordinateTransformer
from scripts.common.io_utils import load_yaml
from scripts.common.kinematics import KNOT_TO_MPS


REQUIRED = [
    "frame", "id", "class_id", "class_name",
    "postTime", "lon", "lat", "sog", "cog",
]


def time_weighted_reference(
    previous_xy: np.ndarray,
    current_time_ns: int,
    previous_time_ns: int,
    next_xy: np.ndarray,
    next_time_ns: int,
) -> np.ndarray:
    denominator = next_time_ns - previous_time_ns
    if denominator <= 0:
        return previous_xy.copy()
    weight = (current_time_ns - previous_time_ns) / denominator
    return previous_xy + weight * (next_xy - previous_xy)


def correct_track(
    track: pd.DataFrame,
    transformer: CoordinateTransformer,
    window_size: int,
    multiplier: float,
    isolated_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = track.sort_values(["postTime", "frame"]).reset_index(drop=True).copy()
    lon = result["lon"].to_numpy(dtype=np.float64)
    lat = result["lat"].to_numpy(dtype=np.float64)
    x, y = transformer.lonlat_to_xy(lon, lat)
    xy = np.column_stack([x, y])
    time_ns = result["postTime"].astype("int64").to_numpy(dtype=np.int64)

    residual = np.full(len(result), np.nan, dtype=np.float64)
    reference = np.full_like(xy, np.nan)

    for i in range(1, len(result) - 1):
        reference[i] = time_weighted_reference(
            xy[i - 1], int(time_ns[i]), int(time_ns[i - 1]),
            xy[i + 1], int(time_ns[i + 1]),
        )
        residual[i] = np.linalg.norm(xy[i] - reference[i])

    half = window_size // 2
    flagged = np.zeros(len(result), dtype=bool)
    local_median = np.full(len(result), np.nan)
    local_mad = np.full(len(result), np.nan)

    for i in range(1, len(result) - 1):
        lo = max(1, i - half)
        hi = min(len(result) - 1, i + half + 1)
        values = residual[lo:hi]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        local_median[i] = median
        local_mad[i] = mad
        flagged[i] = residual[i] > median + multiplier * mad

    if isolated_only:
        replace = flagged.copy()
        for i in range(1, len(result) - 1):
            if flagged[i] and (flagged[i - 1] or flagged[i + 1]):
                replace[i] = False
    else:
        replace = flagged

    xy_corrected = xy.copy()
    valid_reference = np.isfinite(reference).all(axis=1)
    replace &= valid_reference
    xy_corrected[replace] = reference[replace]

    corrected_lon, corrected_lat = transformer.xy_to_lonlat(
        xy_corrected[:, 0], xy_corrected[:, 1]
    )
    result["lon"] = corrected_lon
    result["lat"] = corrected_lat

    # Recompute SOG and COG from corrected positions.
    dt = np.diff(time_ns) / 1.0e9
    dx = np.diff(xy_corrected[:, 0])
    dy = np.diff(xy_corrected[:, 1])
    valid = dt > 0
    speed_mps = np.full(len(result) - 1, np.nan)
    speed_mps[valid] = np.hypot(dx[valid], dy[valid]) / dt[valid]
    cog = np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)

    if len(result) > 1:
        result.loc[1:, "sog"] = speed_mps / KNOT_TO_MPS
        result.loc[1:, "cog"] = cog
        result.loc[0, "sog"] = result.loc[1, "sog"]
        result.loc[0, "cog"] = result.loc[1, "cog"]

    audit = pd.DataFrame({
        "frame": result["frame"],
        "id": result["id"],
        "postTime": result["postTime"],
        "residual_m": residual,
        "local_median_m": local_median,
        "local_mad_m": local_mad,
        "flagged": flagged,
        "replaced": replace,
    })
    return result, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trajectory_correction.yaml"),
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    transformer = CoordinateTransformer(
        cfg["coordinate_system"]["geographic_crs"],
        cfg["coordinate_system"]["metric_crs"],
    )
    ccfg = cfg["correction"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audits = []
    for path in sorted(args.input_dir.glob("*.csv")):
        df = pd.read_csv(path)
        missing = sorted(set(REQUIRED) - set(df.columns))
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        df["postTime"] = pd.to_datetime(df["postTime"], errors="coerce")
        for col in ["lon", "lat", "sog", "cog", "frame", "id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["postTime", "lon", "lat", "frame", "id"])

        corrected_parts = []
        for _, track in df.groupby("id", sort=True):
            if len(track) < int(ccfg["minimum_points"]):
                corrected_parts.append(track)
                continue
            corrected, audit = correct_track(
                track,
                transformer,
                int(ccfg["window_size"]),
                float(ccfg["mad_multiplier"]),
                bool(ccfg["replace_only_isolated_points"]),
            )
            audit.insert(0, "source_file", path.name)
            audits.append(audit)
            corrected_parts.append(corrected)

        output = pd.concat(corrected_parts, ignore_index=True)
        output = output.sort_values(["frame", "id"])
        output["postTime"] = output["postTime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        output.to_csv(args.output_dir / path.name, index=False)

    if audits:
        pd.concat(audits, ignore_index=True).to_csv(
            args.output_dir / "trajectory_correction_audit.csv",
            index=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
