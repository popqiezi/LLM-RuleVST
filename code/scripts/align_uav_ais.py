#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.geo_utils import (
    CoordinateTransformer,
    angular_difference_deg,
)
from scripts.common.io_utils import load_yaml
from scripts.common.kinematics import (
    sog_cog_to_velocity,
    velocity_to_sog_cog,
)


def interpolate_ais(
    ais: pd.DataFrame,
    timestamp: pd.Timestamp,
    transformer: CoordinateTransformer,
    maximum_gap_seconds: float,
):
    times = ais["postTime"].astype("int64").to_numpy()
    query = timestamp.value
    exact = np.searchsorted(times, query)
    if exact < len(times) and times[exact] == query:
        row = ais.iloc[exact]
        return "exact", row

    right = np.searchsorted(times, query, side="right")
    left = right - 1
    if left < 0 or right >= len(times):
        return None, None
    gap = (times[right] - times[left]) / 1.0e9
    if gap <= 0 or gap > maximum_gap_seconds:
        return None, None

    alpha = (query - times[left]) / (times[right] - times[left])
    left_row = ais.iloc[left]
    right_row = ais.iloc[right]

    x, y = transformer.lonlat_to_xy(
        np.asarray([left_row.lon, right_row.lon]),
        np.asarray([left_row.lat, right_row.lat]),
    )
    ve, vn = sog_cog_to_velocity(
        np.asarray([left_row.sog, right_row.sog]),
        np.asarray([left_row.cog, right_row.cog]),
    )
    xi = x[0] + alpha * (x[1] - x[0])
    yi = y[0] + alpha * (y[1] - y[0])
    vei = ve[0] + alpha * (ve[1] - ve[0])
    vni = vn[0] + alpha * (vn[1] - vn[0])
    lon, lat = transformer.xy_to_lonlat(
        np.asarray([xi]), np.asarray([yi])
    )
    sog, cog = velocity_to_sog_cog(
        np.asarray([vei]), np.asarray([vni])
    )
    row = pd.Series({
        "postTime": timestamp,
        "lon": lon[0],
        "lat": lat[0],
        "sog": sog[0],
        "cog": cog[0],
    })
    return "interpolated", row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav-file", type=Path, required=True)
    parser.add_argument("--ais-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/uav_ais_alignment.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    transformer = CoordinateTransformer(
        cfg["coordinate_system"]["geographic_crs"],
        cfg["coordinate_system"]["metric_crs"],
    )
    acfg = cfg["alignment"]
    trend_cfg = cfg["trend_filter"]

    uav = pd.read_csv(args.uav_file)
    uav["postTime"] = pd.to_datetime(
        uav["postTime"], errors="coerce"
    )
    uav = uav.dropna(
        subset=["postTime", "lon", "lat", "sog", "cog"]
    )

    ais_tracks = {}
    for path in sorted(args.ais_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["postTime"] = pd.to_datetime(
            frame["postTime"], errors="coerce"
        )
        for col in ["lon", "lat", "sog", "cog"]:
            frame[col] = pd.to_numeric(
                frame[col], errors="coerce"
            )
        frame = frame.dropna(
            subset=["postTime", "lon", "lat", "sog", "cog"]
        ).sort_values("postTime")
        if not frame.empty:
            ais_tracks[path.stem] = frame.reset_index(
                drop=True
            )

    rows = []
    for uav_row in uav.itertuples(index=False):
        best = None
        for mmsi, ais in ais_tracks.items():
            method, aligned = interpolate_ais(
                ais,
                uav_row.postTime,
                transformer,
                float(acfg["maximum_interpolation_gap_seconds"]),
            )

            if aligned is None:
                deltas = np.abs(
                    (
                        ais["postTime"] - uav_row.postTime
                    ).dt.total_seconds()
                )
                idx = int(deltas.idxmin())
                if (
                    float(deltas.loc[idx])
                    > float(acfg["maximum_time_difference_seconds"])
                ):
                    continue
                aligned = ais.loc[idx]
                method = "nearest"

            ux, uy = transformer.lonlat_to_xy(
                np.asarray([uav_row.lon]),
                np.asarray([uav_row.lat]),
            )
            ax, ay = transformer.lonlat_to_xy(
                np.asarray([aligned.lon]),
                np.asarray([aligned.lat]),
            )
            distance = float(
                np.hypot(ax[0] - ux[0], ay[0] - uy[0])
            )
            if distance > float(acfg["maximum_distance_m"]):
                continue

            course_difference = float(
                angular_difference_deg(
                    uav_row.cog, aligned.cog
                )
            )
            if (
                bool(trend_cfg["enabled"])
                and course_difference
                > float(
                    trend_cfg[
                        "maximum_course_difference_deg"
                    ]
                )
            ):
                continue

            time_difference = abs(
                (
                    pd.Timestamp(aligned.postTime)
                    - uav_row.postTime
                ).total_seconds()
            )
            score = (distance, time_difference)
            if best is None or score < best[0]:
                best = (
                    score,
                    mmsi,
                    method,
                    aligned,
                    distance,
                    course_difference,
                    time_difference,
                )

        if best is None:
            continue
        _, mmsi, method, aligned, distance, course_diff, time_diff = best
        rows.append({
            "uav_source_file": args.uav_file.name,
            "uav_frame": uav_row.frame,
            "uav_track_id": uav_row.id,
            "uav_timestamp": uav_row.postTime.isoformat(),
            "ais_mmsi": mmsi,
            "ais_timestamp": pd.Timestamp(
                aligned.postTime
            ).isoformat(),
            "alignment_method": method,
            "time_difference_s": time_diff,
            "position_error_m": distance,
            "course_difference_deg": course_diff,
            "uav_lon": uav_row.lon,
            "uav_lat": uav_row.lat,
            "ais_lon": aligned.lon,
            "ais_lat": aligned.lat,
            "uav_sog_kn": uav_row.sog,
            "ais_sog_kn": aligned.sog,
            "speed_error_kn": abs(
                float(uav_row.sog) - float(aligned.sog)
            ),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
