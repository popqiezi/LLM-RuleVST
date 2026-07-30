#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


MODEL_FEATURES = ["lon", "lat", "sog", "cog"]
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def parse_mmsi(path: Path) -> str:
    return path.stem


def clean_frame(df: pd.DataFrame, timestamp_col: str, cfg: dict) -> pd.DataFrame:
    required = [timestamp_col, "lon", "lat", "sog", "cog"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df.attrs.get('source_file', '')}: missing columns {missing}")

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    for c in ["lon", "lat", "sog", "cog"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=required)
    out = out.drop_duplicates(subset=required, keep="first")

    c = cfg["cleaning"]
    mask = (
        out["lat"].between(c["latitude_min"], c["latitude_max"], inclusive="both")
        & out["lon"].between(c["longitude_min"], c["longitude_max"], inclusive="both")
        & out["sog"].between(c["sog_min_kn"], c["sog_max_kn"], inclusive="both")
        & out["cog"].between(c["cog_min_deg"], c["cog_max_deg"], inclusive="both")
    )
    out = out.loc[mask].copy()
    out["cog"] = np.mod(out["cog"].to_numpy(dtype=np.float64), 360.0)
    return out.sort_values(timestamp_col).reset_index(drop=True)


def segment_by_time_gap(
    df: pd.DataFrame,
    timestamp_col: str,
    gap_seconds: float,
) -> List[pd.DataFrame]:
    if df.empty:
        return []
    delta = df[timestamp_col].diff().dt.total_seconds()
    group_id = (delta.isna() | (delta > gap_seconds) | (delta <= 0)).cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(group_id, sort=True)]


def segment_uav_track(
    df: pd.DataFrame,
    timestamp_col: str,
    frame_col: str,
    minimum_gap_seconds: float,
    multiplier: float,
) -> List[pd.DataFrame]:
    if df.empty:
        return []

    df = df.sort_values([timestamp_col, frame_col]).reset_index(drop=True)
    positive_dt = df[timestamp_col].diff().dt.total_seconds()
    positive_dt = positive_dt[positive_dt > 0]
    median_dt = float(positive_dt.median()) if not positive_dt.empty else minimum_gap_seconds
    time_threshold = max(minimum_gap_seconds, multiplier * median_dt)

    time_gap = df[timestamp_col].diff().dt.total_seconds()
    frame_gap = df[frame_col].diff()
    split_flag = (
        time_gap.isna()
        | (time_gap > time_threshold)
        | (time_gap <= 0)
        | (frame_gap > 1)
        | (frame_gap <= 0)
    )
    group_id = split_flag.cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(group_id, sort=True)]


def assign_ais_split(
    start: pd.Timestamp,
    end: pd.Timestamp,
    split_cfg: dict,
) -> Optional[str]:
    for split in ["train", "validation", "test"]:
        lo = pd.Timestamp(split_cfg[split]["start"])
        hi = pd.Timestamp(split_cfg[split]["end"])
        if start >= lo and end <= hi:
            return split
    return None


def load_uav_assignments(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, dtype=str)
    if list(df.columns) != ["trajectory_id", "split"]:
        raise ValueError("uav_split_assignments.csv must contain trajectory_id,split")
    result = {}
    for row in df.itertuples(index=False):
        if row.split not in SPLIT_ORDER:
            raise ValueError(f"Invalid split {row.split}")
        result[row.trajectory_id] = row.split
    return result


def make_trajectory_record(
    dataset: str,
    split: str,
    trajectory_id: str,
    source_file: str,
    entity_id: str,
    segment_index: int,
    df: pd.DataFrame,
    timestamp_col: str,
    frame_col: Optional[str] = None,
) -> dict:
    timestamps = df[timestamp_col]
    dt = timestamps.diff().dt.total_seconds()
    positive_dt = dt[dt > 0]
    median_interval = float(positive_dt.median()) if not positive_dt.empty else None
    min_interval = float(positive_dt.min()) if not positive_dt.empty else None
    max_interval = float(positive_dt.max()) if not positive_dt.empty else None

    return {
        "dataset": dataset,
        "split": split,
        "trajectory_id": trajectory_id,
        "source_file": source_file,
        "entity_id": entity_id,
        "segment_index": segment_index,
        "start_time": timestamps.iloc[0].isoformat(),
        "end_time": timestamps.iloc[-1].isoformat(),
        "num_time_steps": int(len(df)),
        "median_sampling_interval_s": median_interval,
        "minimum_sampling_interval_s": min_interval,
        "maximum_sampling_interval_s": max_interval,
        "first_frame": int(df[frame_col].iloc[0]) if frame_col else "",
        "last_frame": int(df[frame_col].iloc[-1]) if frame_col else "",
    }


def generate_windows(
    trajectory_id: str,
    split: str,
    timestamps: np.ndarray,
    cfg: dict,
) -> List[dict]:
    tcfg = cfg["trajectory"]
    h = int(tcfg["historical_length"])
    f = int(tcfg["prediction_horizon"])
    stride = int(tcfg["stride"])
    total = h + f

    rows = []
    for number, start in enumerate(range(0, len(timestamps) - total + 1, stride)):
        hist_end = start + h
        target_end = start + total
        rows.append({
            "split": split,
            "trajectory_id": trajectory_id,
            "window_id": f"{trajectory_id}_W{number:06d}",
            "window_start_index": start,
            "history_start_index": start,
            "history_end_index_exclusive": hist_end,
            "target_start_index": hist_end,
            "target_end_index_exclusive": target_end,
            "history_start_time": pd.Timestamp(timestamps[start]).isoformat(),
            "history_end_time": pd.Timestamp(timestamps[hist_end - 1]).isoformat(),
            "target_start_time": pd.Timestamp(timestamps[hist_end]).isoformat(),
            "target_end_time": pd.Timestamp(timestamps[target_end - 1]).isoformat(),
            "history_length": h,
            "prediction_horizon": f,
            "stride": stride,
        })
    return rows


def stable_file_name(trajectory_id: str) -> str:
    digest = hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()[:12]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in trajectory_id)
    return f"{safe}_{digest}"


def write_trajectory_csv(path: Path, df: pd.DataFrame, timestamp_col: str) -> None:
    out = df.copy()
    out[timestamp_col] = out[timestamp_col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out.to_csv(path, index=False)


def save_processed(
    output_root: Path,
    record: dict,
    df: pd.DataFrame,
    timestamp_col: str,
    mean: np.ndarray,
    std: np.ndarray,
    save_csv: bool,
) -> dict:
    dataset = record["dataset"].lower().replace("-", "_")
    split = record["split"]
    stem = stable_file_name(record["trajectory_id"])
    folder = output_root / "processed" / dataset / split
    folder.mkdir(parents=True, exist_ok=True)

    raw = df[MODEL_FEATURES].to_numpy(dtype=np.float64)
    standardized = (raw - mean) / std
    timestamp_ns = df[timestamp_col].astype("int64").to_numpy(dtype=np.int64)

    npz_path = folder / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        feature_order=np.asarray(MODEL_FEATURES),
        raw_features=raw,
        standardized_features=standardized,
        timestamp_ns=timestamp_ns,
        trajectory_id=np.asarray(record["trajectory_id"]),
        dataset=np.asarray(record["dataset"]),
        split=np.asarray(record["split"]),
    )

    cleaned_csv = ""
    if save_csv:
        csv_path = folder / f"{stem}.csv"
        write_trajectory_csv(csv_path, df, timestamp_col)
        cleaned_csv = str(csv_path.relative_to(output_root))

    return {
        "dataset": record["dataset"],
        "split": record["split"],
        "trajectory_id": record["trajectory_id"],
        "num_time_steps": record["num_time_steps"],
        "npz_file": str(npz_path.relative_to(output_root)),
        "cleaned_csv_file": cleaned_csv,
        "feature_order": "|".join(MODEL_FEATURES),
        "coordinate_system": "WGS84",
        "speed_unit": "kn",
        "course_unit": "degree",
    }


def write_csv(path: Path, rows: List[dict], fieldnames: List[str], compress: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open
    kwargs = {"mode": "wt", "encoding": "utf-8", "newline": ""} if compress else {"mode": "w", "encoding": "utf-8", "newline": ""}
    with opener(path, **kwargs) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/dataset_processing.yaml"))
    args = parser.parse_args()

    project_root = args.config.resolve().parents[1]
    cfg = load_config(args.config)
    ais_dir = resolve(project_root, cfg["paths"]["ais_dir"])
    uav_dir = resolve(project_root, cfg["paths"]["uav_tsd_dir"])
    output_root = resolve(project_root, cfg["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    min_length = int(cfg["trajectory"]["minimum_length"])
    trajectories: List[Tuple[dict, pd.DataFrame, str]] = []
    excluded_rows: List[dict] = []

    # AIS
    ais_gap = float(cfg["trajectory"]["ais_gap_seconds"])
    ais_split_cfg = cfg["splits"]["ais"]

    for path in sorted(ais_dir.glob("*.csv")):
        df = pd.read_csv(path)
        df.attrs["source_file"] = path.name
        try:
            df = clean_frame(df, "postTime", cfg)
        except ValueError as exc:
            excluded_rows.append({
                "dataset": "AIS",
                "source_file": path.name,
                "entity_id": parse_mmsi(path),
                "segment_index": "",
                "num_time_steps": 0,
                "reason": str(exc),
            })
            continue

        for seg_idx, seg in enumerate(segment_by_time_gap(df, "postTime", ais_gap)):
            trajectory_id = f"AIS_MMSI_{parse_mmsi(path)}_SEG_{seg_idx:03d}"
            if len(seg) < min_length:
                excluded_rows.append({
                    "dataset": "AIS",
                    "source_file": path.name,
                    "entity_id": parse_mmsi(path),
                    "segment_index": seg_idx,
                    "num_time_steps": len(seg),
                    "reason": "length_below_60",
                })
                continue

            split = assign_ais_split(seg["postTime"].iloc[0], seg["postTime"].iloc[-1], ais_split_cfg)
            if split is None:
                excluded_rows.append({
                    "dataset": "AIS",
                    "source_file": path.name,
                    "entity_id": parse_mmsi(path),
                    "segment_index": seg_idx,
                    "num_time_steps": len(seg),
                    "reason": "outside_or_crosses_timestamp_split",
                })
                continue

            record = make_trajectory_record(
                "AIS", split, trajectory_id, path.name, parse_mmsi(path),
                seg_idx, seg, "postTime"
            )
            trajectories.append((record, seg, "postTime"))

    # UAV-TSD
    uav_cfg = cfg["columns"]["uav_tsd"]
    assignments = load_uav_assignments(
        resolve(project_root, cfg["splits"]["uav_tsd"]["assignment_file"])
    )
    uav_min_gap = float(cfg["trajectory"]["uav_gap_seconds_minimum"])
    uav_multiplier = float(cfg["trajectory"]["uav_gap_multiplier"])

    for path in sorted(uav_dir.glob("*.csv")):
        df = pd.read_csv(path)
        df.attrs["source_file"] = path.name
        try:
            df = clean_frame(df, uav_cfg["timestamp"], cfg)
        except ValueError as exc:
            excluded_rows.append({
                "dataset": "UAV-TSD",
                "source_file": path.name,
                "entity_id": "",
                "segment_index": "",
                "num_time_steps": 0,
                "reason": str(exc),
            })
            continue

        if uav_cfg["track_id"] not in df.columns or uav_cfg["frame"] not in df.columns:
            excluded_rows.append({
                "dataset": "UAV-TSD",
                "source_file": path.name,
                "entity_id": "",
                "segment_index": "",
                "num_time_steps": 0,
                "reason": "missing_track_or_frame_column",
            })
            continue

        for track_id, track_df in df.groupby(uav_cfg["track_id"], sort=True):
            segments = segment_uav_track(
                track_df,
                uav_cfg["timestamp"],
                uav_cfg["frame"],
                uav_min_gap,
                uav_multiplier,
            )
            for seg_idx, seg in enumerate(segments):
                trajectory_id = f"UAV_{path.stem}_ID_{track_id}_SEG_{seg_idx:03d}"
                if len(seg) < min_length:
                    excluded_rows.append({
                        "dataset": "UAV-TSD",
                        "source_file": path.name,
                        "entity_id": str(track_id),
                        "segment_index": seg_idx,
                        "num_time_steps": len(seg),
                        "reason": "length_below_60",
                    })
                    continue
                split = assignments.get(trajectory_id)
                if split is None:
                    excluded_rows.append({
                        "dataset": "UAV-TSD",
                        "source_file": path.name,
                        "entity_id": str(track_id),
                        "segment_index": seg_idx,
                        "num_time_steps": len(seg),
                        "reason": "missing_split_assignment",
                    })
                    continue

                record = make_trajectory_record(
                    "UAV-TSD", split, trajectory_id, path.name, str(track_id),
                    seg_idx, seg, uav_cfg["timestamp"], uav_cfg["frame"]
                )
                trajectories.append((record, seg, uav_cfg["timestamp"]))

    if not trajectories:
        raise RuntimeError("No eligible trajectories were produced.")

    trajectories.sort(
        key=lambda item: (
            SPLIT_ORDER[item[0]["split"]],
            item[0]["dataset"],
            item[0]["trajectory_id"],
        )
    )

    # Training-only feature statistics across both independently evaluated sources.
    train_arrays = [
        df[MODEL_FEATURES].to_numpy(dtype=np.float64)
        for record, df, _ in trajectories
        if record["split"] == "train"
    ]
    if not train_arrays:
        raise RuntimeError("No training trajectories are available.")
    train_matrix = np.concatenate(train_arrays, axis=0)
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0, ddof=int(cfg["normalization"]["ddof"]))
    if np.any(std <= 0):
        raise RuntimeError("At least one training feature has zero standard deviation.")

    stats = {
        "method": "z_score",
        "statistics_source": "train_only",
        "ddof": int(cfg["normalization"]["ddof"]),
        "feature_order": MODEL_FEATURES,
        "number_of_training_time_steps": int(train_matrix.shape[0]),
        "mean": {name: float(value) for name, value in zip(MODEL_FEATURES, mean)},
        "std": {name: float(value) for name, value in zip(MODEL_FEATURES, std)},
    }
    (output_root / "training_standardization_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    trajectory_rows = []
    window_rows = []
    processed_rows = []

    for record, df, timestamp_col in trajectories:
        timestamps = df[timestamp_col].to_numpy(dtype="datetime64[ns]")
        windows = generate_windows(
            record["trajectory_id"], record["split"], timestamps, cfg
        )
        record = dict(record)
        record["num_windows"] = len(windows)
        record["minimum_required_length"] = min_length
        record["historical_length"] = int(cfg["trajectory"]["historical_length"])
        record["prediction_horizon"] = int(cfg["trajectory"]["prediction_horizon"])
        record["stride"] = int(cfg["trajectory"]["stride"])
        record["split_before_window_generation"] = True
        record["native_sampling_retained"] = True
        trajectory_rows.append(record)
        window_rows.extend(windows)
        processed_rows.append(
            save_processed(
                output_root,
                record,
                df,
                timestamp_col,
                mean,
                std,
                bool(cfg["output"]["save_cleaned_trajectory_csv"]),
            )
        )

    trajectory_fields = [
        "dataset", "split", "trajectory_id", "source_file", "entity_id",
        "segment_index", "start_time", "end_time", "num_time_steps",
        "median_sampling_interval_s", "minimum_sampling_interval_s",
        "maximum_sampling_interval_s", "first_frame", "last_frame",
        "num_windows", "minimum_required_length", "historical_length",
        "prediction_horizon", "stride", "split_before_window_generation",
        "native_sampling_retained",
    ]
    write_csv(
        output_root / "trajectory_split_manifest.csv",
        trajectory_rows,
        trajectory_fields,
    )

    window_fields = [
        "split", "trajectory_id", "window_id", "window_start_index",
        "history_start_index", "history_end_index_exclusive",
        "target_start_index", "target_end_index_exclusive",
        "history_start_time", "history_end_time", "target_start_time",
        "target_end_time", "history_length", "prediction_horizon", "stride",
    ]
    compress = bool(cfg["output"]["compress_window_index"])
    window_path = output_root / (
        "sliding_window_index.csv.gz" if compress else "sliding_window_index.csv"
    )
    write_csv(window_path, window_rows, window_fields, compress=compress)

    processed_fields = [
        "dataset", "split", "trajectory_id", "num_time_steps", "npz_file",
        "cleaned_csv_file", "feature_order", "coordinate_system",
        "speed_unit", "course_unit",
    ]
    write_csv(
        output_root / "processed_feature_index.csv",
        processed_rows,
        processed_fields,
    )

    excluded_fields = [
        "dataset", "source_file", "entity_id", "segment_index",
        "num_time_steps", "reason",
    ]
    write_csv(
        output_root / "excluded_trajectories.csv",
        excluded_rows,
        excluded_fields,
    )

    summary = {
        "eligible_trajectories": {},
        "model_windows": {},
        "time_steps": {},
        "minimum_trajectory_length": {},
        "mean_trajectory_length": {},
        "maximum_trajectory_length": {},
        "configuration": {
            "minimum_length": min_length,
            "historical_length": int(cfg["trajectory"]["historical_length"]),
            "prediction_horizon": int(cfg["trajectory"]["prediction_horizon"]),
            "stride": int(cfg["trajectory"]["stride"]),
            "native_sampling_retained": True,
            "partition_before_window_generation": True,
        },
    }

    for dataset in ["AIS", "UAV-TSD"]:
        for split in ["train", "validation", "test"]:
            key = f"{dataset}:{split}"
            lengths = [
                int(r["num_time_steps"])
                for r in trajectory_rows
                if r["dataset"] == dataset and r["split"] == split
            ]
            windows = [
                r for r in window_rows
                if any(
                    t["trajectory_id"] == r["trajectory_id"]
                    and t["dataset"] == dataset
                    for t in trajectory_rows
                )
                and r["split"] == split
            ]
            summary["eligible_trajectories"][key] = len(lengths)
            summary["model_windows"][key] = len(windows)
            summary["time_steps"][key] = int(sum(lengths))
            summary["minimum_trajectory_length"][key] = min(lengths) if lengths else None
            summary["mean_trajectory_length"][key] = (
                float(np.mean(lengths)) if lengths else None
            )
            summary["maximum_trajectory_length"][key] = max(lengths) if lengths else None

    (output_root / "preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    schema = {
        "raw_formats": {
            "UAV-TSD": [
                "frame", "id", "class_id", "class_name",
                "postTime", "lon", "lat", "sog", "cog",
            ],
            "AIS": [
                "postTime", "lon", "lat", "sog", "cog", "heading", "rot",
            ],
        },
        "model_features": MODEL_FEATURES,
        "model_feature_units": {
            "lon": "degree_WGS84",
            "lat": "degree_WGS84",
            "sog": "kn",
            "cog": "degree",
        },
        "standardization": "training-set feature-wise z-score",
        "window_indexing": "zero_based_end_exclusive",
    }
    (output_root / "column_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
