#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.geo_utils import (
    CoordinateTransformer,
    angular_difference_deg,
    relative_bearing_deg,
)
from scripts.common.io_utils import load_yaml
from scripts.common.kinematics import (
    dcpa_tcpa,
    sog_cog_to_velocity,
)


def coarse_encounter_type(
    relative_bearing: float,
    course_difference: float,
    own_sog: float,
    target_sog: float,
    cfg: dict,
) -> str:
    geometry = cfg["coarse_geometry"]

    if (
        course_difference
        >= float(geometry["head_on_min_course_difference_deg"])
        and abs(relative_bearing)
        <= float(geometry["head_on_max_relative_bearing_deg"])
    ):
        return "head_on"

    if (
        course_difference
        <= float(geometry["overtaking_max_course_difference_deg"])
        and abs(relative_bearing)
        <= float(geometry["overtaking_max_forward_bearing_deg"])
        and own_sog > target_sog
    ):
        return "overtaking"

    if (
        float(geometry["crossing_min_course_difference_deg"])
        <= course_difference
        < float(geometry["crossing_max_course_difference_deg"])
    ):
        return "crossing"

    return "none"


def load_optional_metadata(path_value) -> dict:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str)
    key = "entity_id" if "entity_id" in frame.columns else frame.columns[0]
    return frame.set_index(key).to_dict("index")


def load_local_context(path_value) -> dict[tuple[str, int], str]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    required = {"encounter_id", "time_index", "context_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"local_context_csv is missing columns: {sorted(missing)}"
        )
    allowed = {"boundary_constrained", "bridge_constrained"}
    result = {}
    for row in frame.itertuples(index=False):
        label = str(row.context_label)
        if label not in allowed:
            raise ValueError(f"Unsupported local context label: {label}")
        result[(str(row.encounter_id), int(row.time_index))] = label
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encounter-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rule_reasoning_inputs.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest = pd.read_csv(
        args.encounter_dir / "encounter_pair_manifest.csv"
    )
    transformer = CoordinateTransformer(
        "EPSG:4326",
        cfg["coordinate_system"]["metric_crs"],
    )

    metadata = load_optional_metadata(
        cfg["metadata"].get("vessel_metadata_csv")
    )
    local_context = load_local_context(
        cfg["metadata"].get("local_context_csv")
    )
    history_length = int(cfg["sequence"]["historical_length"])
    short_length = int(cfg["sequence"]["short_history_length"])
    general_rules = list(
        cfg["candidate_rule_retrieval"]["general_rules"]
    )
    encounter_rules = cfg["candidate_rule_retrieval"][
        "encounter_rules"
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in manifest.itertuples(index=False):
            with np.load(
                args.encounter_dir / row.encounter_file,
                allow_pickle=False,
            ) as data:
                own = data["own_states_raw"].astype(np.float64)
                target = data["target_states_raw"].astype(np.float64)
                timestamps = data["timestamps_ns"].astype(np.int64)

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
            dcpa, tcpa = dcpa_tcpa(
                relative_position, relative_velocity
            )
            distance = np.linalg.norm(
                relative_position, axis=1
            )
            rel_bearing = relative_bearing_deg(
                own_x, own_y, target_x, target_y, own[:, 3]
            )
            course_difference = angular_difference_deg(
                own[:, 3], target[:, 3]
            )
            relative_speed = np.linalg.norm(
                relative_velocity, axis=1
            )

            for time_index in range(history_length):
                coarse_type = coarse_encounter_type(
                    float(rel_bearing[time_index]),
                    float(course_difference[time_index]),
                    float(own[time_index, 2]),
                    float(target[time_index, 2]),
                    cfg,
                )
                coarse_type = local_context.get(
                    (row.encounter_id, time_index),
                    coarse_type,
                )
                candidate_rules = list(dict.fromkeys(
                    general_rules
                    + list(encounter_rules.get(coarse_type, []))
                ))

                start = max(0, time_index - short_length + 1)
                short_history = []
                for j in range(start, time_index + 1):
                    short_history.append({
                        "time_index": j,
                        "timestamp_ns": int(timestamps[j]),
                        "own": {
                            "lon": float(own[j, 0]),
                            "lat": float(own[j, 1]),
                            "sog_kn": float(own[j, 2]),
                            "cog_deg": float(own[j, 3]),
                        },
                        "target": {
                            "lon": float(target[j, 0]),
                            "lat": float(target[j, 1]),
                            "sog_kn": float(target[j, 2]),
                            "cog_deg": float(target[j, 3]),
                        },
                    })

                record = {
                    "case_id": (
                        f"{row.encounter_id}_T{time_index:03d}"
                    ),
                    "encounter_id": row.encounter_id,
                    "dataset": row.dataset,
                    "split": row.split,
                    "time_index": time_index,
                    "timestamp_ns": int(timestamps[time_index]),
                    "causal_history_start_index": start,
                    "causal_history_end_index": time_index,
                    "waterway_type": cfg["metadata"][
                        "default_waterway_type"
                    ],
                    "coarse_encounter_type": coarse_type,
                    "retrieved_candidate_rules": candidate_rules,
                    "own_ship": metadata.get(
                        row.own_trajectory_id,
                        {"identifier": row.own_trajectory_id},
                    ),
                    "target_ship": metadata.get(
                        row.target_trajectory_id,
                        {"identifier": row.target_trajectory_id},
                    ),
                    "historical_state_window": short_history,
                    "relative_navigation_state": {
                        "range_m": float(distance[time_index]),
                        "relative_bearing_deg": float(
                            rel_bearing[time_index]
                        ),
                        "relative_course_difference_deg": float(
                            course_difference[time_index]
                        ),
                        "relative_speed_mps": float(
                            relative_speed[time_index]
                        ),
                        "DCPA_m": float(dcpa[time_index]),
                        "TCPA_s": float(tcpa[time_index]),
                    },
                    "future_horizon_used": False,
                }
                out.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
