#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.io_utils import load_json


FIELDS = [
    "encounter_type",
    "active_rule",
    "own_role",
    "compliance_state",
    "violation_action",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL line {line_number}: {exc}"
                ) from exc


def extract_output(record: dict) -> dict:
    for key in ["final_output", "parsed_output", "output"]:
        value = record.get(key)
        if isinstance(value, dict):
            return value
    if all(field in record for field in FIELDS):
        return {field: record[field] for field in FIELDS}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-output", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument(
        "--encounter-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=40)
    args = parser.parse_args()

    vocabulary = load_json(args.vocabulary)
    unk = int(vocabulary["special_tokens"]["UNK"])
    encounter_manifest = pd.read_csv(args.encounter_manifest)
    encounter_meta = encounter_manifest.set_index(
        "encounter_id"
    ).to_dict("index")

    grouped = {}
    for record in iter_jsonl(args.qwen_output):
        encounter_id = record["encounter_id"]
        time_index = int(record["time_index"])
        output = extract_output(record)
        encoded = [
            int(vocabulary[field].get(
                output.get(field, ""), unk
            ))
            for field in FIELDS
        ]
        grouped.setdefault(encounter_id, {})[time_index] = {
            "encoded": encoded,
            "valid": bool(record.get("valid", True)),
            "repair_attempted": bool(
                record.get("repair_attempted", False)
            ),
        }

    sequence_dir = args.output_dir / "sequence_files"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for encounter_id, meta in encounter_meta.items():
        time_map = grouped.get(encounter_id, {})
        matrix = np.full(
            (args.sequence_length, len(FIELDS)),
            unk,
            dtype=np.int64,
        )
        valid_flags = np.zeros(
            args.sequence_length, dtype=np.bool_
        )
        repair_flags = np.zeros(
            args.sequence_length, dtype=np.bool_
        )

        for time_index in range(args.sequence_length):
            item = time_map.get(time_index)
            if item is None:
                continue
            matrix[time_index] = item["encoded"]
            valid_flags[time_index] = item["valid"]
            repair_flags[time_index] = item[
                "repair_attempted"
            ]

        path = (
            sequence_dir
            / meta["split"]
            / f"{encounter_id}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            encounter_id=np.asarray(encounter_id),
            fields=np.asarray(FIELDS),
            rule_feature_ids=matrix,
            valid_flags=valid_flags,
            repair_flags=repair_flags,
        )
        index_rows.append({
            "dataset": meta["dataset"],
            "split": meta["split"],
            "encounter_id": encounter_id,
            "rule_sequence_file": str(
                path.relative_to(args.output_dir)
            ),
            "sequence_length": args.sequence_length,
            "number_of_fields": len(FIELDS),
            "missing_or_invalid_steps": int(
                np.sum(~valid_flags)
            ),
            "repair_count": int(np.sum(repair_flags)),
        })

    pd.DataFrame(index_rows).to_csv(
        args.output_dir / "rule_sequence_index.csv",
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
