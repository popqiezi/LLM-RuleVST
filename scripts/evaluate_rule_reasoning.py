#!/usr/bin/env python3
"""Evaluate five-field rule reasoning on a held-out JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from sklearn.metrics import accuracy_score, classification_report

FIELDS = [
    "encounter_type",
    "active_rule",
    "own_role",
    "compliance_state",
    "violation_action",
]


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    args = parser.parse_args()

    gold = {row["case_id"]: row for row in read_jsonl(args.gold)}
    pred = {row["case_id"]: row for row in read_jsonl(args.pred)}
    common_ids = sorted(set(gold) & set(pred))
    if not common_ids:
        raise SystemExit("No matching case_id values were found.")

    results = {}
    for field in FIELDS:
        y_true = [gold[i][field] for i in common_ids]
        y_pred = [pred[i][field] for i in common_ids]
        results[field] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }

    malformed = sum(bool(pred[i].get("malformed", False)) for i in common_ids)
    unsupported = sum(bool(pred[i].get("unsupported_rule", False)) for i in common_ids)
    results["summary"] = {
        "n_cases": len(common_ids),
        "malformed_output_rate": malformed / len(common_ids),
        "unsupported_rule_rate": unsupported / len(common_ids),
    }

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
