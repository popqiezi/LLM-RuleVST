#!/usr/bin/env python3
"""Encode validated rule fields as categorical IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FIELDS = [
    "encounter_type",
    "active_rule",
    "own_role",
    "compliance_state",
    "violation_action",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def encode(output: dict[str, str], vocabulary: dict[str, Any]) -> dict[str, Any]:
    unk = vocabulary["special_tokens"]["UNK"]
    ids = [vocabulary[field].get(output.get(field, ""), unk) for field in FIELDS]
    return {
        "categorical_rule_features": {
            f"{field}_id": value for field, value in zip(FIELDS, ids)
        },
        "feature_vector": ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    encoded = encode(load_json(args.output), load_json(args.vocab))
    text = json.dumps(encoded, indent=2)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
