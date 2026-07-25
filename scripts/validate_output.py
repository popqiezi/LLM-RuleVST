#!/usr/bin/env python3
"""Validate a structured rule-reasoning output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = [
    "encounter_type",
    "active_rule",
    "own_role",
    "compliance_state",
    "violation_action",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_output(
    output: dict[str, Any],
    vocabulary: dict[str, Any],
    candidate_rules: list[str] | None = None,
) -> list[str]:
    errors: list[str] = []

    if set(output.keys()) != set(REQUIRED_FIELDS):
        missing = sorted(set(REQUIRED_FIELDS) - set(output.keys()))
        extra = sorted(set(output.keys()) - set(REQUIRED_FIELDS))
        if missing:
            errors.append(f"E02_MISSING_FIELD:{','.join(missing)}")
        if extra:
            errors.append(f"E03_EXTRA_FIELD:{','.join(extra)}")

    for field in REQUIRED_FIELDS:
        if field not in output:
            continue
        allowed = vocabulary.get(field, {})
        if output[field] not in allowed:
            errors.append(f"E04_INVALID_LABEL:{field}={output[field]}")

    if candidate_rules is not None and output.get("active_rule") not in candidate_rules:
        errors.append(f"E05_UNSUPPORTED_RULE:{output.get('active_rule')}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--candidate-rules", nargs="*", default=None)
    args = parser.parse_args()

    try:
        output = load_json(args.output)
    except (json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"valid": False, "errors": [f"E01_JSON_PARSE_ERROR:{exc}"]}, indent=2))
        return 1

    vocabulary = load_json(args.vocab)
    errors = validate_output(output, vocabulary, args.candidate_rules)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
