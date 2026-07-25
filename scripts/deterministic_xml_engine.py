#!/usr/bin/env python3
"""Minimal deterministic symbolic baseline for the five rule fields.

This baseline deliberately uses explicit conditions and does not claim to
reproduce every semantic distinction handled by the LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def classify(case: dict[str, Any]) -> dict[str, str]:
    rel = case["relative_navigation_state"]
    bearing = float(rel["relative_bearing_deg"])
    course_diff = abs(float(rel["relative_course_difference_deg"])) % 360
    dcpa = float(rel["DCPA_m"])
    tcpa = float(rel["TCPA_s"])

    risk_developing = dcpa < 100.0 and 0.0 < tcpa < 180.0

    if 165.0 <= course_diff <= 195.0 and abs(bearing) <= 10.0:
        encounter = "head_on"
        rule = "Rule14"
        role = "give_way"
        violation = "late_turn" if risk_developing else "none"
    elif 20.0 <= abs(bearing) <= 112.5:
        encounter = "crossing"
        rule = "Rule15"
        role = "give_way" if bearing > 0 else "stand_on"
        violation = "no_give_way" if risk_developing and role == "give_way" else "none"
    elif abs(bearing) > 112.5:
        encounter = "overtaking"
        rule = "Rule13"
        role = "overtaking_vessel"
        violation = "improper_overtaking" if risk_developing else "none"
    else:
        encounter = "none"
        rule = "Rule5"
        role = "unconstrained"
        violation = "none"

    if violation == "none":
        compliance = "compliant"
    elif risk_developing:
        compliance = "warning"
    else:
        compliance = "violation"

    return {
        "encounter_type": encounter,
        "active_rule": rule,
        "own_role": role,
        "compliance_state": compliance,
        "violation_action": violation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    print(json.dumps(classify(case), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
