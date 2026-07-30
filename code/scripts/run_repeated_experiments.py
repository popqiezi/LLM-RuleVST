#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/training_protocol.yaml"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(
        args.protocol.read_text(encoding="utf-8")
    )
    seeds = protocol["trajectory_comparison"]["seeds"]

    for seed in seeds:
        output_dir = (
            args.output_root
            / args.dataset
            / args.model_config.stem
            / f"seed_{seed}"
        )
        command = [
            sys.executable,
            "scripts/train_baseline.py",
            "--sample-dir", str(args.sample_dir),
            "--dataset", args.dataset,
            "--model-config", str(args.model_config),
            "--training-config", str(args.protocol),
            "--seed", str(seed),
            "--output-dir", str(output_dir),
        ]
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
