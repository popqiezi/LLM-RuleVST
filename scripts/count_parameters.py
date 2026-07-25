#!/usr/bin/env python3
"""Count parameters for an imported PyTorch model.

Usage:
python scripts/count_parameters.py \
  --module your_package.model \
  --factory build_model
"""

from __future__ import annotations

import argparse
import importlib
import json

import torch.nn as nn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="Python module containing the model factory.")
    parser.add_argument("--factory", required=True, help="Callable that returns an nn.Module.")
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    factory = getattr(module, args.factory)
    model = factory()
    if not isinstance(model, nn.Module):
        raise TypeError("The factory must return torch.nn.Module.")

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(json.dumps({"total_parameters": total, "trainable_parameters": trainable}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
