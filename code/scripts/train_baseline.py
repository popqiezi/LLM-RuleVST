#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.baseline_models import build_baseline
from scripts.common.seed_utils import set_global_seed


class FinalSampleDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        root: Path,
    ) -> None:
        self.index = index.reset_index(drop=True)
        self.root = root

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        row = self.index.iloc[item]
        with np.load(
            self.root / row.sample_file,
            allow_pickle=False,
        ) as data:
            motion = torch.from_numpy(
                data["historical_motion_standardized"]
            ).float()
            trajectory = torch.from_numpy(
                data["future_trajectory_standardized"]
            ).float()
            cri = torch.from_numpy(
                data["future_cri"]
            ).float()
        return motion, trajectory, cri


def epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    trajectory_weight: float,
    cri_weight: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    mse = nn.MSELoss()

    for motion, trajectory, cri in loader:
        motion = motion.to(device)
        trajectory = trajectory.to(device)
        cri = cri.to(device)

        with torch.set_grad_enabled(training):
            pred_trajectory, pred_cri = model(motion)
            loss = (
                trajectory_weight
                * mse(pred_trajectory, trajectory)
                + cri_weight * mse(pred_cri, cri)
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

        batch = motion.shape[0]
        total += float(loss.detach()) * batch
        count += batch
    return total / max(count, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training_protocol.yaml"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    set_global_seed(args.seed)
    with args.model_config.open("r", encoding="utf-8") as file:
        model_cfg = yaml.safe_load(file)
    with args.training_config.open("r", encoding="utf-8") as file:
        training_cfg = yaml.safe_load(file)["training"]

    index = pd.read_csv(
        args.sample_dir / "final_sample_index.csv"
    )
    index = index[index["dataset"] == args.dataset]
    train_index = index[index["split"] == "train"]
    val_index = index[index["split"] == "validation"]
    if train_index.empty or val_index.empty:
        raise RuntimeError(
            "Training and validation samples are required"
        )

    model_name = model_cfg.pop("name")
    for descriptive_key in [
        "historical_length",
        "trajectory_output_dimension",
        "cri_output_dimension",
    ]:
        model_cfg.pop(descriptive_key, None)
    model = build_baseline(model_name, **model_cfg)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    train_loader = DataLoader(
        FinalSampleDataset(train_index, args.sample_dir),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        FinalSampleDataset(val_index, args.sample_dir),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_cfg["epochs"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []

    for epoch_number in range(
        1, int(training_cfg["epochs"]) + 1
    ):
        train_loss = epoch(
            model,
            train_loader,
            optimizer,
            device,
            float(training_cfg["trajectory_loss_weight"]),
            float(training_cfg["cri_loss_weight"]),
        )
        val_loss = epoch(
            model,
            val_loader,
            None,
            device,
            float(training_cfg["trajectory_loss_weight"]),
            float(training_cfg["cri_loss_weight"]),
        )
        scheduler.step()
        history.append({
            "epoch": epoch_number,
            "train_total_loss": train_loss,
            "validation_total_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_name": model_name,
                "model_config": model_cfg,
                "dataset": args.dataset,
                "seed": args.seed,
                "epoch": epoch_number,
                "validation_total_loss": val_loss,
                "state_dict": model.state_dict(),
            }, args.output_dir / "best_checkpoint.pt")

    pd.DataFrame(history).to_csv(
        args.output_dir / "training_history.csv",
        index=False,
    )
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps({
            "model_name": model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "best_validation_total_loss": best_loss,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
