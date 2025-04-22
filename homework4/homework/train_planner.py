"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model, RegressionLoss, CNNPlanner

DEFAULT_EXP_DIR: str = 'logs'
DEFAULT_NUM_EPOCH: int = 50
DEFAULT_LR: float = 1e-3
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_SEED: int = 2024
DEFAULT_TRANSFORM_PIPELINE = 'aug'
DEFAULT_CROP_SIZE: tuple[int, int] = (128, 128)
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_NUM_WORKERS: int = 2


def train(
        model_name: str,
        exp_dir: str = DEFAULT_EXP_DIR,
        num_epoch: int = DEFAULT_NUM_EPOCH,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = DEFAULT_SEED,
        transform_pipeline: str = DEFAULT_TRANSFORM_PIPELINE,
        crop_size: tuple[int, int] = DEFAULT_CROP_SIZE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Set up the dataloaders
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size,
                           num_workers=num_workers, transform_pipeline=transform_pipeline, crop_size=crop_size)
    val_data = load_data("drive_data/val", shuffle=False,
                         num_workers=num_workers, transform_pipeline=transform_pipeline, crop_size=crop_size)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir=log_dir.absolute().as_posix())
    # visualize graph
    if not isinstance(model, CNNPlanner):
        dummy_left = torch.zeros(1, model.n_track, 2, device=device)
        dummy_right = torch.zeros(1, model.n_track, 2, device=device)
        logger.add_graph(model, [dummy_left, dummy_right])
    else:
        H, W = crop_size
        dummy_img = torch.zeros(1, 3, H, W, device=device)
        logger.add_graph(model, [dummy_img])
    logger.flush()

    # create loss function and optimizer
    regression_loss = RegressionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    # training loop
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for batch in train_data:
            if isinstance(model, CNNPlanner):
                # aug/default pipelines produce 'image' + ego‑state (but CNN ignores ego‑state)
                img = batch["image"].to(device)
                preds = model(image=img)
            else:
                # state_only or default pipelines produce only track/waypoints
                tl = batch["track_left"].to(device)
                tr = batch["track_right"].to(device)
                preds = model(track_left=tl, track_right=tr)

            tgt = batch["waypoints"].to(device)  # (B, n_waypoints, 2)
            mask = batch["waypoints_mask"].to(device).float()  # (B, n_waypoints)

            loss_reg = regression_loss(preds, tgt) # (B, n_waypoints, 2)
            per_pt_l1 = loss_reg.sum(dim=-1)  # (B, n_waypoints)
            loss_val = (per_pt_l1 * mask).sum() / (mask.sum() + 1e-6)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            train_loss += loss_val.item()
            logger.add_scalar(f"{model_name}/train/loss", loss_val.item(), global_step)
            global_step += 1

        metric_computer = PlannerMetric()

        with torch.inference_mode():
            model.eval()
            for batch in val_data:
                if isinstance(model, CNNPlanner):
                    inputs = batch['image'].to(device)
                    preds = model(image=inputs)
                else:
                    tl = batch['track_left'].to(device)
                    tr = batch['track_right'].to(device)
                    preds = model(track_left=tl, track_right=tr)

                tgt = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device).bool()

                metric_computer.add(preds, tgt, mask)

        # Logging
        metrics = metric_computer.compute()
        logger.add_scalar(f"{model_name}/val/l1_error", metrics["l1_error"], global_step)
        logger.add_scalar(f"{model_name}/val/longitudinal_error", metrics["longitudinal_error"], global_step)
        logger.add_scalar(f"{model_name}/val/lateral_error", metrics["lateral_error"], global_step)
        logger.add_scalar(f"{model_name}/val/num_samples", metrics["num_samples"], global_step)
        logger.flush()

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:2d} | L1 Error: {metrics['l1_error']:.4f} | "
                  f"Longitudinal Error: {metrics['longitudinal_error']:.4f} | "
                  f"Lateral Error: {metrics['lateral_error']:.4f} | "
                  f"Num samples: {metrics['num_samples']}")
            file_name = f"{model_name}_{epoch + 1}.th"
            torch.save(model.state_dict(), file_name)
            print(f"Model saved to {log_dir / file_name} on epoch {epoch + 1}")

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True,
                        choices=["mlp_planner","transformer_planner","cnn_planner"])
    parser.add_argument("--exp_dir", type=str, default=DEFAULT_EXP_DIR)
    parser.add_argument("--num_epoch", type=int, default=DEFAULT_NUM_EPOCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--transform_pipeline", type=str, default=DEFAULT_TRANSFORM_PIPELINE,
                        choices=["state_only","default","aug"])
    parser.add_argument("--crop_size", type=tuple[int, int], nargs=2, default=DEFAULT_CROP_SIZE,
                        help="Required for aug pipeline, e.g. --crop_size 128 128")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)

    # optional: additional model hyperparamters
    parser.add_argument("--n_track", type=int, default=10)
    parser.add_argument("--n_waypoints", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)

    start = time.time()

    # pass all arguments to train
    train(**vars(parser.parse_args()))

    end = time.time()
    formatted_time = str(timedelta(seconds=int(end - start)))  # Convert to hh:mm:ss
    print(f"Training took {formatted_time}")
