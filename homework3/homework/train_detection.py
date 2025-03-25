import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import time

from .datasets.road_dataset import load_data, compute_accuracy
from .metrics import DetectionMetric
from .models import load_model, save_model, ClassificationLoss, RegressionLoss


def train(
        exp_dir: str = "logs",
        model_name: str = "detector",
        num_epoch: int = 30,
        lr: float = 1e-2,
        batch_size: int = 128,
        seed: int = 2024,
        lambda_reg: float = 0.5,
        lambda_cls: float = 3,
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

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size,
                           num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)
    # create loss function and optimizer
    classification_loss = ClassificationLoss(weight=torch.tensor([0.01, 0.495, 0.495]).to(device))
    regression_loss = RegressionLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_step = 0

    for epoch in range(num_epoch):
        model.train()

        for batch in train_data:
            img = batch["image"].to(device)
            seg = batch["track"].to(device)
            depth = batch["depth"].to(device)

            logits, raw_depth = model(img)

            loss_cls = classification_loss(logits, seg)
            loss_reg = regression_loss(raw_depth, depth)
            loss_val = lambda_cls * loss_cls + lambda_reg * loss_reg

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        metric_computer = DetectionMetric()

        with torch.inference_mode():
            model.eval()
            for batch in val_data:
                img = batch["image"].to(device)
                seg = batch["track"].to(device)
                depth = batch["depth"].to(device)

                logits, raw_depth = model(img)

                # Compute metrics using predicted labels and depths
                preds = torch.argmax(logits, dim=1)  # (B, H, W)
                metric_computer.add(preds, seg, depth, raw_depth)

        # Logging
        metrics = metric_computer.compute()
        logger.add_scalar("val/iou", metrics["iou"], global_step)
        logger.add_scalar("val/accuracy", metrics["accuracy"], global_step)
        logger.add_scalar("val/abs_depth_error", metrics["abs_depth_error"], global_step)
        logger.add_scalar("val/tp_depth_error", metrics["tp_depth_error"], global_step)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:2d} | IOU: {metrics['iou']:.4f} | "
                  f"Accuracy: {metrics['accuracy']:.4f} | "
                  f"Depth Err: {metrics['abs_depth_error']:.4f} | "
                  f"TP Depth Err: {metrics['tp_depth_error']:.4f}")

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=2)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    start = time.time()

    # pass all arguments to train
    train(**vars(parser.parse_args()))

    end = time.time()
    formatted_time = str(timedelta(seconds=int(end - start)))  # Convert to hh:mm:ss
    print(f"Training took {formatted_time}")
