from pathlib import Path
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src" / "03_dataset")
)

sys.path.insert(
    0,
    str(ROOT / "src" / "05_model")
)

from dataset import create_datasets
from model import (
    SemiconRestorationNet,
    count_parameters
)


# ============================================================
# CONFIGURATION
# ============================================================

NOISY_DIR = (
    ROOT /
    "data" /
    "extracted" /
    "train" /
    "train" /
    "NoisyLR"
)

GT_DIR = (
    ROOT /
    "data" /
    "extracted" /
    "train" /
    "train" /
    "GT"
)

CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs" / "training"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


EPOCHS = 100

BATCH_SIZE = 4

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-5

VAL_RATIO = 0.10

NUM_WORKERS = 2

GRAD_CLIP = 1.0

EARLY_STOPPING = 20

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class CharbonnierLoss(nn.Module):

    def __init__(self, epsilon=1e-3):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, prediction, target):

        diff = prediction - target

        loss = torch.sqrt(
            diff * diff +
            self.epsilon ** 2
        )

        return loss.mean()


class EdgeLoss(nn.Module):

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer(
            "sobel_x",
            sobel_x
        )

        self.register_buffer(
            "sobel_y",
            sobel_y
        )

    def forward(self, prediction, target):

        # Edge computation intentionally uses FP32.
        # This prevents the previous:
        #
        # HalfTensor vs FloatTensor
        #
        # error.

        prediction = prediction.float()
        target = target.float()

        pred_x = F.conv2d(
            prediction,
            self.sobel_x.float(),
            padding=1
        )

        pred_y = F.conv2d(
            prediction,
            self.sobel_y.float(),
            padding=1
        )

        target_x = F.conv2d(
            target,
            self.sobel_x.float(),
            padding=1
        )

        target_y = F.conv2d(
            target,
            self.sobel_y.float(),
            padding=1
        )

        pred_edges = torch.sqrt(
            pred_x ** 2 +
            pred_y ** 2 +
            1e-6
        )

        target_edges = torch.sqrt(
            target_x ** 2 +
            target_y ** 2 +
            1e-6
        )

        return F.l1_loss(
            pred_edges,
            target_edges
        )


class SSIMLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):

        prediction = prediction.float()
        target = target.float()

        mu_x = F.avg_pool2d(
            prediction,
            7,
            stride=1,
            padding=3
        )

        mu_y = F.avg_pool2d(
            target,
            7,
            stride=1,
            padding=3
        )

        sigma_x = (
            F.avg_pool2d(
                prediction * prediction,
                7,
                stride=1,
                padding=3
            )
            - mu_x * mu_x
        )

        sigma_y = (
            F.avg_pool2d(
                target * target,
                7,
                stride=1,
                padding=3
            )
            - mu_y * mu_y
        )

        sigma_xy = (
            F.avg_pool2d(
                prediction * target,
                7,
                stride=1,
                padding=3
            )
            - mu_x * mu_y
        )

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (
            (2 * mu_x * mu_y + c1)
            *
            (2 * sigma_xy + c2)
        )

        denominator = (
            (mu_x ** 2 + mu_y ** 2 + c1)
            *
            (sigma_x + sigma_y + c2)
        )

        ssim = numerator / (
            denominator + 1e-8
        )

        return 1.0 - ssim.mean()


charbonnier_loss = CharbonnierLoss()

edge_loss = EdgeLoss().to(DEVICE)

ssim_loss = SSIMLoss()


def restoration_loss(
    prediction,
    target
):

    prediction = prediction.clamp(
        0,
        1
    )

    l_char = charbonnier_loss(
        prediction,
        target
    )

    l_edge = edge_loss(
        prediction,
        target
    )

    l_ssim = ssim_loss(
        prediction,
        target
    )

    total = (
        1.0 * l_char
        +
        0.10 * l_edge
        +
        0.20 * l_ssim
    )

    return (
        total,
        l_char.detach(),
        l_edge.detach(),
        l_ssim.detach()
    )


# ============================================================
# METRICS
# ============================================================

def calculate_psnr(
    prediction,
    target
):

    prediction = prediction.clamp(
        0,
        1
    )

    mse = F.mse_loss(
        prediction,
        target
    )

    if mse.item() < 1e-12:
        return 100.0

    return (
        10.0 *
        torch.log10(
            1.0 / mse
        )
    ).item()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler
):

    model.train()

    total_loss = 0.0
    total_psnr = 0.0

    total_char = 0.0
    total_edge = 0.0
    total_ssim = 0.0

    batches = 0

    for noisy, target in loader:

        noisy = noisy.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if USE_AMP:

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                prediction = model(
                    noisy
                )

                loss, l_char, l_edge, l_ssim = (
                    restoration_loss(
                        prediction,
                        target
                    )
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            prediction = model(
                noisy
            )

            loss, l_char, l_edge, l_ssim = (
                restoration_loss(
                    prediction,
                    target
                )
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            optimizer.step()

        with torch.no_grad():

            prediction = prediction.float()

            batch_psnr = calculate_psnr(
                prediction,
                target
            )

        total_loss += loss.item()
        total_psnr += batch_psnr

        total_char += l_char.item()
        total_edge += l_edge.item()
        total_ssim += l_ssim.item()

        batches += 1

    return {
        "loss": total_loss / batches,
        "psnr": total_psnr / batches,
        "charbonnier": total_char / batches,
        "edge": total_edge / batches,
        "ssim": total_ssim / batches
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0
    total_psnr = 0.0

    batches = 0

    for noisy, target in loader:

        noisy = noisy.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        # Validation can also use AMP.

        if USE_AMP:

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                prediction = model(
                    noisy
                )

                loss, _, _, _ = (
                    restoration_loss(
                        prediction,
                        target
                    )
                )

        else:

            prediction = model(
                noisy
            )

            loss, _, _, _ = (
                restoration_loss(
                    prediction,
                    target
                )
            )

        prediction = prediction.float()

        batch_psnr = calculate_psnr(
            prediction,
            target
        )

        total_loss += loss.item()
        total_psnr += batch_psnr

        batches += 1

    return {
        "loss": total_loss / batches,
        "psnr": total_psnr / batches
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_psnr,
    history
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "history": history
        },
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SEMICON IMAGE RESTORATION - TRAINING")
    print("=" * 80)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    print("\n" + "=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        "NoisyLR:",
        NOISY_DIR
    )

    print(
        "GT:",
        GT_DIR
    )

    train_dataset, val_dataset = (
        create_datasets(
            NOISY_DIR,
            GT_DIR,
            val_ratio=VAL_RATIO,
            seed=SEED
        )
    )

    print(
        "\nTotal:",
        len(train_dataset)
        +
        len(val_dataset)
    )

    print(
        "Training:",
        len(train_dataset)
    )

    print(
        "Validation:",
        len(val_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    print(
        "\nTraining batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n" + "=" * 80)
    print("MODEL")
    print("=" * 80)

    model = SemiconRestorationNet(
        in_channels=1,
        out_channels=1,
        features=64,
        groups=6,
        blocks_per_group=4
    ).to(DEVICE)

    params = count_parameters(
        model
    )

    print(
        "Parameters:",
        f"{params:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    if USE_AMP:

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    else:

        scaler = None

    # ========================================================
    # HISTORY
    # ========================================================

    history = {
        "train_loss": [],
        "train_psnr": [],
        "val_loss": [],
        "val_psnr": [],
        "learning_rate": []
    }

    best_psnr = -float("inf")

    patience_counter = 0

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    print(
        f"Epochs          : {EPOCHS}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"Learning rate   : {LEARNING_RATE}"
    )

    print(
        f"AMP             : {USE_AMP}"
    )

    print(
        f"Early stopping  : {EARLY_STOPPING}"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        start_time = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 80)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 80)

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler
        )

        val_metrics = validate(
            model,
            val_loader
        )

        scheduler.step()

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["train_psnr"].append(
            train_metrics["psnr"]
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["val_psnr"].append(
            val_metrics["psnr"]
        )

        history["learning_rate"].append(
            current_lr
        )

        elapsed = time.time() - start_time

        print(
            f"\nTrain Loss : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Train PSNR : "
            f"{train_metrics['psnr']:.4f} dB"
        )

        print(
            f"Val Loss   : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Val PSNR   : "
            f"{val_metrics['psnr']:.4f} dB"
        )

        print(
            f"Charbonnier: "
            f"{train_metrics['charbonnier']:.6f}"
        )

        print(
            f"Edge Loss  : "
            f"{train_metrics['edge']:.6f}"
        )

        print(
            f"SSIM Loss  : "
            f"{train_metrics['ssim']:.6f}"
        )

        print(
            f"Time       : "
            f"{elapsed:.2f} sec"
        )

        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            CHECKPOINT_DIR / "last_model.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_psnr,
            history
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_metrics["psnr"] > best_psnr:

            best_psnr = val_metrics["psnr"]

            patience_counter = 0

            save_checkpoint(
                CHECKPOINT_DIR / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
                history
            )

            print(
                "\n*** NEW BEST MODEL ***"
            )

            print(
                f"Best Val PSNR: "
                f"{best_psnr:.4f} dB"
            )

        else:

            patience_counter += 1

            print(
                f"\nNo improvement."
                f" Patience "
                f"{patience_counter}/"
                f"{EARLY_STOPPING}"
            )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        np.save(
            OUTPUT_DIR / "training_history.npy",
            history
        )

        with open(
            OUTPUT_DIR / "training_history.json",
            "w"
        ) as file:

            json.dump(
                history,
                file,
                indent=4
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= EARLY_STOPPING:

            print(
                "\nEarly stopping triggered."
            )

            break

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    print(
        f"Best validation PSNR: "
        f"{best_psnr:.4f} dB"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        CHECKPOINT_DIR /
        "best_model.pth"
    )


if __name__ == "__main__":
    main()