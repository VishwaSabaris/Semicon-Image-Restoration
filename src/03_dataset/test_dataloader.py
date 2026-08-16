from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src" / "03_dataset")
)

from dataset import create_datasets


NOISY_DIR = ROOT / "data" / "extracted" / "train" / "train" / "NoisyLR"
GT_DIR = ROOT / "data" / "extracted" / "train" / "train" / "GT"


def main():

    print("=" * 80)
    print("DATALOADER TEST")
    print("=" * 80)

    train_dataset, val_dataset = create_datasets(
        NOISY_DIR,
        GT_DIR,
        val_ratio=0.1
    )

    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print("Training samples :", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    noisy, gt = next(iter(loader))

    print("\nBatch information")
    print("-" * 40)

    print("Noisy shape :", noisy.shape)
    print("GT shape    :", gt.shape)

    print("Noisy dtype :", noisy.dtype)
    print("GT dtype    :", gt.dtype)

    print("Noisy min   :", noisy.min().item())
    print("Noisy max   :", noisy.max().item())

    print("GT min      :", gt.min().item())
    print("GT max      :", gt.max().item())

    print("\nDataloader test successful.")


if __name__ == "__main__":
    main()