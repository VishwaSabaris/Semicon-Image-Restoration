from pathlib import Path
import numpy as np
import torch

from torch.utils.data import Dataset


class SemiconDataset(Dataset):

    def __init__(
        self,
        noisy_dir,
        gt_dir,
        augment=False
    ):
        self.noisy_dir = Path(noisy_dir)
        self.gt_dir = Path(gt_dir)
        self.augment = augment

        noisy_files = sorted(self.noisy_dir.glob("*.npy"))
        gt_files = sorted(self.gt_dir.glob("*.npy"))

        gt_map = {
            file.stem: file
            for file in gt_files
        }

        self.pairs = []

        for noisy_file in noisy_files:

            if noisy_file.stem not in gt_map:
                continue

            self.pairs.append(
                (
                    noisy_file,
                    gt_map[noisy_file.stem]
                )
            )

        if not self.pairs:
            raise RuntimeError(
                "No matching NoisyLR/GT pairs found."
            )

    def __len__(self):
        return len(self.pairs)

    def _augment(self, noisy, gt):

        if np.random.random() < 0.5:
            noisy = np.flip(noisy, axis=0).copy()
            gt = np.flip(gt, axis=0).copy()

        if np.random.random() < 0.5:
            noisy = np.flip(noisy, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()

        if np.random.random() < 0.5:
            noisy = np.rot90(noisy, 1).copy()
            gt = np.rot90(gt, 1).copy()

        return noisy, gt

    def __getitem__(self, index):

        noisy_path, gt_path = self.pairs[index]

        noisy = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        if noisy.ndim == 2:
            noisy = noisy[None, :, :]

        if gt.ndim == 2:
            gt = gt[None, :, :]

        if self.augment:
            noisy, gt = self._augment(
                noisy[0],
                gt[0]
            )

            noisy = noisy[None, :, :]
            gt = gt[None, :, :]

        noisy = torch.from_numpy(noisy.copy())
        gt = torch.from_numpy(gt.copy())

        return noisy, gt


def create_datasets(
    noisy_dir,
    gt_dir,
    val_ratio=0.1,
    seed=42
):

    full_dataset = SemiconDataset(
        noisy_dir=noisy_dir,
        gt_dir=gt_dir,
        augment=False
    )

    total = len(full_dataset)

    val_size = int(total * val_ratio)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    return train_dataset, val_dataset