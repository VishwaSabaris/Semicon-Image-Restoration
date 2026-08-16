from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]

NOISY_DIR = ROOT / "data" / "extracted" / "train" / "train" / "NoisyLR"
GT_DIR = ROOT / "data" / "extracted" / "train" / "train" / "GT"


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


def psnr(pred, target):

    error = mse(pred, target)

    if error <= 1e-12:
        return 100.0

    return float(
        10.0 * torch.log10(
            1.0 / error
        )
    )


def ssim_simple(pred, target):

    pred = pred.flatten()
    target = target.flatten()

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_var = ((pred - pred_mean) ** 2).mean()
    target_var = ((target - target_mean) ** 2).mean()

    covariance = (
        (pred - pred_mean) *
        (target - target_mean)
    ).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (
        (2 * pred_mean * target_mean + c1) *
        (2 * covariance + c2)
    )

    denominator = (
        (pred_mean ** 2 + target_mean ** 2 + c1) *
        (pred_var + target_var + c2)
    )

    return float(numerator / (denominator + 1e-12))


def main():

    noisy_files = sorted(NOISY_DIR.glob("*.npy"))
    gt_map = {
        f.stem: f
        for f in GT_DIR.glob("*.npy")
    }

    psnr_values = []
    ssim_values = []

    print("=" * 80)
    print("BICUBIC BASELINE")
    print("=" * 80)

    for index, noisy_file in enumerate(noisy_files):

        if noisy_file.stem not in gt_map:
            continue

        noisy = np.load(noisy_file).astype(np.float32)
        gt = np.load(
            gt_map[noisy_file.stem]
        ).astype(np.float32)

        noisy_tensor = torch.from_numpy(
            noisy
        ).float()

        gt_tensor = torch.from_numpy(
            gt
        ).float()

        if noisy_tensor.ndim == 2:
            noisy_tensor = noisy_tensor[None, None]

        elif noisy_tensor.ndim == 3:
            noisy_tensor = noisy_tensor[None]

        if gt_tensor.ndim == 2:
            gt_tensor = gt_tensor[None, None]

        elif gt_tensor.ndim == 3:
            gt_tensor = gt_tensor[None]

        prediction = F.interpolate(
            noisy_tensor,
            size=gt_tensor.shape[-2:],
            mode="bicubic",
            align_corners=False
        )

        prediction = prediction.clamp(0, 1)

        psnr_values.append(
            psnr(
                prediction,
                gt_tensor
            )
        )

        ssim_values.append(
            ssim_simple(
                prediction,
                gt_tensor
            )
        )

        if (index + 1) % 500 == 0:
            print(
                f"Processed: {index + 1}"
            )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"Average PSNR: {np.mean(psnr_values):.4f} dB"
    )

    print(
        f"Average SSIM: {np.mean(ssim_values):.6f}"
    )


if __name__ == "__main__":
    main()