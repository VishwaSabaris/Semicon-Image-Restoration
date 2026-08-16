from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

NOISY_DIR = ROOT / "data" / "extracted" / "train" / "train" / "NoisyLR"
GT_DIR = ROOT / "data" / "extracted" / "train" / "train" / "GT"


def statistics(files):
    mins = []
    maxs = []
    means = []
    stds = []

    for file in files:
        image = np.load(file).astype(np.float32)

        mins.append(image.min())
        maxs.append(image.max())
        means.append(image.mean())
        stds.append(image.std())

    return {
        "min": float(np.min(mins)),
        "max": float(np.max(maxs)),
        "mean": float(np.mean(means)),
        "std": float(np.mean(stds)),
    }


def main():
    print("=" * 80)
    print("SEMICON IMAGE RESTORATION - DATASET ANALYSIS")
    print("=" * 80)

    noisy_files = sorted(NOISY_DIR.glob("*.npy"))
    gt_files = sorted(GT_DIR.glob("*.npy"))

    print("\nNoisyLR files:", len(noisy_files))
    print("GT files    :", len(gt_files))

    if not noisy_files or not gt_files:
        raise RuntimeError("Dataset is empty.")

    noisy_stats = statistics(noisy_files)
    gt_stats = statistics(gt_files)

    print("\n" + "=" * 80)
    print("NOISYLR STATISTICS")
    print("=" * 80)

    for key, value in noisy_stats.items():
        print(f"{key:10s}: {value:.6f}")

    print("\n" + "=" * 80)
    print("GROUND TRUTH STATISTICS")
    print("=" * 80)

    for key, value in gt_stats.items():
        print(f"{key:10s}: {value:.6f}")

    noisy_shape = np.load(noisy_files[0]).shape
    gt_shape = np.load(gt_files[0]).shape

    print("\n" + "=" * 80)
    print("SHAPES")
    print("=" * 80)

    print("NoisyLR shape:", noisy_shape)
    print("GT shape     :", gt_shape)

    print("\nScale factor:")

    if len(noisy_shape) >= 2 and len(gt_shape) >= 2:
        scale_h = gt_shape[-2] / noisy_shape[-2]
        scale_w = gt_shape[-1] / noisy_shape[-1]

        print("Height scale:", scale_h)
        print("Width scale :", scale_w)


if __name__ == "__main__":
    main()