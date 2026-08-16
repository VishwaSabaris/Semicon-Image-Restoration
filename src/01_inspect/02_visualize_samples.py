from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

NOISY_DIR = ROOT / "data" / "extracted" / "train" / "train" / "NoisyLR"
GT_DIR = ROOT / "data" / "extracted" / "train" / "train" / "GT"

OUTPUT_DIR = ROOT / "outputs" / "inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_image(image):
    image = image.astype(np.float32)

    min_val = image.min()
    max_val = image.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(image)

    return (image - min_val) / (max_val - min_val)


def main():
    noisy_files = sorted(NOISY_DIR.glob("*.npy"))
    gt_files = sorted(GT_DIR.glob("*.npy"))

    if not noisy_files:
        raise RuntimeError("No NoisyLR files found.")

    if not gt_files:
        raise RuntimeError("No GT files found.")

    gt_map = {file.stem: file for file in gt_files}

    samples = []

    for noisy_file in noisy_files:
        if noisy_file.stem in gt_map:
            samples.append((noisy_file, gt_map[noisy_file.stem]))

    samples = samples[:6]

    fig, axes = plt.subplots(
        len(samples),
        2,
        figsize=(10, 4 * len(samples))
    )

    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (noisy_file, gt_file) in enumerate(samples):

        noisy = np.load(noisy_file)
        gt = np.load(gt_file)

        axes[row, 0].imshow(normalize_image(noisy), cmap="gray")
        axes[row, 0].set_title(
            f"NoisyLR\n{noisy.shape}"
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(normalize_image(gt), cmap="gray")
        axes[row, 1].set_title(
            f"Ground Truth\n{gt.shape}"
        )
        axes[row, 1].axis("off")

    plt.tight_layout()

    output = OUTPUT_DIR / "dataset_samples.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved visualization to:\n{output}")


if __name__ == "__main__":
    main()