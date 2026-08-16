from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

NOISY_DIR = ROOT / "data" / "extracted" / "train" / "train" / "NoisyLR"
GT_DIR = ROOT / "data" / "extracted" / "train" / "train" / "GT"


def inspect_directory(directory):
    files = sorted(directory.glob("*.npy"))

    print(f"\nDirectory: {directory}")
    print(f"Number of files: {len(files)}")

    if not files:
        return []

    print("\nFirst 10 files:")
    for file in files[:10]:
        print(" ", file.name)

    return files


def inspect_file(path):
    data = np.load(path)

    print("\n" + "=" * 70)
    print(f"File: {path.name}")
    print("=" * 70)

    print("Shape :", data.shape)
    print("Dtype :", data.dtype)
    print("Min   :", float(np.min(data)))
    print("Max   :", float(np.max(data)))
    print("Mean  :", float(np.mean(data)))
    print("Std   :", float(np.std(data)))


def main():
    print("=" * 80)
    print("SEMICON IMAGE RESTORATION - DATASET INSPECTION")
    print("=" * 80)

    print("\nProject root:")
    print(ROOT)

    if not NOISY_DIR.exists():
        raise FileNotFoundError(f"NoisyLR directory not found:\n{NOISY_DIR}")

    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT directory not found:\n{GT_DIR}")

    noisy_files = inspect_directory(NOISY_DIR)
    gt_files = inspect_directory(GT_DIR)

    noisy_names = {f.stem for f in noisy_files}
    gt_names = {f.stem for f in gt_files}

    print("\n" + "=" * 80)
    print("PAIRING CHECK")
    print("=" * 80)

    print("NoisyLR files:", len(noisy_files))
    print("GT files    :", len(gt_files))

    missing_gt = noisy_names - gt_names
    missing_noisy = gt_names - noisy_names

    print("Missing GT    :", len(missing_gt))
    print("Missing Noisy :", len(missing_noisy))

    if missing_gt:
        print("\nExamples missing GT:")
        for name in sorted(missing_gt)[:10]:
            print(name)

    if missing_noisy:
        print("\nExamples missing NoisyLR:")
        for name in sorted(missing_noisy)[:10]:
            print(name)

    if noisy_files:
        inspect_file(noisy_files[0])

    if gt_files:
        inspect_file(gt_files[0])

    print("\nInspection completed.")


if __name__ == "__main__":
    main()