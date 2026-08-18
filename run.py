import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "src" / "05_model"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"

sys.path.insert(0, str(MODEL_DIR))

from model import SemiconRestorationNet


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = 64
GROUPS = 6
BLOCKS_PER_GROUP = 4


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 80)
print("SEMICON IMAGE RESTORATION - DEMO")
print("=" * 80)

print()
print("Device :", DEVICE)

if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
    print("CUDA   :", torch.version.cuda)
else:
    print("GPU    : None")

print()


# ============================================================
# CHECK CHECKPOINT
# ============================================================

if not CHECKPOINT.exists():
    print("ERROR: Checkpoint not found:")
    print(CHECKPOINT)
    sys.exit(1)


# ============================================================
# GET INPUT FILE
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print(
        r'python demo.py "C:\path\to\image.npy"'
    )
    sys.exit(1)

input_path = Path(sys.argv[1])

if not input_path.exists():
    print("ERROR: Input file does not exist:")
    print(input_path)
    sys.exit(1)

if input_path.suffix.lower() != ".npy":
    print("ERROR: Input file must be a .npy file.")
    sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 80)
print("LOADING MODEL")
print("=" * 80)

print()
print("Checkpoint:")
print(CHECKPOINT)
print()

model = SemiconRestorationNet(
    in_channels=1,
    out_channels=1,
    features=FEATURES,
    groups=GROUPS,
    blocks_per_group=BLOCKS_PER_GROUP
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


# ============================================================
# HANDLE DIFFERENT CHECKPOINT FORMATS
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# Remove possible "module." prefix
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


# ============================================================
# LOAD WEIGHTS
# ============================================================

try:

    model.load_state_dict(
        clean_state_dict,
        strict=True
    )

except RuntimeError as e:

    print()
    print("=" * 80)
    print("CHECKPOINT / MODEL ARCHITECTURE MISMATCH")
    print("=" * 80)
    print()
    print(e)
    print()
    print("Your current model architecture is:")
    print(
        f"features={FEATURES}, "
        f"groups={GROUPS}, "
        f"blocks_per_group={BLOCKS_PER_GROUP}"
    )
    print()
    print(
        "The checkpoint was most likely trained using a different "
        "architecture."
    )

    sys.exit(1)


model = model.to(DEVICE)
model.eval()


# ============================================================
# MODEL INFORMATION
# ============================================================

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Model loaded successfully.")
print("Features           :", FEATURES)
print("Residual groups    :", GROUPS)
print("Blocks/group       :", BLOCKS_PER_GROUP)
print("Trainable params   :", f"{parameters:,}")

print()


# ============================================================
# LOAD NPY
# ============================================================

print("=" * 80)
print("LOADING INPUT")
print("=" * 80)

print()
print("Input:")
print(input_path)

noisy = np.load(input_path)

print()
print("Original shape :", noisy.shape)
print("Dtype          :", noisy.dtype)
print("Min            :", float(np.min(noisy)))
print("Max            :", float(np.max(noisy)))
print("Mean           :", float(np.mean(noisy)))


# ============================================================
# PREPARE IMAGE
# ============================================================

noisy = noisy.astype(np.float32)


# Expected input is 128 x 128
if noisy.ndim == 2:

    input_array = noisy

elif noisy.ndim == 3:

    # Handle (1,H,W)
    if noisy.shape[0] == 1:
        input_array = noisy[0]

    # Handle (H,W,1)
    elif noisy.shape[-1] == 1:
        input_array = noisy[:, :, 0]

    else:
        print()
        print("ERROR: Unsupported 3D NPY shape:", noisy.shape)
        sys.exit(1)

else:

    print()
    print("ERROR: Unsupported NPY dimensions:", noisy.ndim)
    sys.exit(1)


# ============================================================
# NORMALIZATION
# ============================================================

# The training dataset uses float images.
# Keep the same [0,1] representation expected by the model.

if np.max(input_array) > 1.0 or np.min(input_array) < 0.0:

    print()
    print("Input is outside [0,1]. Clipping to [0,1].")

    input_array = np.clip(
        input_array,
        0.0,
        1.0
    )


# ============================================================
# CREATE TENSOR
# ============================================================

tensor = torch.from_numpy(
    input_array
).float()

tensor = tensor.unsqueeze(0).unsqueeze(0)

tensor = tensor.to(DEVICE)

print()
print("Tensor shape   :", tensor.shape)


# ============================================================
# MODEL INFERENCE
# ============================================================

print()
print("=" * 80)
print("RUNNING RESTORATION")
print("=" * 80)
print()

with torch.inference_mode():

    restored = model(tensor)


# ============================================================
# CONVERT OUTPUT
# ============================================================

restored = restored.squeeze().detach().cpu().numpy()

restored = np.clip(
    restored,
    0.0,
    1.0
)

print("Output shape   :", restored.shape)
print("Output min     :", float(np.min(restored)))
print("Output max     :", float(np.max(restored)))
print("Output mean    :", float(np.mean(restored)))


# ============================================================
# SAVE OUTPUT DIRECTORY
# ============================================================

output_dir = PROJECT_ROOT / "outputs" / "demo"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE RESTORED NPY
# ============================================================

restored_npy = output_dir / (
    input_path.stem + "_restored.npy"
)

np.save(
    restored_npy,
    restored.astype(np.float32)
)


# ============================================================
# SAVE NOISY PNG
# ============================================================

noisy_png = output_dir / (
    input_path.stem + "_noisy.png"
)

plt.imsave(
    noisy_png,
    input_array,
    cmap="gray",
    vmin=0,
    vmax=1
)


# ============================================================
# SAVE RESTORED PNG
# ============================================================

restored_png = output_dir / (
    input_path.stem + "_restored.png"
)

plt.imsave(
    restored_png,
    restored,
    cmap="gray",
    vmin=0,
    vmax=1
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_png = output_dir / (
    input_path.stem + "_comparison.png"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)

axes[0].imshow(
    input_array,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0].set_title(
    "Input - Noisy LR"
)

axes[0].axis("off")


axes[1].imshow(
    restored,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "Output - Restored"
)

axes[1].axis("off")


plt.tight_layout()

plt.savefig(
    comparison_png,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 80)
print("DEMO COMPLETED")
print("=" * 80)

print()
print("Input NPY:")
print(input_path)

print()
print("Generated files:")

print("Noisy image:")
print(noisy_png)

print()
print("Restored image:")
print(restored_png)

print()
print("Restored NPY:")
print(restored_npy)

print()
print("Comparison:")
print(comparison_png)

print()
print("=" * 80)
