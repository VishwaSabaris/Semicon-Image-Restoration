"""
SEMICON IMAGE RESTORATION
Evaluation Script with PSNR, SSIM and LPIPS

Evaluates:
    1. Bicubic baseline
    2. Trained restoration model

Metrics:
    - PSNR       Higher is better
    - SSIM       Higher is better
    - LPIPS      Lower is better

Outputs:
    outputs/evaluation_current/
        comparison_XXXXXX.png
        model_XXXXXX.png

Results:
    outputs/evaluation_current/evaluation_results.json
"""

import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from PIL import Image

# ---------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

SRC_DIR = PROJECT_ROOT / "src"
DATASET_DIR = SRC_DIR / "03_dataset"
MODEL_DIR = SRC_DIR / "05_model"

sys.path.insert(0, str(DATASET_DIR))
sys.path.insert(0, str(MODEL_DIR))

# ---------------------------------------------------------------------
# IMPORT PROJECT MODULES
# ---------------------------------------------------------------------

try:
    from dataset import SemiconDataset
except ImportError as e:
    print("\nERROR: Could not import SemiconDataset")
    print("Expected file:")
    print(DATASET_DIR / "dataset.py")
    print(f"\nOriginal error: {e}")
    sys.exit(1)

try:
    from model import SemiconRestorationNet
except ImportError as e:
    print("\nERROR: Could not import SemiconRestorationNet")
    print("Expected file:")
    print(MODEL_DIR / "model.py")
    print(f"\nOriginal error: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------

try:
    import lpips
except ImportError:
    print("\nERROR: LPIPS is not installed.")
    print("\nInstall it with:")
    print("    pip install lpips")
    print("\nThen run this script again.")
    sys.exit(1)


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

BATCH_SIZE = 4
NUM_WORKERS = 0

VALIDATION_RATIO = 0.10

# Model configuration used during training
FEATURES = 96
RESIDUAL_GROUPS = 8
BLOCKS_PER_GROUP = 6

# Paths
NOISY_DIR = (
    PROJECT_ROOT
    / "data"
    / "extracted"
    / "train"
    / "train"
    / "NoisyLR"
)

GT_DIR = (
    PROJECT_ROOT
    / "data"
    / "extracted"
    / "train"
    / "train"
    / "GT"
)

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation_current"

RESULTS_PATH = OUTPUT_DIR / "evaluation_results.json"


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic validation
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# PRINT HEADER
# =====================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


# =====================================================================
# PSNR
# =====================================================================

def calculate_psnr(prediction, target):
    """
    prediction: [B,1,H,W] or [B,C,H,W]
    target:     same shape

    Images are assumed to be in [0,1].
    """

    prediction = torch.clamp(prediction, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    mse = torch.mean((prediction - target) ** 2)

    if mse.item() <= 1e-12:
        return float("inf")

    psnr = 10.0 * torch.log10(1.0 / mse)

    return psnr.item()


# =====================================================================
# SSIM
# =====================================================================

def gaussian_kernel(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(
        window_size,
        dtype=torch.float32,
        device=device
    )

    coords -= window_size // 2

    gaussian = torch.exp(
        -(coords ** 2) / (2 * sigma ** 2)
    )

    gaussian /= gaussian.sum()

    kernel = gaussian[:, None] @ gaussian[None, :]

    return kernel


def calculate_ssim(
    prediction,
    target,
    window_size=11,
    sigma=1.5,
    data_range=1.0
):
    """
    SSIM implementation for grayscale/multichannel images.
    """

    prediction = torch.clamp(prediction, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    channels = prediction.shape[1]

    kernel = gaussian_kernel(
        window_size=window_size,
        sigma=sigma,
        device=prediction.device
    )

    kernel = kernel.expand(
        channels,
        1,
        window_size,
        window_size
    )

    padding = window_size // 2

    mu_x = F.conv2d(
        prediction,
        kernel,
        padding=padding,
        groups=channels
    )

    mu_y = F.conv2d(
        target,
        kernel,
        padding=padding,
        groups=channels
    )

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(
            prediction * prediction,
            kernel,
            padding=padding,
            groups=channels
        )
        - mu_x_sq
    )

    sigma_y_sq = (
        F.conv2d(
            target * target,
            kernel,
            padding=padding,
            groups=channels
        )
        - mu_y_sq
    )

    sigma_xy = (
        F.conv2d(
            prediction * target,
            kernel,
            padding=padding,
            groups=channels
        )
        - mu_xy
    )

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    numerator = (
        (2 * mu_xy + C1)
        * (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x_sq + mu_y_sq + C1)
        * (sigma_x_sq + sigma_y_sq + C2)
    )

    ssim_map = numerator / (denominator + 1e-12)

    return ssim_map.mean().item()


# =====================================================================
# LPIPS PREPARATION
# =====================================================================

def prepare_for_lpips(image):
    """
    LPIPS expects:
        [B,3,H,W]
        values in [-1,1]

    Our dataset is grayscale:
        [B,1,H,W]
        values in [0,1]

    Therefore:
        grayscale -> 3 channels
        [0,1] -> [-1,1]
    """

    image = torch.clamp(image, 0.0, 1.0)

    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)

    elif image.shape[1] == 2:
        image = image[:, :1].repeat(1, 3, 1, 1)

    elif image.shape[1] > 3:
        image = image[:, :3]

    image = image * 2.0 - 1.0

    return image


# =====================================================================
# IMAGE SAVING
# =====================================================================

def tensor_to_uint8(image):
    image = torch.clamp(image, 0.0, 1.0)

    image = image.detach().cpu()

    if image.dim() == 4:
        image = image[0]

    if image.shape[0] == 1:
        image = image[0]

    else:
        image = image.permute(1, 2, 0)

    image = (image.numpy() * 255.0).round().astype(np.uint8)

    return image


def save_tensor_image(image, path):
    """
    Save tensor as PNG.
    """

    image = torch.clamp(image.detach().cpu(), 0.0, 1.0)

    if image.dim() == 4:
        image = image[0]

    save_image(
        image,
        str(path)
    )


def create_comparison_image(
    noisy,
    bicubic,
    prediction,
    ground_truth,
    path,
    sample_index,
    bicubic_psnr,
    model_psnr,
    bicubic_ssim,
    model_ssim,
    bicubic_lpips,
    model_lpips
):
    """
    Creates:

        NoisyLR | Bicubic | Model | Ground Truth
    """

    noisy_img = tensor_to_uint8(noisy)
    bicubic_img = tensor_to_uint8(bicubic)
    prediction_img = tensor_to_uint8(prediction)
    gt_img = tensor_to_uint8(ground_truth)

    # Convert grayscale to RGB for PIL
    if noisy_img.ndim == 2:
        noisy_img = np.stack(
            [noisy_img] * 3,
            axis=-1
        )

    if bicubic_img.ndim == 2:
        bicubic_img = np.stack(
            [bicubic_img] * 3,
            axis=-1
        )

    if prediction_img.ndim == 2:
        prediction_img = np.stack(
            [prediction_img] * 3,
            axis=-1
        )

    if gt_img.ndim == 2:
        gt_img = np.stack(
            [gt_img] * 3,
            axis=-1
        )

    images = [
        noisy_img,
        bicubic_img,
        prediction_img,
        gt_img
    ]

    labels = [
        "NoisyLR",
        f"Bicubic\nPSNR {bicubic_psnr:.2f}\nSSIM {bicubic_ssim:.4f}\nLPIPS {bicubic_lpips:.4f}",
        f"Model\nPSNR {model_psnr:.2f}\nSSIM {model_ssim:.4f}\nLPIPS {model_lpips:.4f}",
        "Ground Truth"
    ]

    # Resize all images to same size
    height = max(img.shape[0] for img in images)

    resized = []

    for img in images:
        pil = Image.fromarray(img)

        if pil.height != height:
            width = int(
                pil.width * height / pil.height
            )

            pil = pil.resize(
                (width, height),
                Image.Resampling.BICUBIC
            )

        resized.append(
            np.array(pil)
        )

    total_width = sum(
        img.shape[1]
        for img in resized
    )

    label_height = 100

    canvas = Image.new(
        "RGB",
        (total_width, height + label_height),
        "white"
    )

    x = 0

    for img, label in zip(resized, labels):

        pil = Image.fromarray(img)

        canvas.paste(
            pil,
            (x, label_height)
        )

        x += pil.width

    canvas.save(
        str(path)
    )


# =====================================================================
# CHECKPOINT LOADING
# =====================================================================

def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint:")
    print(checkpoint_path)

    if not checkpoint_path.exists():
        print()
        print("ERROR: Checkpoint not found.")
        print(f"Expected:")
        print(checkpoint_path)
        print()
        sys.exit(1)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    print()
    print("Checkpoint type:", type(checkpoint))

    # -------------------------------------------------------------
    # Different possible checkpoint formats
    # -------------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

            model.load_state_dict(
                state_dict,
                strict=True
            )

            epoch = checkpoint.get(
                "epoch",
                "unknown"
            )

            best_psnr = checkpoint.get(
                "best_psnr",
                "unknown"
            )

            print(f"Checkpoint epoch : {epoch}")
            print(f"Training best PSNR: {best_psnr}")

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"],
                strict=True
            )

        elif all(
            isinstance(v, torch.Tensor)
            for v in checkpoint.values()
        ):

            model.load_state_dict(
                checkpoint,
                strict=True
            )

        else:
            print()
            print("ERROR: Unknown checkpoint dictionary format.")
            print("Checkpoint keys:")
            print(list(checkpoint.keys()))
            sys.exit(1)

    else:
        print()
        print("ERROR: Unsupported checkpoint format.")
        sys.exit(1)

    print("Checkpoint loaded successfully.")


# =====================================================================
# DATASET
# =====================================================================

def create_dataset():
    print_header("CREATING DATASET")

    if not NOISY_DIR.exists():
        print("ERROR: NoisyLR directory not found:")
        print(NOISY_DIR)
        sys.exit(1)

    if not GT_DIR.exists():
        print("ERROR: Ground Truth directory not found:")
        print(GT_DIR)
        sys.exit(1)

    print("NoisyLR:")
    print(NOISY_DIR)

    print()
    print("Ground Truth:")
    print(GT_DIR)

    try:
        dataset = SemiconDataset(
            noisy_dir=str(NOISY_DIR),
            gt_dir=str(GT_DIR)
        )

    except TypeError:

        # Some versions may use positional arguments
        dataset = SemiconDataset(
            str(NOISY_DIR),
            str(GT_DIR)
        )

    print()
    print("Dataset initialized successfully.")
    print("Total samples:", len(dataset))

    return dataset


# =====================================================================
# MAIN EVALUATION
# =====================================================================

def main():

    set_seed(SEED)

    print_header(
        "SEMICON IMAGE RESTORATION - "
        "PSNR / SSIM / LPIPS EVALUATION"
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    else:
        print(
            "WARNING: CUDA is not available."
        )

    # -------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Current output directory:")
    print(OUTPUT_DIR)

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    dataset = create_dataset()

    total_samples = len(dataset)

    validation_size = int(
        total_samples * VALIDATION_RATIO
    )

    training_size = (
        total_samples - validation_size
    )

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_dataset, validation_dataset = random_split(
        dataset,
        [
            training_size,
            validation_size
        ],
        generator=generator
    )

    print()
    print("Dataset split:")
    print(
        f"Training samples   : {len(train_dataset)}"
    )

    print(
        f"Validation samples : {len(validation_dataset)}"
    )

    # -------------------------------------------------------------
    # DataLoader
    # -------------------------------------------------------------

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    print()
    print("Validation batches:", len(validation_loader))
    print("Batch size:", BATCH_SIZE)

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    print_header("CREATING MODEL")

    try:

        model = SemiconRestorationNet(
            features=FEATURES,
            num_groups=RESIDUAL_GROUPS,
            blocks_per_group=BLOCKS_PER_GROUP
        )

    except TypeError:

        try:

            model = SemiconRestorationNet(
                features=FEATURES,
                residual_groups=RESIDUAL_GROUPS,
                blocks_per_group=BLOCKS_PER_GROUP
            )

        except TypeError:

            model = SemiconRestorationNet()

    model = model.to(device)

    model.eval()

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        f"{total_parameters:,}"
    )

    # -------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------

    print_header("LOADING BEST CHECKPOINT")

    load_checkpoint(
        model,
        CHECKPOINT_PATH,
        device
    )

    model.eval()

    # -------------------------------------------------------------
    # LPIPS model
    # -------------------------------------------------------------

    print_header("LOADING LPIPS")

    print(
        "LPIPS network: AlexNet"
    )

    print(
        "LPIPS expects 3-channel images in [-1, 1]."
    )

    lpips_model = lpips.LPIPS(
        net="alex"
    )

    lpips_model = lpips_model.to(device)

    lpips_model.eval()

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------

    print_header(
        "STARTING CURRENT MODEL EVALUATION"
    )

    print(
        "IMPORTANT:"
    )

    print(
        "The old outputs/evaluation folder is NOT being used."
    )

    print(
        "Fresh predictions will be saved to:"
    )

    print(OUTPUT_DIR)

    print()

    total_bicubic_psnr = 0.0
    total_model_psnr = 0.0

    total_bicubic_ssim = 0.0
    total_model_ssim = 0.0

    total_bicubic_lpips = 0.0
    total_model_lpips = 0.0

    sample_count = 0

    saved_images = 0

    start_time = time.time()

    with torch.inference_mode():

        for batch_index, batch in enumerate(
            validation_loader
        ):

            # -----------------------------------------------------
            # Dataset output handling
            # -----------------------------------------------------

            if isinstance(batch, dict):

                noisy = batch.get(
                    "noisy"
                )

                if noisy is None:
                    noisy = batch.get(
                        "noisy_lr"
                    )

                ground_truth = batch.get(
                    "gt"
                )

                if ground_truth is None:
                    ground_truth = batch.get(
                        "ground_truth"
                    )

                if ground_truth is None:
                    ground_truth = batch.get(
                        "target"
                    )

                if noisy is None or ground_truth is None:
                    print()
                    print(
                        "ERROR: Dataset dictionary does not contain "
                        "recognized keys."
                    )

                    print(
                        "Available keys:",
                        batch.keys()
                    )

                    sys.exit(1)

            elif isinstance(batch, (list, tuple)):

                if len(batch) < 2:

                    print(
                        "ERROR: Dataset must return "
                        "(noisy, ground_truth)."
                    )

                    sys.exit(1)

                noisy = batch[0]
                ground_truth = batch[1]

            else:

                print(
                    "ERROR: Unknown DataLoader output format."
                )

                print(
                    type(batch)
                )

                sys.exit(1)

            noisy = noisy.to(
                device,
                non_blocking=True
            ).float()

            ground_truth = ground_truth.to(
                device,
                non_blocking=True
            ).float()

            # -----------------------------------------------------
            # Model prediction
            # -----------------------------------------------------

            prediction = model(noisy)

            prediction = prediction.float()

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0
            )

            ground_truth = torch.clamp(
                ground_truth,
                0.0,
                1.0
            )

            # -----------------------------------------------------
            # Bicubic baseline
            # -----------------------------------------------------

            bicubic = F.interpolate(
                noisy,
                size=ground_truth.shape[-2:],
                mode="bicubic",
                align_corners=False
            )

            bicubic = torch.clamp(
                bicubic,
                0.0,
                1.0
            )

            # -----------------------------------------------------
            # Batch PSNR
            # -----------------------------------------------------

            batch_bicubic_psnr = calculate_psnr(
                bicubic,
                ground_truth
            )

            batch_model_psnr = calculate_psnr(
                prediction,
                ground_truth
            )

            # -----------------------------------------------------
            # Batch SSIM
            # -----------------------------------------------------

            batch_bicubic_ssim = calculate_ssim(
                bicubic,
                ground_truth
            )

            batch_model_ssim = calculate_ssim(
                prediction,
                ground_truth
            )

            # -----------------------------------------------------
            # LPIPS
            # -----------------------------------------------------

            bicubic_lpips_input = prepare_for_lpips(
                bicubic
            )

            prediction_lpips_input = prepare_for_lpips(
                prediction
            )

            gt_lpips_input = prepare_for_lpips(
                ground_truth
            )

            bicubic_lpips_value = lpips_model(
                bicubic_lpips_input,
                gt_lpips_input
            ).mean().item()

            model_lpips_value = lpips_model(
                prediction_lpips_input,
                gt_lpips_input
            ).mean().item()

            # -----------------------------------------------------
            # Accumulate
            # -----------------------------------------------------

            batch_size_actual = ground_truth.shape[0]

            total_bicubic_psnr += (
                batch_bicubic_psnr
                * batch_size_actual
            )

            total_model_psnr += (
                batch_model_psnr
                * batch_size_actual
            )

            total_bicubic_ssim += (
                batch_bicubic_ssim
                * batch_size_actual
            )

            total_model_ssim += (
                batch_model_ssim
                * batch_size_actual
            )

            total_bicubic_lpips += (
                bicubic_lpips_value
                * batch_size_actual
            )

            total_model_lpips += (
                model_lpips_value
                * batch_size_actual
            )

            sample_count += batch_size_actual

            # -----------------------------------------------------
            # Save selected current predictions
            # -----------------------------------------------------

            # Save first 10 validation samples.
            if saved_images < 10:

                for i in range(
                    batch_size_actual
                ):

                    if saved_images >= 10:
                        break

                    sample_number = (
                        sample_count
                        - batch_size_actual
                        + i
                    )

                    comparison_path = (
                        OUTPUT_DIR
                        / f"comparison_{sample_number:06d}.png"
                    )

                    model_path = (
                        OUTPUT_DIR
                        / f"model_{sample_number:06d}.png"
                    )

                    save_tensor_image(
                        prediction[i:i + 1],
                        model_path
                    )

                    create_comparison_image(
                        noisy[i:i + 1],
                        bicubic[i:i + 1],
                        prediction[i:i + 1],
                        ground_truth[i:i + 1],
                        comparison_path,
                        sample_number,
                        batch_bicubic_psnr,
                        batch_model_psnr,
                        batch_bicubic_ssim,
                        batch_model_ssim,
                        bicubic_lpips_value,
                        model_lpips_value
                    )

                    saved_images += 1

            # -----------------------------------------------------
            # Progress
            # -----------------------------------------------------

            if (
                (batch_index + 1) % 10 == 0
                or
                (batch_index + 1)
                == len(validation_loader)
            ):

                print(
                    f"Processed "
                    f"{batch_index + 1}/"
                    f"{len(validation_loader)} batches"
                )

    # =================================================================
    # FINAL RESULTS
    # =================================================================

    elapsed = time.time() - start_time

    if sample_count == 0:

        print()
        print(
            "ERROR: No samples were evaluated."
        )

        sys.exit(1)

    avg_bicubic_psnr = (
        total_bicubic_psnr
        / sample_count
    )

    avg_model_psnr = (
        total_model_psnr
        / sample_count
    )

    avg_bicubic_ssim = (
        total_bicubic_ssim
        / sample_count
    )

    avg_model_ssim = (
        total_model_ssim
        / sample_count
    )

    avg_bicubic_lpips = (
        total_bicubic_lpips
        / sample_count
    )

    avg_model_lpips = (
        total_model_lpips
        / sample_count
    )

    # -------------------------------------------------------------
    # Improvements
    # -------------------------------------------------------------

    psnr_improvement = (
        avg_model_psnr
        - avg_bicubic_psnr
    )

    ssim_improvement = (
        avg_model_ssim
        - avg_bicubic_ssim
    )

    lpips_improvement = (
        avg_bicubic_lpips
        - avg_model_lpips
    )

    if abs(avg_bicubic_lpips) > 1e-12:

        lpips_improvement_percent = (
            lpips_improvement
            / avg_bicubic_lpips
            * 100.0
        )

    else:

        lpips_improvement_percent = 0.0

    # =================================================================
    # PRINT RESULTS
    # =================================================================

    print_header("FINAL RESULTS")

    print(
        f"{'Metric':<15}"
        f"{'Bicubic':>18}"
        f"{'Our Model':>18}"
    )

    print("-" * 55)

    print(
        f"{'PSNR (dB)':<15}"
        f"{avg_bicubic_psnr:>18.4f}"
        f"{avg_model_psnr:>18.4f}"
    )

    print(
        f"{'SSIM':<15}"
        f"{avg_bicubic_ssim:>18.6f}"
        f"{avg_model_ssim:>18.6f}"
    )

    print(
        f"{'LPIPS':<15}"
        f"{avg_bicubic_lpips:>18.6f}"
        f"{avg_model_lpips:>18.6f}"
    )

    print("-" * 55)

    print()
    print(
        f"PSNR improvement : "
        f"+{psnr_improvement:.4f} dB"
    )

    print(
        f"SSIM improvement : "
        f"+{ssim_improvement:.6f}"
    )

    print(
        f"LPIPS reduction  : "
        f"{lpips_improvement:.6f}"
    )

    print(
        f"LPIPS reduction %: "
        f"{lpips_improvement_percent:.2f}%"
    )

    print()
    print(
        "Samples evaluated:",
        sample_count
    )

    print(
        f"Evaluation time: "
        f"{elapsed:.2f} seconds"
    )

    # =================================================================
    # SAVE JSON
    # =================================================================

    results = {

        "evaluation": {
            "checkpoint": str(CHECKPOINT_PATH),
            "samples": sample_count,
            "batch_size": BATCH_SIZE,
            "validation_ratio": VALIDATION_RATIO,
            "seed": SEED
        },

        "bicubic": {
            "psnr_db": avg_bicubic_psnr,
            "ssim": avg_bicubic_ssim,
            "lpips": avg_bicubic_lpips
        },

        "model": {
            "checkpoint": str(CHECKPOINT_PATH),
            "psnr_db": avg_model_psnr,
            "ssim": avg_model_ssim,
            "lpips": avg_model_lpips
        },

        "improvement": {
            "psnr_db": psnr_improvement,
            "ssim": ssim_improvement,
            "lpips_reduction": lpips_improvement,
            "lpips_reduction_percent":
                lpips_improvement_percent
        },

        "model_configuration": {
            "features": FEATURES,
            "residual_groups": RESIDUAL_GROUPS,
            "blocks_per_group": BLOCKS_PER_GROUP
        },

        "hardware": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            )
        }
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # =================================================================
    # FINAL MESSAGE
    # =================================================================

    print()
    print_header("EVALUATION COMPLETED")

    print("Results saved to:")
    print(RESULTS_PATH)

    print()
    print("Current model images saved to:")
    print(OUTPUT_DIR)

    print()
    print("Important:")
    print(
        "These images/results were generated from the CURRENT "
        "best_model.pth."
    )

    print(
        "The previous outputs/evaluation folder was not used."
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  PSNR  -> higher is better"
    )

    print(
        "  SSIM  -> higher is better"
    )

    print(
        "  LPIPS -> lower is better"
    )

    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()