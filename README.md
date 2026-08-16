# Semicon Image Restoration

> A deep learning-based image restoration and 2× super-resolution system for reconstructing high-quality images from noisy, low-resolution inputs.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Acceleration-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)](#license)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Specifications](#key-specifications)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Dataset Inspection](#dataset-inspection)
- [Dataset Analysis](#dataset-analysis)
- [DataLoader Verification](#dataloader-verification)
- [Bicubic Baseline](#bicubic-baseline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Evaluation Metrics](#evaluation-metrics)
- [Current Experimental Results](#current-experimental-results)
- [Trained Model Weights](#trained-model-weights)
- [Single Image Inference](#single-image-inference)
- [Output](#output)
- [Reproducibility](#reproducibility)
- [Technology Stack](#technology-stack)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Project Workflow](#project-workflow)
- [Academic / Research Use](#academic--research-use)
- [License](#license)
- [Author](#author)

---

## Overview

**Semicon Image Restoration** is a supervised deep learning project designed to restore degraded low-resolution images and reconstruct higher-resolution outputs.

The system takes a noisy **128 × 128 grayscale image** as input and produces a restored **256 × 256 grayscale image** using a custom residual super-resolution architecture implemented in PyTorch.

The project includes the complete machine learning pipeline:

- Dataset inspection
- Dataset analysis
- Data loading and preprocessing
- Bicubic interpolation baseline
- Deep learning model
- Training pipeline
- Validation
- Model checkpointing
- PSNR evaluation
- SSIM evaluation
- LPIPS perceptual evaluation
- Standalone inference
- Restored test outputs

---

## Problem Statement

Image acquisition and processing pipelines can produce images affected by:

- Noise
- Resolution degradation
- Loss of fine details
- Structural artifacts

The objective of this project is to learn a mapping from a degraded low-resolution image to a restored high-resolution image.

```text
Noisy Low-Resolution Image
            │
            ▼
     Deep Learning Model
            │
            ▼
Restored High-Resolution Image
```

---

## Key Specifications

| Component | Specification |
|---|---|
| Input | 128 × 128 grayscale |
| Output | 256 × 256 grayscale |
| Scale Factor | 2× |
| Input Format | `.npy` |
| Ground Truth Format | `.npy` |
| Output Format | `.png` |
| Framework | PyTorch |
| Training | Supervised Learning |
| Architecture | Residual Super-Resolution Network |
| GPU Acceleration | CUDA |
| Evaluation | PSNR, SSIM, LPIPS |

---

## Dataset

The training dataset consists of paired **NoisyLR → Ground Truth** images.

### Directory Structure

```text
data/
└── extracted/
    └── train/
        └── train/
            ├── NoisyLR/
            │   ├── 000000.npy
            │   ├── 000001.npy
            │   └── ...
            │
            └── GT/
                ├── 000000.npy
                ├── 000001.npy
                └── ...
```

### Dataset Statistics

| Split | Count |
|---|---|
| Total image pairs | 3,200 |
| Training samples | 2,880 |
| Validation samples | 320 |

### Data Representation

The images are stored as NumPy arrays.

- **Typical input:** `128 × 128 × 1`
- **Ground truth:** `256 × 256 × 1`

Corresponding input and ground-truth files use matching identifiers.

**Example:**

```text
NoisyLR/000040.npy
GT/000040.npy
```

---

## Model Architecture

The proposed model is **SemiconRestorationNet**, which combines residual feature extraction with 2× PixelShuffle-based upsampling.

```text
                 Input
              128 × 128 × 1
                    │
                    ▼
             Head Convolution
                    │
                    ▼
             Residual Groups
                    │
                    ▼
             Body Convolution
                    │
                    ▼
          Global Residual Connection
                    │
                    ▼
             Upsampling Block
                    │
                    ▼
             PixelShuffle ×2
                    │
                    ▼
             Output Convolution
                    │
                    ▼
              256 × 256 × 1
```

### Main Components

**1. Feature Extraction**
The input image is initially transformed into a higher-dimensional feature representation using convolution.

**2. Residual Groups**
Multiple residual groups are used to learn image features while maintaining stable gradient propagation. Each residual group contains multiple residual blocks.

**3. Residual Learning**
The network uses skip connections to preserve useful low-level information while learning the required restoration features.

**4. Upsampling**
The final feature representation is converted to a higher spatial resolution using:

```text
Convolution
     │
     ▼
PixelShuffle ×2
     │
     ▼
Convolution
```

This reconstructs the final 256 × 256 image.

---

## Project Structure

```text
Semicon-Image-Restoration/
│
├── README.md
├── requirements.txt
├── .gitignore
├── demo.py
│
├── data/
│   └── extracted/
│       └── train/
│           └── train/
│               ├── NoisyLR/
│               └── GT/
│
├── checkpoints/
│   └── best_model.pth
│
├── outputs/
│   ├── baseline/
│   ├── evaluation/
│   ├── inspection/
│   └── training/
│
└── src/
    │
    ├── 01_inspect/
    │   ├── 01_inspect_dataset.py
    │   └── 02_visualize_samples.py
    │
    ├── 02_analyse/
    │   └── analyze_dataset.py
    │
    ├── 03_dataset/
    │   ├── dataset.py
    │   └── test_dataloader.py
    │
    ├── 04_baseline/
    │   └── bicubic_baseline.py
    │
    ├── 05_model/
    │   └── model.py
    │
    ├── 06_training/
    │   └── train.py
    │
    └── 07_evaluation/
        └── evaluate.py
```

---

## Installation

### Requirements

Recommended environment:

- Python 3.10+
- PyTorch
- CUDA-capable NVIDIA GPU for accelerated training
- CUDA-compatible PyTorch installation

The complete dependency list is provided in `requirements.txt`.

### 1. Clone the Repository

```bash
git clone https://github.com/VishwaSabaris/Semicon-Image-Restoration.git
cd Semicon-Image-Restoration
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Place the dataset in the following structure:

```text
data/
└── extracted/
    └── train/
        └── train/
            ├── NoisyLR/
            └── GT/
```

Ensure that corresponding files use matching identifiers.

**Example:**

```text
NoisyLR/000040.npy
GT/000040.npy
```

---

## Dataset Inspection

Before training, inspect the dataset:

```bash
python src/01_inspect/01_inspect_dataset.py
```

To visualize representative samples:

```bash
python src/01_inspect/02_visualize_samples.py
```

Generated inspection results are stored under `outputs/inspection/`.

---

## Dataset Analysis

Run the dataset analysis script:

```bash
python src/02_analyse/analyze_dataset.py
```

This stage is used to examine the dataset characteristics and image statistics prior to model training.

---

## DataLoader Verification

Verify that the paired dataset can be loaded correctly:

```bash
python src/03_dataset/test_dataloader.py
```

This confirms:

- Dataset discovery
- Input/target pairing
- Tensor conversion
- Image dimensions
- Batch loading

---

## Bicubic Baseline

A conventional bicubic interpolation baseline is included for quantitative comparison.

```bash
python src/04_baseline/bicubic_baseline.py
```

The baseline provides a non-deep-learning reference for evaluating the improvement achieved by the proposed model.

---

## Training

The complete training implementation is available at `src/06_training/train.py`.

```bash
python src/06_training/train.py
```

The training pipeline performs:

```text
Dataset Loading
      │
      ▼
Train / Validation Split
      │
      ▼
DataLoader
      │
      ▼
Model Forward Pass
      │
      ▼
Loss Calculation
      │
      ▼
Backpropagation
      │
      ▼
Optimizer Update
      │
      ▼
Validation
      │
      ▼
Metric Calculation
      │
      ▼
Checkpointing
```

The best-performing checkpoint is saved under `checkpoints/`.

---

## Evaluation

The standalone evaluation script is `src/07_evaluation/evaluate.py`. It loads the trained model, processes the test images, and writes restored outputs to the specified output directory.

### Usage

```bash
python src/07_evaluation/evaluate.py <input_directory> <output_directory>
```

### Example

**Windows PowerShell:**

```powershell
python src\07_evaluation\evaluate.py `
    "C:\path\to\Test_NoisyLR\NoisyLR" `
    "C:\path\to\restored_outputs"
```

**Linux:**

```bash
python src/07_evaluation/evaluate.py \
    "/path/to/Test_NoisyLR/NoisyLR" \
    "/path/to/restored_outputs"
```

The evaluation script is designed to run without manual source-code modifications.

---

## Evaluation Metrics

The model is evaluated using multiple complementary metrics.

| Metric | Description | Interpretation |
|---|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio — measures pixel-level reconstruction fidelity | Higher = Better |
| **SSIM** | Structural Similarity Index — evaluates structural similarity between restored and reference images | Higher = Better |
| **LPIPS** | Learned Perceptual Image Patch Similarity — measures perceptual similarity using learned deep feature representations | Lower = Better |

Using PSNR, SSIM, and LPIPS together provides a more comprehensive assessment of restoration quality.

---

## Current Experimental Results

The current evaluated model achieved:

| Metric | Value |
|---|---|
| Validation PSNR | 28.8457 dB |
| Validation SSIM | 0.789075 |
| Samples | 320 |

The corresponding training run reported:

| Metric | Value |
|---|---|
| Best Validation PSNR | 27.1817 dB |

The evaluation script independently processes the evaluation samples and calculates the final metrics. Results may vary depending on the exact checkpoint, dataset split, preprocessing configuration, and evaluation configuration.

---

## Trained Model Weights

The trained checkpoint is located at `checkpoints/best_model.pth`.

The checkpoint must be loaded using the architecture implemented in `src/05_model/model.py`.

> **Note:** If the model weight file exceeds standard GitHub repository storage limits, it should be distributed using Git LFS, Google Drive, Hugging Face, or another appropriate model-storage service.

---

## Single Image Inference

A demonstration script is provided: `demo.py`. It accepts a single `.npy` image and generates a restored image.

### Example

```bash
python demo.py "C:\path\to\000040.npy"
```

### Processing Pipeline

```text
.npy Input
    │
    ▼
NumPy Array
    │
    ▼
Tensor Conversion
    │
    ▼
SemiconRestorationNet
    │
    ▼
Model Inference
    │
    ▼
Restored Tensor
    │
    ▼
Image Conversion
    │
    ▼
PNG Output
```

---

## Output

The model produces restored high-resolution images.

```text
Input
128 × 128
    │
    ▼
SemiconRestorationNet
    │
    ▼
Output
256 × 256
```

**Conceptual comparison:**

| Noisy / LR Input | Restored Output |
|---|---|
| 128 × 128 | 256 × 256 |

Generated evaluation outputs are stored under `outputs/evaluation/`.

---

## Reproducibility

A reviewer can reproduce the project using the following workflow.

**Step 1 — Clone**

```bash
git clone https://github.com/VishwaSabaris/Semicon-Image-Restoration.git
cd Semicon-Image-Restoration
```

**Step 2 — Create Environment**

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate`
Linux: `source .venv/bin/activate`

**Step 3 — Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Prepare Dataset**

```text
data/
└── extracted/
    └── train/
        └── train/
            ├── NoisyLR/
            └── GT/
```

**Step 5 — Train**

```bash
python src/06_training/train.py
```

**Step 6 — Evaluate**

```bash
python src/07_evaluation/evaluate.py <input_directory> <output_directory>
```

### Reproducibility Checklist

Before submitting or evaluating the repository, verify that it contains:

- [x] README.md
- [x] requirements.txt
- [x] Model architecture
- [x] Training script
- [x] Standalone evaluation script
- [x] Trained model checkpoint
- [x] Restored test outputs
- [x] Inference demonstration
- [x] Dataset directory documentation
- [x] Evaluation metrics
- [x] Installation instructions

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| CUDA | GPU acceleration |
| NumPy | Numerical data processing |
| Pillow | Image processing |
| OpenCV | Image processing |
| scikit-image | Image quality metrics |
| LPIPS | Perceptual evaluation |
| Git | Version control |
| GitHub | Source code hosting |

---

## Limitations

The current implementation is trained and evaluated on the provided paired dataset. Performance may change when:

- The input distribution differs significantly from the training dataset.
- The noise characteristics differ from the training data.
- The input resolution differs from the expected resolution.
- The image dynamic range differs from the training preprocessing pipeline.

Therefore, the model should be evaluated on representative samples from the target deployment environment before production use.

---

## Future Improvements

Potential improvements include:

- More advanced residual architectures
- Attention mechanisms
- Multi-scale feature extraction
- Improved perceptual loss functions
- Larger and more diverse training datasets
- Data augmentation
- Advanced learning-rate scheduling
- Mixed-precision training
- Model compression and quantization
- ONNX / TensorRT deployment
- Real-time inference optimization

---

## Project Workflow

```text
        Dataset
           │
           ▼
   Dataset Inspection
           │
           ▼
    Dataset Analysis
           │
           ▼
    Data Preparation
           │
           ▼
   Bicubic Baseline
           │
           ▼
    Model Training
           │
           ▼
   Best Checkpoint
           │
           ▼
      Evaluation
           │
           ▼
 PSNR / SSIM / LPIPS
           │
           ▼
   Restored Outputs
```

---

## Academic / Research Use

This repository is intended for academic, research, and demonstration purposes. The implementation provides an end-to-end reproducible workflow for studying image restoration and super-resolution using deep learning.

---

## License

This project is intended for academic and research purposes.

Please contact the repository author before using the implementation or trained model for commercial applications.

---

## Author

**Vishwa Sabaris V**

- GitHub: [https://github.com/VishwaSabaris](https://github.com/VishwaSabaris)
- Repository: [https://github.com/VishwaSabaris/Semicon-Image-Restoration](https://github.com/VishwaSabaris/Semicon-Image-Restoration)
