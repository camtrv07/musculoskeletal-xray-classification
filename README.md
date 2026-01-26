# MURA Musculoskeletal Abnormality Detection

A deep learning system for detecting abnormalities in musculoskeletal radiographs using multi-view fusion.

## Overview

This project implements a multi-view classification model for musculoskeletal abnormality detection on the MURA (Musculoskeletal Radiographs) dataset. The model processes multiple X-ray images per study and fuses them to make a final diagnosis prediction.

**Key Features:**
- Multi-view fusion for handling multiple X-ray images per study
- Supports both simple CNN (ResNet18) and transformer-based (ViT) architectures
- Experiment tracking and comparison system
- Comprehensive evaluation metrics (AUC-ROC, sensitivity, specificity)
- Visualization tools for comparing different training runs

## Dataset

**MURA Dataset** (Stanford ML Group)
- **Training:** 36,808 images across 13,457 studies
- **Validation:** 3,197 images across 1,199 studies
- **Body Parts:** Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist
- **Task:** Binary classification (normal vs abnormal)
- **Class Distribution:** 38.5% abnormal, 61.5% normal

Download from: https://stanfordmlgroup.github.io/competitions/mura/

## Quick Setup

### Prerequisites
- Python 3.8+ (Python 3.13 recommended)
- pip or conda
- Git

### Installation Steps

1. **Clone the repository:**
```bash
git clone <your-repository-url>
cd AI_project
```

2. **Download MURA dataset:**

   Place the MURA dataset in the project root:
   ```
   AI_project/
   ├── MURA-v1.1/
   │   ├── train/
   │   ├── valid/
   │   ├── train_labeled_studies.csv
   │   └── valid_labeled_studies.csv
   ├── config/
   ├── data/
   └── ...
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or using conda:
   ```bash
   conda create -n mura python=3.13
   conda activate mura
   pip install -r requirements.txt
   ```

4. **Verify setup:**
   ```bash
   python experiments/train.py --name "test" --epochs 1 --train_samples 100
   ```

   This should complete in ~2-3 minutes on CPU.

## Project Structure

```
AI_project/
├── config/
│   └── config.py              # Central configuration
├── data/
│   ├── dataset.py             # Full MURA dataset loader
│   ├── simple_dataset.py      # Simplified dataset for quick training
│   ├── transforms.py          # Image augmentations
│   └── utils.py               # Data utilities
├── models/
│   ├── simple_classifier.py   # ResNet18-based model (fast, CPU-friendly)
│   ├── vit_backbone.py        # Vision Transformer backbone
│   ├── multiview_fusion.py    # Cross-attention fusion modules
│   └── mura_classifier.py     # Complete transformer-based model
├── training/
│   ├── losses.py              # Weighted BCE loss
│   ├── metrics.py             # Evaluation metrics
│   └── checkpoint_manager.py  # Model checkpointing
├── experiments/
│   └── train.py               # Main training script with experiment tracking
├── experiments_log/           # Saved experiment results
├── checkpoints/               # Saved model checkpoints
├── notebooks/
│   └── Compare_Experiments.ipynb  # Experiment comparison notebook
└── utils/
    └── seed.py                # Reproducibility utilities
```

## Model Architectures

### Classifier
- **Backbone:** ResNet18 CNN (11M parameters)
- **Fusion:** Average pooling across views
- **Multi-view:** Yes (processes multiple X-rays per study)
- **Transformer-based:** No
- **Speed:** Fast on CPU
- **Use case:** Quick experiments and baseline

## Usage

### Training with Experiment Tracking

The main training script automatically logs all experiments for easy comparison:

**Basic training:**
```bash
python experiments/train.py --name "baseline"
```

**Custom parameters:**
```bash
python experiments/train.py \
  --name "my_experiment" \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-3 \
  --train_samples 2000 \
  --val_samples 400
```

**Available options:**
- `--name`: Experiment name (required)
- `--epochs`: Number of epochs (default: 10)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-3)
- `--train_samples`: Training samples (default: 1000)
- `--val_samples`: Validation samples (default: 200)
- `--optimizer`: Optimizer choice (default: adam, options: adam, sgd)

### Experiment Comparison

Run multiple experiments with different settings:

```bash
# Experiment 1: Baseline
python experiments/train.py --name "baseline" --epochs 10 --lr 1e-3

# Experiment 2: More epochs
python experiments/train.py --name "more_epochs" --epochs 20 --lr 1e-3

# Experiment 3: Higher learning rate
python experiments/train.py --name "higher_lr" --epochs 10 --lr 5e-3

# Experiment 4: More training data
python experiments/train.py --name "more_data" --epochs 10 --train_samples 3000
```

Each experiment is automatically saved to `experiments_log/` with:
- Unique ID and timestamp
- All hyperparameters
- Complete training history
- Best validation metrics

### Comparing Results

Open the comparison notebook:
```bash
jupyter notebook notebooks/Compare_Experiments.ipynb
```

The notebook will show:
- Side-by-side training curves
- Final metrics comparison
- Hyperparameter effect analysis
- Best experiment recommendation