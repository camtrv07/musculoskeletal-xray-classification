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

### Simple Classifier (Current - CPU Friendly)
- **Backbone:** ResNet18 CNN (11M parameters)
- **Fusion:** Average pooling across views
- **Multi-view:** Yes (processes multiple X-rays per study)
- **Transformer-based:** No
- **Speed:** Fast on CPU
- **Use case:** Quick experiments and baseline

### Full Classifier (Advanced - GPU Recommended)
- **Backbone:** Vision Transformer (ViT, 86M parameters)
- **Fusion:** Cross-attention between views
- **Multi-view:** Yes (processes multiple X-rays per study)
- **Transformer-based:** Yes
- **Speed:** Slow on CPU, requires GPU
- **Use case:** Best performance, research-grade

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

## What Gets Logged

Every training run automatically saves:

1. **Master log:** `experiments_log/experiments.json` - All experiments in one file
2. **Individual logs:** `experiments_log/[name]_[timestamp].json` - Detailed results per experiment
3. **Checkpoints:** `checkpoints/[name]_best.pth` - Best model for each experiment

Example experiment log:
```json
{
  "id": "baseline_20260105_143022",
  "name": "baseline",
  "timestamp": "2026-01-05T14:30:22",
  "config": {
    "epochs": 10,
    "batch_size": 4,
    "lr": 0.001,
    "train_samples": 1000
  },
  "best_metrics": {
    "auc": 0.7823,
    "accuracy": 0.7456
  }
}
```

## Example Workflows

### Find Best Learning Rate
```bash
python experiments/train.py --name "lr_1e-2" --lr 1e-2
python experiments/train.py --name "lr_1e-3" --lr 1e-3
python experiments/train.py --name "lr_1e-4" --lr 1e-4
python experiments/train.py --name "lr_5e-4" --lr 5e-4
```

### Test Effect of More Data
```bash
python experiments/train.py --name "data_500" --train_samples 500
python experiments/train.py --name "data_1000" --train_samples 1000
python experiments/train.py --name "data_2000" --train_samples 2000
python experiments/train.py --name "data_3000" --train_samples 3000
```

### Determine Optimal Epochs
```bash
python experiments/train.py --name "epochs_5" --epochs 5
python experiments/train.py --name "epochs_10" --epochs 10
python experiments/train.py --name "epochs_20" --epochs 20
python experiments/train.py --name "epochs_30" --epochs 30
```

Then compare all results in the Jupyter notebook.

## Common Issues & Solutions

### Issue: "FileNotFoundError: train_labeled_studies.csv"
**Solution:** Ensure MURA dataset is in the correct location:
```
AI_project/MURA-v1.1/train_labeled_studies.csv
```

### Issue: Out of Memory
**Solution:** Reduce batch size and training samples:
```bash
python experiments/train.py --name "test" --batch_size 2 --train_samples 500
```

### Issue: Module Import Errors
**Solution:** Run commands from project root:
```bash
cd /path/to/AI_project
python experiments/train.py --name "test"
```

## System Requirements

### Minimum (CPU only)
- 4 GB RAM
- 2 GB free disk space
- Python 3.8+

### Recommended (CPU)
- 8 GB RAM
- 5 GB free disk space
- Python 3.10+

### Optimal (GPU)
- 16 GB RAM
- GPU with 8+ GB VRAM
- 10 GB free disk space
- CUDA-compatible GPU

## Performance Expectations

**Simple Classifier (ResNet18):**
- Training speed: ~3-5 min/epoch on CPU (1000 samples)
- Expected AUC: 0.75-0.85
- Use case: Quick experiments, baseline comparisons

**Full Classifier (ViT):**
- Training speed: Requires GPU
- Expected AUC: 0.93-0.95
- Use case: Final model, publication-ready results

## Interpreting Results

### Good Signs
- Increasing validation AUC - Model is learning
- Small train/val gap - Not overfitting
- Stable curves - Reliable training
- Balanced sensitivity/specificity - No bias

### Warning Signs
- Decreasing validation AUC - Overfitting or bad hyperparameters
- Large train/val gap - Model memorizing training data
- Oscillating curves - Learning rate too high
- Very low sensitivity OR specificity - Imbalanced predictions

## Citation

If you use this code for your research, please cite:

**MURA Dataset:**
```
@inproceedings{rajpurkar2017mura,
  title={MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs},
  author={Rajpurkar, Pranav and Irvin, Jeremy and Bagul, Aarti and others},
  booktitle={Medical Imaging with Deep Learning},
  year={2017}
}
```

**Vision Transformer:**
```
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

## License

This project is for educational and research purposes.

## Acknowledgments

- Stanford ML Group for the MURA dataset
- Google Research for the Vision Transformer
- HuggingFace for the Transformers library

---

**Note:** This is a research project. The model should not be used for clinical decisions without proper validation and regulatory approval.
