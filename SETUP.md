# Setup Guide

This guide helps you set up the MURA project on a new computer.

## Prerequisites

- Python 3.8+ (Python 3.13 recommended)
- pip or conda
- Git

## Quick Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd AI_project
```

### 2. Download MURA Dataset

Download the MURA dataset and place it in the project root:

```bash
# Your project structure should look like:
AI_project/
├── MURA-v1.1/
│   ├── train/
│   ├── valid/
│   ├── train_labeled_studies.csv
│   └── valid_labeled_studies.csv
├── config/
├── data/
├── experiments/
└── ...
```

**Important:** The `MURA-v1.1` folder must be directly inside the `AI_project` folder.

Download MURA from: https://stanfordmlgroup.github.io/competitions/mura/

### 3. Install Dependencies

**Option A: Using pip**
```bash
pip install -r requirements.txt
```

**Option B: Using conda**
```bash
conda create -n mura python=3.13
conda activate mura
pip install -r requirements.txt
```

### 4. Verify Setup

Run a quick test to verify everything is configured correctly:

```bash
python experiments/train_with_logging.py --name "test" --epochs 1 --train_samples 100
```

This should:
- Load data successfully
- Train for 1 epoch on 100 samples
- Complete in ~2-3 minutes on CPU

If it works, you're all set!

## Common Issues

### Issue 1: "FileNotFoundError: train_labeled_studies.csv"

**Problem:** MURA dataset is not in the correct location.

**Solution:** Ensure the folder structure is:
```
AI_project/
└── MURA-v1.1/
    ├── train_labeled_studies.csv
    └── valid_labeled_studies.csv
```

The `MURA-v1.1` folder should be **directly inside** your project root (same level as `config/`, `data/`, etc.).

### Issue 2: "No such file or directory: /mnt/c/Users/..."

**Problem:** This means the code still has hardcoded paths from the original computer.

**Solution:** Make sure you have the latest version with relative paths:
```bash
git pull origin main
```

The `config/config.py` file should have:
```python
DATA_ROOT = os.path.join(PROJECT_ROOT, "MURA-v1.1")
```

NOT:
```python
DATA_ROOT = "/mnt/c/Users/camil/..."  # ❌ Wrong
```

### Issue 3: Module Import Errors

**Problem:** Python can't find project modules.

**Solution:** Make sure you're running commands from the project root:
```bash
cd /path/to/AI_project  # Navigate to project root first
python experiments/train_with_logging.py --name "test"
```

### Issue 4: Out of Memory

**Problem:** Training crashes or computer freezes.

**Solution:** Reduce batch size and training samples:
```bash
python experiments/train_with_logging.py --name "test" --batch_size 2 --train_samples 500
```

## Configuration

The project uses relative paths by default, so it should work on any computer once you:
1. Clone the repository
2. Place the MURA dataset in the correct location
3. Install dependencies

All paths in `config/config.py` are now relative to the project root, so no manual path editing is needed.

## Recommended First Steps

After setup, try these in order:

### 1. Quick Test (3 minutes)
```bash
python experiments/train_with_logging.py --name "quick_test" --epochs 1 --train_samples 100
```

### 2. Short Training (30 minutes)
```bash
python experiments/train_with_logging.py --name "baseline" --epochs 10 --train_samples 1000
```

### 3. Compare Experiments
```bash
# Run a few experiments
python experiments/train_with_logging.py --name "exp1" --epochs 10
python experiments/train_with_logging.py --name "exp2" --epochs 20
python experiments/train_with_logging.py --name "exp3" --lr 5e-4 --epochs 10

# Compare results
jupyter notebook notebooks/Compare_Experiments.ipynb
```

## Additional Documentation

- [README.md](README.md) - Full project overview
- [QUICKSTART.md](QUICKSTART.md) - 3-step quick start
- [SIMPLE_VERSION.md](SIMPLE_VERSION.md) - CPU training guide
- [EXPERIMENT_COMPARISON_GUIDE.md](EXPERIMENT_COMPARISON_GUIDE.md) - Compare different trainings
- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Visualize training results
- [COLAB_GUIDE.md](COLAB_GUIDE.md) - Use Google Colab for GPU training

## Getting Help

If you encounter issues:

1. Check this setup guide for common problems
2. Verify your folder structure matches the examples
3. Ensure all dependencies are installed
4. Try running with smaller parameters (fewer samples, smaller batch size)

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

## Next Steps

Once setup is complete, see [EXPERIMENT_COMPARISON_GUIDE.md](EXPERIMENT_COMPARISON_GUIDE.md) to learn how to run and compare experiments systematically.
