# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and torchvision
- HuggingFace Transformers (for Vision Transformer)
- pandas, numpy, PIL (for data processing)
- scikit-learn (for metrics)
- matplotlib, seaborn (for visualization)
- TensorBoard (for logging)
- tqdm (for progress bars)

**Installation time:** ~5-10 minutes

## 2. Verify Dataset

Make sure the MURA dataset is in the correct location:

```bash
ls MURA-v1.1/
```

You should see:
- `train/` directory
- `valid/` directory
- `train_labeled_studies.csv`
- `valid_labeled_studies.csv`
- `train_image_paths.csv`
- `valid_image_paths.csv`

## 3. Train the Model

### Option A: Quick Test (Small Run)

Test that everything works with a short training run:

```bash
python experiments/train.py --epochs 2 --batch_size 4
```

This will:
- Train for 2 epochs
- Use batch size 4 (safe for most GPUs)
- Take ~10-20 minutes on GPU

### Option B: Full Training

For the full training run:

```bash
python experiments/train.py
```

This will:
- Train for 50 epochs (default)
- Use batch size 8
- Save checkpoints to `checkpoints/`
- Log to `logs/`
- Take ~6-8 hours on GPU

**Monitor training:**
```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 to see:
- Loss curves
- AUC-ROC progression
- Learning rate schedule
- Other metrics

## 4. Evaluate the Model

Once training is complete:

```bash
python experiments/evaluate.py --checkpoint checkpoints/best.pth
```

This will:
- Load the best model
- Evaluate on validation set
- Print metrics (AUC, accuracy, sensitivity, specificity)
- Save predictions to `results/predictions.csv`

**Expected output:**
```
AUC-ROC:      0.9300+
Accuracy:     0.8800+
Sensitivity:  0.8500+
Specificity:  0.9000+
```

## 5. Generate Visualizations

Create Grad-CAM visualizations to see what the model focuses on:

```bash
python experiments/visualize.py \
    --checkpoint checkpoints/best.pth \
    --num_samples 20
```

This will:
- Generate 20 Grad-CAM visualizations
- Save to `results/gradcam/`
- Show original X-rays and attention heatmaps

**Output:** 20 PNG images showing which regions the model focuses on for each prediction.

## 6. Resume Training (if interrupted)

If training is interrupted, resume from the last checkpoint:

```bash
python experiments/train.py --resume checkpoints/latest.pth
```

## Common Commands

### Training with Custom Settings

```bash
# Smaller batch size (if GPU memory issues)
python experiments/train.py --batch_size 4

# Different learning rate
python experiments/train.py --lr 5e-5

# Try concatenation fusion instead of cross-attention
python experiments/train.py --fusion_type concatenation

# CPU training (slow)
python experiments/train.py --device cpu
```

### Evaluation

```bash
# Evaluate with predictions saved
python experiments/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --save_predictions results/my_predictions.csv

# Evaluate specific epoch checkpoint
python experiments/evaluate.py --checkpoint checkpoints/epoch_030.pth
```

### Visualization

```bash
# More samples
python experiments/visualize.py \
    --checkpoint checkpoints/best.pth \
    --num_samples 50

# Custom output directory
python experiments/visualize.py \
    --checkpoint checkpoints/best.pth \
    --output_dir my_visualizations/
```

## Expected File Sizes

After training, you should see:
- `checkpoints/best.pth`: ~350 MB (best model)
- `checkpoints/latest.pth`: ~350 MB (latest model)
- `checkpoints/epoch_*.pth`: ~350 MB each (per epoch)
- `logs/`: ~50-100 MB (TensorBoard logs)

## Troubleshooting

### GPU Out of Memory
```bash
python experiments/train.py --batch_size 4  # or even 2
```

### Import Errors
```bash
# Make sure you're in the project root
cd AI_project
python experiments/train.py
```

### Dataset Not Found
Check that `DATA_ROOT` in `config/config.py` points to your MURA dataset:
```python
DATA_ROOT = "/path/to/your/MURA-v1.1"
```

### Slow Training
- Increase `NUM_WORKERS` in config.py (e.g., to 8)
- Ensure you're using GPU (`--device cuda`)
- Check that CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`

## Next Steps

1. **Analyze Results:** Look at the confusion matrix and metrics
2. **Visualize Predictions:** Check Grad-CAM to understand model decisions
3. **Hyperparameter Tuning:** Try different learning rates, fusion types
4. **Compare Approaches:** Train with different fusion strategies
5. **Error Analysis:** Examine false positives and false negatives

## File Overview

**Configuration:**
- `config/config.py` - All hyperparameters and settings

**Data:**
- `data/dataset.py` - Multi-view data loader
- `data/transforms.py` - Image augmentations

**Models:**
- `models/mura_classifier.py` - Complete model
- `models/multiview_fusion.py` - Fusion mechanisms
- `models/vit_backbone.py` - ViT feature extractor

**Training:**
- `training/trainer.py` - Training loop
- `training/metrics.py` - Evaluation metrics

**Experiments:**
- `experiments/train.py` - Training script
- `experiments/evaluate.py` - Evaluation script
- `experiments/visualize.py` - Visualization script

## Tips for Best Results

1. **Start Small:** Test with 2 epochs first to verify everything works
2. **Monitor Training:** Use TensorBoard to watch metrics in real-time
3. **Check Checkpoints:** Best model is saved based on validation AUC
4. **Use GPU:** Training on CPU will take days instead of hours
5. **Experiment:** Try different fusion types and hyperparameters

Good luck with your training! 🚀
