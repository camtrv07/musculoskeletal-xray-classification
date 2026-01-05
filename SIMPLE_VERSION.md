# Simplified MURA Training (CPU-Friendly)

**Fast training on CPU - completes in 30-60 minutes!**

## What's Different?

This simplified version is designed for quick experiments on CPU:

| Component | Full Version | Simple Version |
|-----------|--------------|----------------|
| **Model** | Vision Transformer (86M params) | ResNet18 (11M params) |
| **Fusion** | Cross-attention | Average pooling |
| **Training Data** | 13,457 studies | 1,000 studies (7%) |
| **Validation Data** | 1,199 studies | 200 studies (17%) |
| **Training Time** | 9 hours/epoch (CPU) | 3-6 min/epoch (CPU) |
| **Total Time (10 epochs)** | 90 hours | 30-60 minutes |

**Speed improvement: 90-180× faster!**

## Quick Start

### 1. Run Simple Training

```bash
python experiments/train_simple.py
```

That's it! Training will complete in 30-60 minutes.

### 2. Custom Settings

```bash
# More epochs (better results)
python experiments/train_simple.py --epochs 20

# Larger subset (better accuracy, slower)
python experiments/train_simple.py --train_samples 2000 --val_samples 400

# Smaller batch for very limited RAM
python experiments/train_simple.py --batch_size 2

# Different learning rate
python experiments/train_simple.py --lr 5e-4
```

## What You Get

After training:
- **Model checkpoint:** `checkpoints/simple_best.pth`
- **Expected AUC:** 0.75-0.82 (on subset)
- **Training time:** 30-60 minutes

## Architecture

### Simple Model

```
Multi-view X-rays [batch, 4, 3, 224, 224]
    ↓
ResNet18 Backbone (shared across views)
    ↓
View Features [batch, 4, 512]
    ↓
Average Pooling Fusion
    ↓
Fused Features [batch, 512]
    ↓
Classification Head
    ↓
Prediction [batch, 1]
```

### Key Simplifications

1. **ResNet18 instead of ViT**
   - 11M params vs 86M params
   - Pre-trained on ImageNet
   - 10× faster on CPU

2. **Average Pooling instead of Cross-Attention**
   - Simple mean across views
   - No learnable parameters
   - Instant computation

3. **Small Dataset Subset**
   - 1,000 training samples (stratified)
   - Maintains class balance
   - Representative of full dataset

## Expected Results

### Performance (on 1000-sample subset)

| Metric | Simple Model | Full Model (reference) |
|--------|--------------|------------------------|
| AUC-ROC | 0.75-0.82 | 0.93+ |
| Accuracy | 0.72-0.78 | 0.88+ |
| Sensitivity | 0.70-0.75 | 0.85+ |
| Specificity | 0.75-0.80 | 0.90+ |

**Note:** Lower performance is expected because:
- Training on only 7% of data
- Simpler model architecture
- No hyperparameter tuning

### Time Comparison

| Setup | Time (10 epochs) |
|-------|------------------|
| Full model + full data (CPU) | 90 hours |
| Full model + subset (CPU) | 8-10 hours |
| **Simple model + subset (CPU)** | **30-60 min** ⚡ |
| Simple model + subset (GPU) | 5-10 min |

## Options & Parameters

### Command Line Arguments

```bash
python experiments/train_simple.py \
    --epochs 10 \              # Number of training epochs
    --batch_size 4 \           # Batch size (reduce if out of memory)
    --lr 1e-3 \                # Learning rate
    --train_samples 1000 \     # Number of training samples
    --val_samples 200          # Number of validation samples
```

### Recommended Configurations

**Quick Test (10 minutes):**
```bash
python experiments/train_simple.py --epochs 3 --train_samples 500
```

**Standard (30 minutes):**
```bash
python experiments/train_simple.py --epochs 10 --train_samples 1000
```

**Better Accuracy (2 hours):**
```bash
python experiments/train_simple.py --epochs 20 --train_samples 3000 --val_samples 500
```

**Best Results (4-6 hours):**
```bash
python experiments/train_simple.py --epochs 30 --train_samples 5000 --val_samples 1000
```

## Memory Requirements

| Configuration | RAM Needed |
|---------------|------------|
| batch_size=2 | ~4 GB |
| batch_size=4 | ~6 GB |
| batch_size=8 | ~10 GB |

If you get out of memory errors, reduce `--batch_size`.

## Evaluation

After training, evaluate your model:

```python
import torch
from models.simple_classifier import create_simple_model
from data.simple_dataset import get_simple_dataloaders
from config.config import Config

# Load model
config = Config()
model = create_simple_model(device='cpu')
checkpoint = torch.load('checkpoints/simple_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Load data
_, val_loader = get_simple_dataloaders(config, val_limit=200)

# Evaluate
from training.metrics import MetricsCalculator
metrics_calc = MetricsCalculator()

model.eval()
with torch.no_grad():
    for batch in val_loader:
        images = batch['images']
        mask = batch['mask']
        labels = batch['label']

        logits, _ = model(images, mask)
        metrics_calc.update(logits, labels)

metrics = metrics_calc.compute()
print(f"AUC: {metrics['auc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## When to Use Each Version

### Use Simple Version When:
- ✅ Testing code quickly
- ✅ Only have CPU available
- ✅ Learning/experimenting
- ✅ Limited time (<1 hour)
- ✅ Want to see results fast

### Use Full Version When:
- ✅ Have GPU available
- ✅ Need best performance
- ✅ Publishing results
- ✅ Clinical application
- ✅ Full dataset needed

## Next Steps

After training the simple version:

1. **Verify it works:** Check that training completes and AUC > 0.70
2. **Try different settings:** Experiment with epochs, samples, learning rate
3. **Understand the code:** Simple version is easier to learn from
4. **Scale up:** Move to full version on GPU when ready

## Troubleshooting

### "Out of memory"
```bash
python experiments/train_simple.py --batch_size 2
```

### "Too slow"
```bash
# Use fewer samples
python experiments/train_simple.py --train_samples 500 --epochs 5
```

### "Accuracy too low"
```bash
# Use more data and train longer
python experiments/train_simple.py --train_samples 3000 --epochs 20
```

### "Want faster training"
- Use GPU (10-20× faster)
- Or use even fewer samples: `--train_samples 500`

## Files Created

- **models/simple_classifier.py** - Lightweight model (ResNet18)
- **data/simple_dataset.py** - Subset dataloader
- **experiments/train_simple.py** - Fast training script

## Comparison to Original

The simple version trades accuracy for speed:

| Aspect | Simple | Full |
|--------|--------|------|
| Complexity | ⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ | ⭐ |
| Accuracy | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CPU-friendly | ⭐⭐⭐⭐⭐ | ⭐ |
| Memory | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**Perfect for quick experiments and learning!**

---

Ready to train? Just run:

```bash
python experiments/train_simple.py
```

And grab a coffee! ☕ (30-60 minutes)
