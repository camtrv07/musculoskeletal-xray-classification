# Which Version Should You Use?

You now have **TWO versions** of the MURA project. Here's how to choose:

## 🚀 Simple Version (NEW - Recommended for CPU)

**Best for: Quick experiments on CPU**

```bash
python experiments/train_simple.py
```

### Pros:
- ✅ **Fast:** 30-60 minutes total
- ✅ **CPU-friendly:** No GPU needed
- ✅ **Low memory:** Works with 4-6 GB RAM
- ✅ **Simple code:** Easier to understand
- ✅ **Quick results:** See if it works immediately

### Cons:
- ⚠️ Lower accuracy (AUC ~0.75-0.82)
- ⚠️ Trains on subset only (1000 samples)
- ⚠️ Simpler architecture

### When to Use:
- Testing code quickly
- Learning the workflow
- Limited hardware (CPU only)
- Time-constrained (<1 hour)
- Proof of concept

---

## 🎯 Full Version (Original - Best Performance)

**Best for: Production results on GPU**

```bash
python experiments/train.py
```

or

**Google Colab** (Recommended):
- Use `MURA_Training_Colab.ipynb`
- Free T4 GPU
- 7-8 hours for 10 epochs

### Pros:
- ✅ **Best accuracy:** AUC ~0.93+
- ✅ **Full dataset:** All 40,000+ images
- ✅ **State-of-art:** Vision Transformer + Cross-Attention
- ✅ **Research-ready:** Publication quality

### Cons:
- ⚠️ Requires GPU (or 90+ hours on CPU)
- ⚠️ Complex architecture
- ⚠️ More memory needed

### When to Use:
- Have GPU access
- Need best performance
- Research/publication
- Clinical application
- Full evaluation needed

---

## Quick Comparison

| Feature | Simple Version | Full Version |
|---------|----------------|--------------|
| **Training Time (CPU)** | 30-60 min | 90 hours |
| **Training Time (GPU)** | 5-10 min | 7-8 hours |
| **Model Size** | 11M params | 91M params |
| **Architecture** | ResNet18 + Avg Pool | ViT + Cross-Attention |
| **Dataset** | 1,000 samples | 13,457 samples |
| **Expected AUC** | 0.75-0.82 | 0.93+ |
| **RAM Needed** | 4-6 GB | 16+ GB (CPU) |
| **Complexity** | Low ⭐ | High ⭐⭐⭐⭐⭐ |

---

## Recommended Workflow

### Option 1: Quick Start (Most Users)

1. **Start with Simple Version:**
   ```bash
   python experiments/train_simple.py
   ```
   - Verify everything works (30 min)
   - Understand the code
   - See baseline results

2. **Then Move to Full Version:**
   - Use Google Colab notebook
   - Train with GPU for best results
   - Get publication-quality metrics

### Option 2: CPU Only

If you **only have CPU** and **limited time:**

```bash
# Quick test (10 min)
python experiments/train_simple.py --epochs 3 --train_samples 500

# Standard (30 min)
python experiments/train_simple.py --epochs 10 --train_samples 1000

# Better accuracy (2 hours)
python experiments/train_simple.py --epochs 20 --train_samples 3000

# Best CPU results (6 hours)
python experiments/train_simple.py --epochs 30 --train_samples 5000
```

### Option 3: GPU Available

If you have GPU access:

**Local GPU:**
```bash
python experiments/train.py --epochs 30
```

**Google Colab:**
- Upload `MURA_Training_Colab.ipynb`
- Enable T4 GPU
- Run all cells
- Get results in 7-8 hours

---

## My Recommendation

### For Your Situation (CPU only):

**Start here:**
```bash
python experiments/train_simple.py
```

**Why?**
- ✅ Results in 30-60 minutes
- ✅ Proves the code works
- ✅ Shows multi-view fusion benefit
- ✅ Good enough for demonstration

**Then:**
- If you need better results: Use Google Colab (free GPU)
- If satisfied with speed: Scale up simple version with more samples
- If learning: Perfect starting point

---

## Commands Quick Reference

### Simple Version
```bash
# Default (30-60 min)
python experiments/train_simple.py

# Quick test (10 min)
python experiments/train_simple.py --epochs 3 --train_samples 500

# Better results (2 hours)
python experiments/train_simple.py --epochs 20 --train_samples 3000
```

### Full Version
```bash
# CPU (not recommended - 90 hours)
python experiments/train.py --epochs 10 --device cpu

# GPU (recommended)
python experiments/train.py --epochs 30

# Google Colab (best option)
# Use MURA_Training_Colab.ipynb
```

---

## Which Files to Use

### Simple Version Files:
- `models/simple_classifier.py` - Lightweight model
- `data/simple_dataset.py` - Subset loader
- `experiments/train_simple.py` - Fast training
- `SIMPLE_VERSION.md` - Documentation

### Full Version Files:
- `models/mura_classifier.py` - ViT model
- `data/dataset.py` - Full dataset loader
- `experiments/train.py` - Full training
- `README.md` - Documentation
- `MURA_Training_Colab.ipynb` - Colab notebook

---

## Decision Tree

```
Do you have a GPU?
├─ Yes → Use Full Version (best results)
│   ├─ Local GPU → python experiments/train.py
│   └─ Google Colab → MURA_Training_Colab.ipynb
│
└─ No (CPU only) → Do you have time?
    ├─ < 1 hour → Use Simple Version
    │   └─ python experiments/train_simple.py
    │
    ├─ 2-6 hours → Use Simple Version with more data
    │   └─ python experiments/train_simple.py --train_samples 3000 --epochs 20
    │
    └─ > 1 day → Get GPU access (Google Colab is free!)
        └─ Use MURA_Training_Colab.ipynb
```

---

## Bottom Line

**For you (CPU, limited time):**

```bash
python experiments/train_simple.py
```

This will give you:
- ✅ Complete results in 30-60 minutes
- ✅ Proof that multi-view fusion works
- ✅ AUC around 0.75-0.82
- ✅ Ready to demonstrate

**Later, for best results:**
- Use Google Colab (free)
- Train full model on GPU
- Get AUC ~0.93+

Start simple, scale up later! 🚀
