# Training Visualization & Analysis Guide

## ✅ What Gets Saved Now

I've created an **enhanced training script** that saves everything!

### Automatic Saves

When you run:
```bash
python experiments/train_simple_enhanced.py
```

**You get:**

1. **Model Checkpoint** → `checkpoints/simple_best.pth`
   - Model weights
   - Optimizer state
   - Training history
   - All metrics

2. **Training History CSV** → `results/simple_training/training_history.csv`
   - Loss per epoch (train & validation)
   - AUC per epoch
   - Accuracy per epoch
   - Sensitivity & specificity

3. **Training History JSON** → `results/simple_training/training_history.json`
   - Same data in JSON format
   - Easy to load programmatically

4. **Training Plots** → `results/simple_training/training_plots.png`
   - Loss curves
   - AUC curves
   - Accuracy curves
   - Sensitivity & specificity

5. **Training Summary** → `results/simple_training/training_summary.txt`
   - Full text report
   - Configuration details
   - Best metrics
   - Epoch-by-epoch history

## 📊 Visualization Options

### Option 1: Automatic Plots (Instant)

The enhanced training script automatically generates plots at the end!

```bash
python experiments/train_simple_enhanced.py
```

**After training, check:**
- `results/simple_training/training_plots.png` - Ready-made visualization

### Option 2: Jupyter Notebook (Interactive)

For deep analysis and exploration:

```bash
jupyter notebook notebooks/Training_Analysis.ipynb
```

**What the notebook shows:**
1. ✅ Training curves (loss, AUC, accuracy)
2. ✅ Confusion matrix with heatmap
3. ✅ ROC curve
4. ✅ Prediction distributions
5. ✅ Error analysis (false positives/negatives)
6. ✅ Confidence analysis
7. ✅ Comprehensive summary report

### Option 3: View Existing Results

If you already trained with the old script, you can still analyze:

1. Load the checkpoint:
   ```python
   import torch
   checkpoint = torch.load('checkpoints/simple_best.pth')
   print(checkpoint.keys())
   ```

2. Extract metrics:
   ```python
   if 'metrics' in checkpoint:
       print(checkpoint['metrics'])
   if 'history' in checkpoint:
       print(checkpoint['history'])
   ```

## 🚀 Quick Start

### 1. Train with Full Logging

```bash
# Instead of the old script
python experiments/train_simple.py

# Use the enhanced version
python experiments/train_simple_enhanced.py
```

**Same training, but with:**
- ✅ Automatic plot generation
- ✅ CSV export of metrics
- ✅ Detailed summary report

### 2. Open Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/Training_Analysis.ipynb
# Run all cells (Cell → Run All)
```

**You'll see:**
- Beautiful interactive plots
- Detailed analysis
- Error breakdowns
- Confidence metrics

## 📁 File Structure After Training

```
AI_project/
├── checkpoints/
│   └── simple_best.pth              # Best model
├── results/
│   └── simple_training/
│       ├── training_history.csv     # Metrics per epoch
│       ├── training_history.json    # Same in JSON
│       ├── training_plots.png       # Auto-generated plots
│       ├── training_summary.txt     # Text report
│       ├── predictions.csv          # All predictions (from notebook)
│       ├── confusion_matrix.png     # From notebook
│       ├── roc_curve.png            # From notebook
│       ├── prediction_distribution.png
│       └── confidence_analysis.png
└── notebooks/
    └── Training_Analysis.ipynb      # Interactive analysis
```

## 📈 What Each Visualization Shows

### Training Plots (Auto-generated)

**4 panels:**
1. **Loss:** Shows if model is learning (should decrease)
2. **AUC:** Main metric (should increase to 0.75-0.85)
3. **Accuracy:** Overall correctness
4. **Sensitivity & Specificity:** Balance between detecting abnormals vs normals

### Confusion Matrix (Notebook)

Shows:
- **True Positives (TP):** Correctly identified abnormal cases
- **True Negatives (TN):** Correctly identified normal cases
- **False Positives (FP):** Normal cases wrongly labeled as abnormal
- **False Negatives (FN):** Abnormal cases missed

### ROC Curve (Notebook)

Shows trade-off between:
- True Positive Rate (sensitivity)
- False Positive Rate (1 - specificity)

**Higher AUC = Better model** (max = 1.0)

### Prediction Distribution (Notebook)

Shows:
- How confident the model is
- Separation between normal and abnormal cases
- If threshold (0.5) is appropriate

## 🔍 Reading the Results

### Good Training Signs

✅ **Validation loss decreasing** (model improving)
✅ **Val AUC ≥ 0.75** (good performance on subset)
✅ **Train/Val gap < 0.1** (not overfitting)
✅ **Balanced sensitivity & specificity** (not biased)

### Warning Signs

⚠️ **Validation loss increasing** (overfitting)
⚠️ **Large train/val gap** (model memorizing)
⚠️ **Very low sensitivity OR specificity** (imbalanced)
⚠️ **AUC not improving after epoch 5** (need longer training or more data)

## 💡 Tips

### Compare Different Runs

Train with different settings and compare:

```bash
# Run 1: Default
python experiments/train_simple_enhanced.py
mv results/simple_training results/run1

# Run 2: More data
python experiments/train_simple_enhanced.py --train_samples 2000
mv results/simple_training results/run2

# Run 3: More epochs
python experiments/train_simple_enhanced.py --epochs 20
mv results/simple_training results/run3
```

Then compare the `training_plots.png` files!

### Export for Reports

All plots are saved as high-resolution PNG files:
- Perfect for LaTeX documents
- Ready for PowerPoint
- Publication quality (300 DPI)

### Interactive Analysis

The Jupyter notebook is fully interactive:
- Change plot styles
- Add custom analysis
- Export to PDF
- Generate custom reports

## 🎯 Use Cases

### Quick Check
```bash
python experiments/train_simple_enhanced.py
# Look at: results/simple_training/training_plots.png
```

### Deep Dive
```bash
jupyter notebook notebooks/Training_Analysis.ipynb
# Explore interactively
```

### Report Generation
```bash
# Train with enhanced script
python experiments/train_simple_enhanced.py

# Grab files for your report:
# - results/simple_training/training_plots.png
# - results/simple_training/training_summary.txt
# - results/simple_training/confusion_matrix.png (from notebook)
```

## 📝 Example: Reading Your Results

After training completes:

1. **Check summary:**
   ```bash
   cat results/simple_training/training_summary.txt
   ```

2. **View plots:**
   - Open `results/simple_training/training_plots.png`
   - Look for decreasing loss, increasing AUC

3. **Open notebook for details:**
   ```bash
   jupyter notebook notebooks/Training_Analysis.ipynb
   ```

4. **Check specific metrics:**
   ```python
   import pandas as pd
   history = pd.read_csv('results/simple_training/training_history.csv')
   print(history[['val_auc', 'val_acc']].describe())
   ```

## ❓ FAQ

**Q: Can I visualize the old training (train_simple.py)?**
A: Partially. The old script only saves the model. Use the enhanced version for full logging.

**Q: Where are the plots saved?**
A: `results/simple_training/*.png`

**Q: Can I modify the notebook?**
A: Yes! It's fully customizable. Add your own analysis.

**Q: How do I share results?**
A: Share the entire `results/simple_training/` folder or just the PNG files.

**Q: Can I use this for my article?**
A: Yes! All plots are publication-ready (300 DPI).

## 🎓 Next Steps

1. **Run enhanced training:**
   ```bash
   python experiments/train_simple_enhanced.py
   ```

2. **Check automatic plots:**
   ```bash
   open results/simple_training/training_plots.png
   ```

3. **Deep dive in notebook:**
   ```bash
   jupyter notebook notebooks/Training_Analysis.ipynb
   ```

4. **Use plots in your article/presentation!**

---

**Everything is now automatically saved and visualized!** 🎉
