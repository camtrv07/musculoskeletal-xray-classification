# Experiment Comparison Guide

## 🎯 Overview

This guide shows you how to **compare different training runs** to find the best hyperparameters and understand how changes affect your model's performance.

## What You Can Compare

- **Different epochs** (e.g., 10 vs 20 vs 30)
- **Different learning rates** (e.g., 1e-3 vs 1e-4 vs 5e-4)
- **Different batch sizes** (e.g., 4 vs 8 vs 16)
- **Different training data sizes** (e.g., 1000 vs 2000 vs 3000 samples)
- **Any combination** of the above

## 🚀 Quick Start

### Step 1: Run Multiple Experiments

Use `train_with_logging.py` to run experiments with different settings:

```bash
# Experiment 1: Baseline
python experiments/train_with_logging.py --name "baseline" --epochs 10 --lr 1e-3

# Experiment 2: More epochs
python experiments/train_with_logging.py --name "more_epochs" --epochs 20 --lr 1e-3

# Experiment 3: Different learning rate
python experiments/train_with_logging.py --name "higher_lr" --epochs 10 --lr 5e-4

# Experiment 4: More training data
python experiments/train_with_logging.py --name "more_data" --epochs 10 --lr 1e-3 --train_samples 2000
```

**Each experiment is automatically saved** to `experiments_log/` with:
- Unique ID and timestamp
- All hyperparameters
- Complete training history
- Best validation metrics

### Step 2: Compare Results

Open the comparison notebook:

```bash
jupyter notebook notebooks/Compare_Experiments.ipynb
```

Run all cells to see:
- Side-by-side training curves
- Final metrics comparison
- Hyperparameter effect analysis
- Best experiment recommendation

## 📊 What Gets Saved

### Automatic Logging

Every time you run `train_with_logging.py`, the following is saved:

#### 1. Master Log File
**Location**: `experiments_log/experiments.json`

Contains all experiments in one place:
```json
{
  "baseline_20260105_143022": {
    "name": "baseline",
    "timestamp": "2026-01-05 14:30:22",
    "config": {
      "epochs": 10,
      "batch_size": 4,
      "learning_rate": 0.001,
      "train_samples": 1000
    },
    "history": {...},
    "best_metrics": {...}
  }
}
```

#### 2. Individual Experiment Files
**Location**: `experiments_log/baseline_20260105_143022.json`

Each experiment gets its own detailed file with:
- Full configuration
- Epoch-by-epoch history
- Best metrics achieved
- Training duration

#### 3. Training Artifacts
**Location**: `results/exp_baseline_20260105_143022/`

Each experiment creates a directory with:
- `training_history.csv` - Metrics per epoch
- `training_plots.png` - Automatic visualization
- `training_summary.txt` - Text report

#### 4. Model Checkpoint
**Location**: `checkpoints/baseline_20260105_143022_best.pth`

Best model for each experiment is saved separately.

## 🔍 Using the Comparison Notebook

### Loading Experiments

The notebook automatically loads all experiments from `experiments_log/experiments.json`:

```python
# Cell 2: Load all experiments
experiments = load_experiments('experiments_log/experiments.json')
print(f"Found {len(experiments)} experiments")

# List all experiments
list_experiments(experiments)
```

Output:
```
Experiment 1: baseline_20260105_143022
  Name: baseline
  Date: 2026-01-05 14:30:22
  Epochs: 10, LR: 0.001, Batch: 4
  Best AUC: 0.7823

Experiment 2: more_epochs_20260105_151045
  Name: more_epochs
  Date: 2026-01-05 15:10:45
  Epochs: 20, LR: 0.001, Batch: 4
  Best AUC: 0.8156
```

### Selecting Experiments to Compare

**Option 1: Select by name**
```python
experiment_names = ['baseline', 'more_epochs', 'higher_lr']
```

**Option 2: Select by index**
```python
experiment_indices = [0, 1, 2]  # First three experiments
```

**Option 3: Compare all**
```python
experiment_names = None  # Compare everything
```

### Generated Visualizations

#### 1. Training Curves Comparison
**4 panels showing:**
- Training loss over epochs (all experiments)
- Validation loss over epochs
- Training AUC over epochs
- Validation AUC over epochs

**What to look for:**
- Which experiment reaches best AUC fastest
- Which has most stable training (less oscillation)
- Signs of overfitting (train/val gap)

#### 2. Final Metrics Comparison
**4 bar charts showing:**
- Final AUC (primary metric)
- Final Accuracy
- Final Sensitivity (detecting abnormal cases)
- Final Specificity (detecting normal cases)

**What to look for:**
- Highest AUC = best overall performance
- Balanced sensitivity & specificity = no bias

#### 3. Hyperparameter Effects
**4 scatter plots:**
- Learning Rate vs AUC
- Number of Epochs vs AUC
- Batch Size vs AUC
- Training Samples vs AUC

**What to look for:**
- Optimal learning rate
- When more epochs stop helping
- Effect of more training data

#### 4. Summary Table
Markdown table with all key metrics:

| Experiment | Epochs | LR | Batch | Samples | AUC | Acc | Sens | Spec |
|------------|--------|----|----|---------|-----|-----|------|------|
| baseline | 10 | 0.001 | 4 | 1000 | 0.78 | 0.75 | 0.72 | 0.77 |
| more_epochs | 20 | 0.001 | 4 | 1000 | 0.82 | 0.79 | 0.76 | 0.81 |

## 💡 Example Workflows

### Workflow 1: Find Best Learning Rate

```bash
# Try different learning rates with same setup
python experiments/train_with_logging.py --name "lr_1e-2" --lr 1e-2 --epochs 10
python experiments/train_with_logging.py --name "lr_1e-3" --lr 1e-3 --epochs 10
python experiments/train_with_logging.py --name "lr_1e-4" --lr 1e-4 --epochs 10
python experiments/train_with_logging.py --name "lr_5e-4" --lr 5e-4 --epochs 10
```

Then in notebook:
```python
experiment_names = ['lr_1e-2', 'lr_1e-3', 'lr_1e-4', 'lr_5e-4']
```

**Look at:** "Learning Rate vs AUC" scatter plot to find optimal LR.

### Workflow 2: Determine Optimal Epochs

```bash
# Try different training durations
python experiments/train_with_logging.py --name "epochs_5" --epochs 5 --lr 1e-3
python experiments/train_with_logging.py --name "epochs_10" --epochs 10 --lr 1e-3
python experiments/train_with_logging.py --name "epochs_20" --epochs 20 --lr 1e-3
python experiments/train_with_logging.py --name "epochs_30" --epochs 30 --lr 1e-3
```

**Look at:** Training curves to see when validation AUC stops improving.

### Workflow 3: Test Effect of More Data

```bash
# Scale up training data
python experiments/train_with_logging.py --name "data_500" --train_samples 500 --epochs 15
python experiments/train_with_logging.py --name "data_1000" --train_samples 1000 --epochs 15
python experiments/train_with_logging.py --name "data_2000" --train_samples 2000 --epochs 15
python experiments/train_with_logging.py --name "data_3000" --train_samples 3000 --epochs 15
```

**Look at:** "Training Samples vs AUC" to see diminishing returns point.

### Workflow 4: Complete Grid Search

```bash
# Systematic exploration (4 experiments)
for lr in 1e-3 5e-4; do
  for epochs in 10 20; do
    python experiments/train_with_logging.py \
      --name "grid_lr${lr}_ep${epochs}" \
      --lr $lr --epochs $epochs
  done
done
```

**Look at:** All visualizations to find best combination.

## 📈 Interpreting Results

### Good Signs

✅ **Increasing validation AUC** - Model is learning
✅ **Small train/val gap** - Not overfitting
✅ **Stable curves** - Reliable training
✅ **Balanced sensitivity/specificity** - No bias

### Warning Signs

⚠️ **Decreasing validation AUC** - Overfitting or bad hyperparameters
⚠️ **Large train/val gap** - Model memorizing training data
⚠️ **Oscillating curves** - Learning rate too high
⚠️ **Very low sensitivity OR specificity** - Imbalanced predictions

### Making Decisions

**If validation AUC plateaus early:**
- Try higher learning rate
- Try different architecture
- Check for data issues

**If train/val gap is large:**
- Reduce model complexity
- Add regularization
- Get more training data

**If training is unstable:**
- Lower learning rate
- Reduce batch size
- Check data preprocessing

## 🎯 Best Practices

### Naming Experiments

Use **descriptive names** that explain what you're testing:

```bash
# Good names
python experiments/train_with_logging.py --name "baseline_resnet18"
python experiments/train_with_logging.py --name "augmented_data"
python experiments/train_with_logging.py --name "lr_sweep_1e3"

# Bad names
python experiments/train_with_logging.py --name "test1"
python experiments/train_with_logging.py --name "new"
python experiments/train_with_logging.py --name "final"
```

### Organizing Experiments

Group related experiments with prefixes:

```bash
# Learning rate experiments
--name "lr_1e-2"
--name "lr_1e-3"
--name "lr_1e-4"

# Epoch experiments
--name "epoch_10"
--name "epoch_20"
--name "epoch_30"

# Data size experiments
--name "data_1k"
--name "data_2k"
--name "data_3k"
```

### Running Experiments

**Sequential (safe):**
```bash
python experiments/train_with_logging.py --name "exp1" --epochs 10
python experiments/train_with_logging.py --name "exp2" --epochs 20
```

**Batch script:**
```bash
#!/bin/bash
# run_experiments.sh

experiments=(
  "baseline:10:1e-3"
  "more_epochs:20:1e-3"
  "higher_lr:10:5e-4"
)

for exp in "${experiments[@]}"; do
  IFS=':' read -r name epochs lr <<< "$exp"
  python experiments/train_with_logging.py \
    --name "$name" --epochs "$epochs" --lr "$lr"
done
```

### Comparing Results

**Start with few experiments** (2-3) to understand patterns:
```python
experiment_names = ['baseline', 'more_epochs']
```

**Then expand** to full comparison:
```python
experiment_names = None  # Compare all
```

## 📁 File Structure

After running multiple experiments:

```
AI_project/
├── experiments_log/
│   ├── experiments.json                    # Master log (all experiments)
│   ├── baseline_20260105_143022.json      # Individual experiment files
│   ├── more_epochs_20260105_151045.json
│   └── higher_lr_20260105_163012.json
├── checkpoints/
│   ├── baseline_20260105_143022_best.pth  # Best model per experiment
│   ├── more_epochs_20260105_151045_best.pth
│   └── higher_lr_20260105_163012_best.pth
├── results/
│   ├── exp_baseline_20260105_143022/
│   │   ├── training_history.csv
│   │   ├── training_plots.png
│   │   └── training_summary.txt
│   ├── exp_more_epochs_20260105_151045/
│   └── exp_higher_lr_20260105_163012/
└── notebooks/
    └── Compare_Experiments.ipynb           # Analysis notebook
```

## 🔧 Command Reference

### Training with Logging

**Basic:**
```bash
python experiments/train_with_logging.py --name "experiment_name"
```

**All options:**
```bash
python experiments/train_with_logging.py \
  --name "my_experiment" \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-3 \
  --train_samples 2000 \
  --val_samples 400
```

**Default values:**
- `--name`: Required (no default)
- `--epochs`: 10
- `--batch_size`: 4
- `--lr`: 1e-3 (0.001)
- `--train_samples`: 1000
- `--val_samples`: 200

### Viewing Results

**List all experiments:**
```bash
cat experiments_log/experiments.json | python -m json.tool
```

**Count experiments:**
```bash
python -c "import json; print(len(json.load(open('experiments_log/experiments.json'))))"
```

**Compare in notebook:**
```bash
jupyter notebook notebooks/Compare_Experiments.ipynb
```

## ❓ FAQ

**Q: Can I delete old experiments?**
A: Yes, but do it carefully:
```bash
# Remove individual experiment
rm experiments_log/old_experiment_*.json
rm -rf results/exp_old_experiment_*
rm checkpoints/old_experiment_*_best.pth

# Then manually edit experiments_log/experiments.json
```

**Q: How many experiments can I compare at once?**
A: The notebook can handle 10+ experiments, but visualizations are clearest with 3-5 experiments.

**Q: Can I resume a failed experiment?**
A: No, currently each run is independent. If training fails, run it again with a new name.

**Q: Can I compare experiments from different dates?**
A: Yes! All experiments are stored together and can be compared regardless of when they were run.

**Q: What if I run the same name twice?**
A: Each experiment gets a unique timestamp, so "baseline" run at different times becomes:
- `baseline_20260105_143022`
- `baseline_20260105_163045`

**Q: Can I export comparison results?**
A: Yes! The notebook's last cell generates a markdown report:
```python
# Export to markdown
report = generate_comparison_report(selected_experiments)
with open('comparison_report.md', 'w') as f:
    f.write(report)
```

**Q: How do I share experiments with collaborators?**
A: Share the entire `experiments_log/` folder. They can run the comparison notebook with your data.

## 🎓 Example Session

Here's a complete example session from start to finish:

### 1. Run Baseline

```bash
python experiments/train_with_logging.py --name "baseline" --epochs 10
```

Output:
```
Training Complete!
Best validation AUC: 0.7823
Experiment ID: baseline_20260105_143022
```

### 2. Try Improvements

```bash
# More training
python experiments/train_with_logging.py --name "more_epochs" --epochs 20

# Different learning rate
python experiments/train_with_logging.py --name "lower_lr" --lr 5e-4 --epochs 10

# More data
python experiments/train_with_logging.py --name "more_data" --train_samples 2000 --epochs 15
```

### 3. Compare Results

```bash
jupyter notebook notebooks/Compare_Experiments.ipynb
```

In the notebook, set:
```python
experiment_names = ['baseline', 'more_epochs', 'lower_lr', 'more_data']
```

### 4. Analyze Results

Looking at the comparison plots:
- **Training curves:** "more_epochs" reaches AUC 0.82 (best)
- **Final metrics:** "more_data" has highest accuracy (0.81)
- **Hyperparameter effects:** LR 5e-4 is optimal

### 5. Run Best Configuration

```bash
python experiments/train_with_logging.py \
  --name "final_best" \
  --epochs 20 \
  --lr 5e-4 \
  --train_samples 2000
```

Result: AUC 0.84 🎉

## 🚀 Next Steps

1. **Start simple:** Run 2-3 experiments with different epoch counts
2. **Compare:** Use the notebook to see which is best
3. **Iterate:** Test different hyperparameters based on results
4. **Document:** Keep notes on what you learn from each comparison
5. **Share:** Use the markdown export to share findings

---

**Happy experimenting!** 🔬

For more details on training and visualization, see:
- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - All visualization options
- [SIMPLE_VERSION.md](SIMPLE_VERSION.md) - CPU training guide
- [README.md](README.md) - Full project documentation
