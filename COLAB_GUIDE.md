# Google Colab Training Guide

This guide shows you how to train the MURA model on Google Colab's free GPU (10-15x faster than CPU).

## Quick Start (5 minutes)

### Step 1: Upload Dataset to Google Drive

1. Open [Google Drive](https://drive.google.com)
2. Create a folder called `MURA_Project`
3. Upload your `MURA-v1.1` folder to this location
4. Final path should be: `MyDrive/MURA_Project/MURA-v1.1/`

**Note:** This is a one-time upload. You can reuse it in future sessions.

### Step 2: Upload Notebook to Colab

**Option A: Direct Upload**
1. Go to [Google Colab](https://colab.research.google.com)
2. Click `File` → `Upload notebook`
3. Upload `MURA_Training_Colab.ipynb` from your project folder

**Option B: From Drive**
1. Upload `MURA_Training_Colab.ipynb` to Google Drive
2. Right-click → `Open with` → `Google Colaboratory`

### Step 3: Enable GPU

**CRITICAL STEP!**
1. In Colab, click `Runtime` → `Change runtime type`
2. Select `Hardware accelerator: GPU`
3. Choose `GPU type: T4` (free tier)
4. Click `Save`

### Step 4: Upload Project Code

You have 3 options:

#### Option A: Upload ZIP (Easiest)

1. On your local machine, zip the entire AI_project folder
2. In Colab, click the 📁 folder icon on the left
3. Click upload button and upload `AI_project.zip`
4. In the first code cell, uncomment and run:
   ```python
   !unzip -q AI_project.zip
   %cd AI_project
   ```

#### Option B: Upload to Google Drive (Best for Reuse)

1. Upload entire `AI_project` folder to Google Drive
2. In Colab, mount Drive (run the mount cell)
3. Your code will be at `/content/drive/MyDrive/AI_project/`

#### Option C: Upload Files Individually

1. Use Colab's file browser (📁 icon)
2. Upload each `.py` file to its corresponding folder
3. Follow the structure in the notebook

### Step 5: Update Dataset Path

In the notebook, find this line and update it:

```python
MURA_PATH = '/content/drive/MyDrive/MURA_Project/MURA-v1.1'
```

Change to wherever you uploaded your dataset in Drive.

### Step 6: Run Training

Simply run all cells in order (or `Runtime` → `Run all`)!

## What to Expect

### Training Speed Comparison

| Device | Time per Epoch | Total (10 epochs) |
|--------|----------------|-------------------|
| CPU (local) | ~9 hours | ~90 hours |
| **T4 GPU (Colab)** | **~45 min** | **~7.5 hours** |
| A100 GPU (Colab Pro) | ~20 min | ~3.3 hours |

### Expected Results

After 10 epochs on Colab GPU:
- **AUC-ROC:** 0.88-0.91 (excellent progress)
- **Accuracy:** 0.85-0.88
- **Training time:** ~7-8 hours total

For best results (AUC ~0.93+), train for 30-50 epochs (~1-2 days on Colab).

## Monitoring Training

### TensorBoard (Built-in)

The notebook includes TensorBoard integration. Run the TensorBoard cell to see:
- Loss curves
- AUC progression
- Learning rate schedule
- Real-time updates

### Check Progress

```python
# In any code cell:
!tail -20 /content/logs/events.out.tfevents.*
```

## Saving Your Work

### Important: Colab Sessions are Temporary!

Colab deletes everything when the session ends (~12 hours or on disconnect).

### Auto-Save Strategy

The notebook automatically:
1. ✅ Saves checkpoints every epoch
2. ✅ Keeps best model based on AUC
3. ✅ Logs metrics to TensorBoard

### Manual Backup to Drive

Run this cell periodically:

```python
!cp -r /content/checkpoints /content/drive/MyDrive/MURA_Checkpoints/
!cp -r /content/results /content/drive/MyDrive/MURA_Results/
```

This ensures you won't lose work if disconnected!

## Resuming Training

If your session disconnects mid-training:

1. Re-run setup cells (mount Drive, imports, etc.)
2. Find the resume cell and run:
   ```python
   trainer.train(num_epochs=50, resume_from='/content/checkpoints/latest.pth')
   ```

Your training will continue from the last saved epoch!

## Downloading Results

### Download to Your Computer

Run the download cell at the end:
```python
files.download('results.zip')
```

This downloads:
- ✅ Best model checkpoint (`best.pth`)
- ✅ Latest checkpoint (`latest.pth`)
- ✅ Predictions CSV
- ✅ Grad-CAM visualizations

### Keep in Google Drive

Easier: Just copy everything to Drive and access anytime:
```python
!cp -r /content/checkpoints /content/drive/MyDrive/MURA_Final/
```

## Common Issues

### "CUDA out of memory"

**Solution:** Reduce batch size
```python
Config.BATCH_SIZE = 8  # Or even 4
```

### "Session disconnected"

**Causes:**
- Browser tab closed
- Inactive for >90 minutes
- Used too much GPU

**Solutions:**
- Keep tab open
- Move mouse occasionally
- Use Colab Pro for longer sessions ($10/month)

### "Dataset not found"

**Check:**
1. Drive is mounted: `!ls /content/drive/MyDrive`
2. Path is correct: Update `MURA_PATH` variable
3. Dataset uploaded fully: `!ls /content/drive/MyDrive/MURA-v1.1`

### "No GPU available"

**Solutions:**
1. Go to `Runtime` → `Change runtime type`
2. Make sure `GPU` is selected
3. Click `Save`
4. Restart runtime: `Runtime` → `Restart runtime`

### "Quota exceeded"

Free Colab has limits:
- ~12 hours GPU per day
- ~50 GB storage

**Solutions:**
- Wait 12-24 hours for reset
- Upgrade to Colab Pro
- Use Kaggle Notebooks (similar free GPU)

## Tips & Tricks

### 1. Keep Session Alive

```javascript
// Run this in browser console (F12 → Console)
function KeepClicking(){
    console.log("Keeping alive...");
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepClicking, 60000);
```

### 2. Monitor GPU Usage

```python
!nvidia-smi
```

Shows GPU memory usage, temperature, etc.

### 3. Speed Up Data Loading

```python
Config.NUM_WORKERS = 4  # Use more CPU cores
```

### 4. Use Mixed Precision (Faster Training)

```python
Config.USE_AMP = True  # Automatic Mixed Precision
```

Can speed up training by 20-30%!

### 5. Early Stopping

If AUC plateaus, stop early and use the best checkpoint.

## Alternative: Kaggle Notebooks

If Colab quota runs out, try Kaggle:
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Create new notebook
3. Enable GPU (right sidebar)
4. Similar to Colab, ~30 hours/week GPU

## Next Steps After Training

1. **Evaluate:** Run evaluation cell to get final metrics
2. **Visualize:** Generate Grad-CAM to see what model learned
3. **Download:** Save best model for later use
4. **Analyze:** Review predictions CSV for error analysis
5. **Iterate:** Try different hyperparameters or fusion types

## Cost Comparison

| Option | Cost | GPU Time | Notes |
|--------|------|----------|-------|
| **Colab Free** | $0 | ~12 hrs/day | Best for this project |
| Colab Pro | $10/mo | ~unlimited | Faster GPU, priority |
| Kaggle | $0 | 30 hrs/week | Good alternative |
| Local GPU | $500-2000 | unlimited | One-time cost |
| AWS/GCP | ~$1/hour | pay-as-go | Expensive for long training |

**Recommendation:** Start with free Colab. It's perfect for this 7-8 hour training job.

## Expected Timeline

**Complete workflow on Colab:**
- Setup & upload: 10-15 minutes
- Training (10 epochs): 7-8 hours
- Evaluation & visualization: 15 minutes
- **Total:** ~8-9 hours

**Pro tip:** Start training before bed or leaving for the day. It'll be done when you return!

## Questions?

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Make sure GPU is enabled
4. Verify dataset path is correct
5. Try reducing batch size

Good luck with your training! 🚀
