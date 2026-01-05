# Google Colab Quick Start (3 Steps)

Train your MURA model 10-15× faster with free GPU!

## Step 1: Upload to Google Drive (One Time)

1. Open [Google Drive](https://drive.google.com)
2. Upload `MURA-v1.1` folder
3. Note the path (e.g., `MyDrive/MURA-v1.1`)

## Step 2: Open Notebook in Colab

1. Upload `MURA_Training_Colab.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory
3. **IMPORTANT:** Click `Runtime` → `Change runtime type` → Select `T4 GPU` → Save

## Step 3: Update Path & Run

1. In the notebook, update this line:
   ```python
   MURA_PATH = '/content/drive/MyDrive/MURA-v1.1'  # Your actual path
   ```

2. Click `Runtime` → `Run all`

3. Wait ~8 hours for training to complete!

## Results

After training:
- Best model: `checkpoints/best.pth`
- Metrics: AUC ~0.88-0.91 (10 epochs)
- Visualizations: `results/gradcam/`

## Need Help?

See detailed guide: [COLAB_GUIDE.md](COLAB_GUIDE.md)

## Speed Comparison

- **CPU (local):** 90 hours for 10 epochs
- **Colab GPU (free):** 7-8 hours for 10 epochs ⚡

Start your training now! 🚀
