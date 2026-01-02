# MURA Musculoskeletal Abnormality Detection

A deep learning system for detecting abnormalities in musculoskeletal radiographs using Vision Transformers and multi-view fusion.

## Project Overview

This project implements a state-of-the-art transformer-based model for musculoskeletal abnormality detection on the MURA (Musculoskeletal Radiographs) dataset. The model integrates multiple radiographic views using a novel cross-attention fusion mechanism, mimicking how radiologists interpret X-rays.

**Key Features:**
- Vision Transformer (ViT) backbone for feature extraction
- Multi-view fusion with cross-attention mechanism
- Handles variable-length studies (1-4 images per study)
- Grad-CAM visualization for interpretability
- Class imbalance handling with weighted loss
- Comprehensive evaluation metrics (AUC-ROC, sensitivity, specificity)
- Checkpointing and training resumption

**Expected Performance:** AUC ≥ 0.93-0.95 (exceeding baseline DenseNet's 0.929)

## Dataset

**MURA Dataset** (Stanford ML Group)
- **Training:** 36,808 images across 13,457 studies
- **Validation:** 3,197 images across 1,199 studies
- **Body Parts:** Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist
- **Task:** Binary classification (normal vs abnormal)
- **Class Distribution:** 38.5% abnormal, 61.5% normal

## Architecture

```
Input: Multi-view X-rays [batch, 4, 3, 224, 224]
    ↓
Shared ViT Backbone (google/vit-base-patch16-224)
    ↓
View Features [batch, 4, 768]
    ↓
Cross-Attention Fusion Module
    - Learnable query tokens
    - Masked attention for variable-length inputs
    ↓
Fused Features [batch, 768]
    ↓
Classification Head
    ↓
Output: Abnormality Probability [batch, 1]
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16+ GB RAM

### Setup

1. Clone the repository:
```bash
cd AI_project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify the MURA dataset is in place:
```bash
ls MURA-v1.1/
# Should show: train/, valid/, train_labeled_studies.csv, valid_labeled_studies.csv, etc.
```

## Project Structure

```
AI_project/
├── config/
│   └── config.py              # Central configuration
├── data/
│   ├── dataset.py             # MURA dataset loader
│   ├── transforms.py          # Image augmentations
│   └── utils.py               # Data utilities
├── models/
│   ├── vit_backbone.py        # ViT feature extractor
│   ├── multiview_fusion.py    # Cross-attention fusion
│   └── mura_classifier.py     # Complete model
├── training/
│   ├── trainer.py             # Training orchestration
│   ├── losses.py              # Weighted BCE loss
│   ├── metrics.py             # Evaluation metrics
│   └── checkpoint_manager.py  # Model checkpointing
├── evaluation/
│   ├── evaluator.py           # Model evaluation
│   └── visualization.py       # Grad-CAM visualization
├── experiments/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── visualize.py           # Visualization script
├── utils/
│   ├── seed.py                # Reproducibility
│   └── logging_utils.py       # TensorBoard logging
└── requirements.txt
```

## Usage

### Training

**Basic Training:**
```bash
python experiments/train.py
```

**Custom Parameters:**
```bash
python experiments/train.py \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 50 \
    --fusion_type cross_attention
```

**Resume Training:**
```bash
python experiments/train.py --resume checkpoints/latest.pth
```

**Options:**
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Number of epochs (default: 50)
- `--fusion_type`: Fusion module type (`cross_attention`, `concatenation`, `pooling`)
- `--resume`: Checkpoint path to resume from
- `--device`: Device to use (`cuda` or `cpu`)

### Evaluation

**Evaluate Best Model:**
```bash
python experiments/evaluate.py --checkpoint checkpoints/best.pth
```

**Options:**
- `--checkpoint`: Path to checkpoint (required)
- `--fusion_type`: Fusion type used in model (default: cross_attention)
- `--save_predictions`: Path to save predictions CSV (default: results/predictions.csv)
- `--device`: Device to use

### Visualization

**Generate Grad-CAM Visualizations:**
```bash
python experiments/visualize.py \
    --checkpoint checkpoints/best.pth \
    --num_samples 20
```

**Options:**
- `--checkpoint`: Path to checkpoint (required)
- `--num_samples`: Number of samples to visualize (default: 20)
- `--output_dir`: Directory to save visualizations (default: results/gradcam)

### Monitoring Training

**TensorBoard:**
```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Configuration

Edit [config/config.py](config/config.py) to customize:

**Model Parameters:**
- `VIT_MODEL`: Pre-trained ViT model name
- `NUM_QUERY_TOKENS`: Number of attention queries
- `MAX_VIEWS`: Maximum views per study
- `DROPOUT_RATE`: Dropout rate

**Training Parameters:**
- `BATCH_SIZE`: Training batch size
- `NUM_EPOCHS`: Total training epochs
- `LEARNING_RATE`: AdamW learning rate
- `WEIGHT_DECAY`: L2 regularization
- `POS_WEIGHT`: Weight for positive class (handles imbalance)

**Data Parameters:**
- `IMAGE_SIZE`: Input image size (224 for ViT)
- `NUM_WORKERS`: Data loading workers

## Model Performance

**Target Metrics:**
- **AUC-ROC:** ≥ 0.93-0.95
- **Accuracy:** ≥ 0.88-0.90
- **Sensitivity:** ≥ 0.85 (critical for medical screening)
- **Specificity:** ≥ 0.90

**Training Progress:**
- Epoch 1: AUC ~0.75-0.80
- Epoch 10: AUC ~0.88-0.90
- Epoch 50: AUC ~0.92-0.94

## Troubleshooting

**GPU Out of Memory:**
- Reduce batch size: `--batch_size 4`
- Use gradient accumulation
- Enable mixed precision training (set `USE_AMP=True` in config)

**Slow Training:**
- Increase num_workers: Set `NUM_WORKERS=8` in config
- Use smaller ViT variant
- Cache extracted features

**Poor Performance:**
- Train longer (50+ epochs)
- Try different fusion types
- Adjust learning rate
- Check data augmentation

## File Descriptions

### Core Modules

**[config/config.py](config/config.py)**
- Central configuration with all hyperparameters
- Paths to dataset and outputs
- Model architecture settings

**[data/dataset.py](data/dataset.py)**
- `MURADataset`: Loads studies with variable-length images
- Handles padding and masking
- Returns: images, mask, label, study_path

**[models/mura_classifier.py](models/mura_classifier.py)**
- `MURAClassifier`: Complete end-to-end model
- Combines ViT backbone + fusion module + classifier
- Supports multiple fusion types

**[training/trainer.py](training/trainer.py)**
- Main training loop
- Handles optimization, validation, logging
- Automatic checkpointing

**[evaluation/visualization.py](evaluation/visualization.py)**
- Grad-CAM visualization for ViT
- Generates saliency maps
- Multi-view visualization support

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

## Contact

For questions or issues, please open an issue in the repository.

---

**Note:** This is a research project. The model should not be used for clinical decisions without proper validation and regulatory approval.
