"""
Central configuration for MURA abnormality detection project.
Contains all paths, hyperparameters, and settings.
"""

import os
import torch


class Config:
    """Configuration class for MURA project."""

    # ==================== Paths ====================
    # Project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Data paths (relative to project root)
    DATA_ROOT = os.path.join(PROJECT_ROOT, "MURA-v1.1")
    TRAIN_CSV = os.path.join(DATA_ROOT, "train_labeled_studies.csv")
    VALID_CSV = os.path.join(DATA_ROOT, "valid_labeled_studies.csv")

    # Output paths
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")

    # ==================== Model Architecture ====================
    # ViT backbone
    VIT_MODEL = "google/vit-base-patch16-224"  # Pre-trained ViT model
    HIDDEN_DIM = 768  # ViT-Base hidden dimension

    # Multi-view fusion
    NUM_QUERY_TOKENS = 4  # Number of learnable query tokens for cross-attention
    NUM_ATTENTION_HEADS = 8  # Number of attention heads in fusion module
    MAX_VIEWS = 4  # Maximum number of views per study (for padding)

    # Classification head
    DROPOUT_RATE = 0.2  # Dropout rate in classifier
    FUSION_DROPOUT = 0.1  # Dropout rate in fusion module

    # ==================== Training Hyperparameters ====================
    # Optimization
    BATCH_SIZE = 8  # Batch size (reduce to 4 if GPU memory issues)
    NUM_EPOCHS = 50  # Total training epochs
    LEARNING_RATE = 1e-4  # AdamW learning rate
    WEIGHT_DECAY = 0.01  # L2 regularization
    GRADIENT_CLIP = 1.0  # Max gradient norm for clipping

    # Learning rate scheduler
    SCHEDULER_T0 = 10  # CosineAnnealingWarmRestarts T_0
    SCHEDULER_T_MULT = 2  # CosineAnnealingWarmRestarts T_mult
    WARMUP_EPOCHS = 5  # Number of warmup epochs (optional)

    # Class imbalance
    # Calculated from training set: 38.5% positive, 61.5% negative
    # pos_weight = num_negatives / num_positives = 0.615 / 0.385 ≈ 1.597
    POS_WEIGHT = 1.597  # Weight for positive class in BCE loss

    # ==================== Data Processing ====================
    IMAGE_SIZE = 224  # Input image size for ViT (must be 224 for ViT-Base-Patch16-224)
    NUM_WORKERS = 0 if not torch.cuda.is_available() else 4  # Reduce workers for CPU to save memory
    PIN_MEMORY = torch.cuda.is_available()  # Pin memory only if GPU available

    # Image normalization (grayscale X-rays converted to 3-channel)
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

    # Data augmentation
    RANDOM_CROP_SCALE = (0.8, 1.0)  # Scale range for RandomResizedCrop
    RANDOM_ROTATION = 10  # Max rotation degrees
    BRIGHTNESS = 0.2  # ColorJitter brightness
    CONTRAST = 0.2  # ColorJitter contrast
    HORIZONTAL_FLIP_PROB = 0.5  # Probability of horizontal flip

    # ==================== Checkpointing ====================
    MAX_CHECKPOINTS = 5  # Keep last N epoch checkpoints
    SAVE_EVERY_N_EPOCHS = 1  # Save checkpoint every N epochs

    # ==================== Logging ====================
    LOG_INTERVAL = 50  # Log every N batches during training

    # ==================== Evaluation ====================
    # Metrics to compute
    METRICS = ['accuracy', 'auc', 'sensitivity', 'specificity']

    # Prediction threshold
    THRESHOLD = 0.5  # Classification threshold for binary predictions

    # ==================== Visualization ====================
    GRADCAM_SAMPLES = 20  # Number of samples to generate Grad-CAM for
    GRADCAM_DPI = 150  # DPI for saved Grad-CAM images

    # ==================== Device ====================
    DEVICE = "cuda"  # Will be set to "cpu" if CUDA not available

    # ==================== Random Seed ====================
    SEED = 42  # Random seed for reproducibility

    # ==================== Mixed Precision Training ====================
    USE_AMP = False  # Use automatic mixed precision (set True if GPU supports)

    # ==================== Body Parts ====================
    BODY_PARTS = [
        "XR_ELBOW",
        "XR_FINGER",
        "XR_FOREARM",
        "XR_HAND",
        "XR_HUMERUS",
        "XR_SHOULDER",
        "XR_WRIST"
    ]

    @classmethod
    def create_dirs(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.GRADCAM_DIR, exist_ok=True)

    @classmethod
    def display(cls):
        """Display configuration settings."""
        print("=" * 60)
        print("MURA Abnormality Detection - Configuration")
        print("=" * 60)
        print(f"Data Root: {cls.DATA_ROOT}")
        print(f"Model: {cls.VIT_MODEL}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Max Views: {cls.MAX_VIEWS}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Device: {cls.DEVICE}")
        print("=" * 60)
