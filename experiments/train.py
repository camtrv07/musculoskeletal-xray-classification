"""
Main training script for MURA abnormality detection.

Usage:
    python experiments/train.py --batch_size 8 --lr 1e-4
    python experiments/train.py --resume checkpoints/latest.pth
"""

import sys
import os
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.dataset import get_dataloaders
from models.mura_classifier import create_model
from training.trainer import Trainer
from utils.seed import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MURA abnormality detection model')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')

    # Model parameters
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        choices=['cross_attention', 'concatenation', 'pooling'],
                        help='Type of multi-view fusion (default: cross_attention)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on (default: cuda)')

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config()

    # Override config with command line arguments
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    config.DEVICE = args.device

    # Set random seed for reproducibility
    set_seed(config.SEED)

    # Create directories
    config.create_dirs()

    # Display configuration
    config.display()

    print("\n" + "="*60)
    print("MURA Abnormality Detection - Training")
    print("="*60)

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(config)

    # Create model
    print("\nInitializing model...")
    model = create_model(config, fusion_type=args.fusion_type, device=args.device)

    # Create trainer
    print("\nSetting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )

    # Train
    print("\nStarting training...")
    trainer.train(
        num_epochs=config.NUM_EPOCHS,
        resume_from=args.resume
    )

    # Print final results
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation AUC: {trainer.get_best_auc():.4f}")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")
    print(f"Logs saved to: {config.LOG_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
