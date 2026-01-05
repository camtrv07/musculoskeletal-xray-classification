"""
Simplified training script for CPU.

Fast training on a subset of data:
- ResNet18 instead of ViT (10x faster)
- 1000 training samples instead of 13,457
- Simple averaging instead of attention
- Completes in 30-60 minutes on CPU

Usage:
    python experiments/train_simple.py
    python experiments/train_simple.py --epochs 5 --batch_size 8
"""

import sys
import os
import argparse
import torch
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.simple_dataset import get_simple_dataloaders
from models.simple_classifier import create_simple_model
from training.losses import WeightedBCELoss
from training.metrics import MetricsCalculator
from utils.seed import set_seed
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    metrics_calc = MetricsCalculator()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['images'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)

        # Forward
        logits, _ = model(images, mask)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        metrics_calc.update(logits.detach(), labels.detach())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = metrics_calc.compute()
    metrics['loss'] = total_loss / len(train_loader)
    return metrics


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    metrics_calc = MetricsCalculator()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['images'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(images, mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            metrics_calc.update(logits, labels)

    metrics = metrics_calc.compute()
    metrics['loss'] = total_loss / len(val_loader)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Simple MURA training for CPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=1000, help='Training samples')
    parser.add_argument('--val_samples', type=int, default=200, help='Validation samples')
    args = parser.parse_args()

    print("="*60)
    print("MURA Simple Training (CPU Optimized)")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Training samples: {args.train_samples}")
    print(f"Validation samples: {args.val_samples}")
    print("="*60 + "\n")

    # Setup
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.create_dirs()
    set_seed(config.SEED)

    device = 'cpu'

    # Data
    print("Loading data...")
    train_loader, val_loader = get_simple_dataloaders(
        config,
        train_limit=args.train_samples,
        val_limit=args.val_samples
    )

    # Model
    print("\nCreating model...")
    model = create_simple_model(device=device, pretrained=True)

    # Training components
    criterion = WeightedBCELoss(pos_weight=config.POS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    best_auc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | AUC: {train_metrics['auc']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': best_auc,
            }, os.path.join(config.CHECKPOINT_DIR, 'simple_best.pth'))
            print(f"  ✓ New best AUC: {best_auc:.4f}")

    # Final results
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Time per epoch: {elapsed/args.epochs/60:.1f} minutes")
    print(f"\nModel saved to: {config.CHECKPOINT_DIR}/simple_best.pth")
    print("="*60)


if __name__ == '__main__':
    main()
