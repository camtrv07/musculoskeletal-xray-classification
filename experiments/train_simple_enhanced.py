"""
Enhanced simple training script with full logging and saving.

Saves:
- Training history (loss, metrics per epoch)
- Best model checkpoint
- Training plots
- Final evaluation results

Usage:
    python experiments/train_simple_enhanced.py
"""

import sys
import os
import argparse
import torch
import time
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

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


def plot_training_history(history, save_path):
    """Create training visualization plots."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # AUC
    axes[0, 1].plot(epochs, history['train_auc'], 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs, history['val_auc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('AUC-ROC', fontsize=12)
    axes[0, 1].set_title('AUC-ROC Score', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 0].plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2)
    axes[1, 0].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Sensitivity & Specificity
    axes[1, 1].plot(epochs, history['val_sensitivity'], 'g-', label='Sensitivity', linewidth=2)
    axes[1, 1].plot(epochs, history['val_specificity'], 'm-', label='Specificity', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Sensitivity & Specificity', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved training plots to: {save_path}")


def save_training_summary(history, best_metrics, args, elapsed_time, save_path):
    """Save training summary as text file."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MURA Simple Training - Summary Report\n")
        f.write("="*60 + "\n\n")

        f.write("Training Configuration:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Training samples: {args.train_samples}\n")
        f.write(f"  Validation samples: {args.val_samples}\n")
        f.write(f"  Total time: {elapsed_time/60:.1f} minutes\n")
        f.write(f"  Time per epoch: {elapsed_time/args.epochs/60:.1f} minutes\n\n")

        f.write("Final Results (Best Epoch):\n")
        f.write(f"  AUC-ROC: {best_metrics['auc']:.4f}\n")
        f.write(f"  Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"  Sensitivity: {best_metrics['sensitivity']:.4f}\n")
        f.write(f"  Specificity: {best_metrics['specificity']:.4f}\n")
        f.write(f"  Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"  F1 Score: {best_metrics['f1']:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"  True Positives:  {best_metrics['tp']}\n")
        f.write(f"  True Negatives:  {best_metrics['tn']}\n")
        f.write(f"  False Positives: {best_metrics['fp']}\n")
        f.write(f"  False Negatives: {best_metrics['fn']}\n\n")

        f.write("Training History (per epoch):\n")
        f.write(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>9} | {'Train AUC':>10} | {'Val AUC':>8} | {'Val Acc':>8}\n")
        f.write("-" * 80 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:6d} | {history['train_loss'][i]:11.4f} | "
                   f"{history['val_loss'][i]:9.4f} | {history['train_auc'][i]:10.4f} | "
                   f"{history['val_auc'][i]:8.4f} | {history['val_acc'][i]:8.4f}\n")

    print(f"  Saved training summary to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced simple MURA training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=1000, help='Training samples')
    parser.add_argument('--val_samples', type=int, default=200, help='Validation samples')
    args = parser.parse_args()

    print("="*60)
    print("MURA Simple Training (Enhanced with Full Logging)")
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

    # Create results directory
    results_dir = os.path.join(config.RESULTS_DIR, 'simple_training')
    os.makedirs(results_dir, exist_ok=True)

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

    # Training history
    history = {
        'train_loss': [],
        'train_auc': [],
        'train_acc': [],
        'val_loss': [],
        'val_auc': [],
        'val_acc': [],
        'val_sensitivity': [],
        'val_specificity': []
    }

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    best_auc = 0.0
    best_metrics = None
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Store history
        history['train_loss'].append(train_metrics['loss'])
        history['train_auc'].append(train_metrics['auc'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | AUC: {train_metrics['auc']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"Sens: {val_metrics['sensitivity']:.4f} | Spec: {val_metrics['specificity']:.4f}")

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'history': history,
                'args': vars(args)
            }, os.path.join(config.CHECKPOINT_DIR, 'simple_best.pth'))
            print(f"  ✓ New best AUC: {best_auc:.4f}")

    elapsed = time.time() - start_time

    # Save training history as CSV
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(results_dir, 'training_history.csv')
    history_df.to_csv(history_csv, index=False)
    print(f"\n✓ Saved training history to: {history_csv}")

    # Save training history as JSON
    history_json = os.path.join(results_dir, 'training_history.json')
    with open(history_json, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to: {history_json}")

    # Plot training curves
    plot_path = os.path.join(results_dir, 'training_plots.png')
    plot_training_history(history, plot_path)

    # Save summary
    summary_path = os.path.join(results_dir, 'training_summary.txt')
    save_training_summary(history, best_metrics, args, elapsed, summary_path)

    # Final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Time per epoch: {elapsed/args.epochs/60:.1f} minutes")
    print(f"\nSaved files:")
    print(f"  Model: {config.CHECKPOINT_DIR}/simple_best.pth")
    print(f"  History CSV: {history_csv}")
    print(f"  History JSON: {history_json}")
    print(f"  Training plots: {plot_path}")
    print(f"  Summary: {summary_path}")
    print("="*60)


if __name__ == '__main__':
    main()
