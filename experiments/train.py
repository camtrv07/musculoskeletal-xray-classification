"""
Training script with experiment tracking and comparison.

Automatically logs all experiments with unique IDs for easy comparison.

Usage:
    python experiments/train.py --name "baseline"
    python experiments/train.py --name "more_epochs" --epochs 20
    python experiments/train.py --name "higher_lr" --lr 5e-3
    python experiments/train.py --name "more_data" --train_samples 3000
"""

import sys
import os
import argparse
import torch
import time
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.simple_dataset import get_simple_dataloaders
from models.simple_classifier import create_simple_model
from training.losses import WeightedBCELoss
from training.metrics import MetricsCalculator
from utils.seed import set_seed
from tqdm import tqdm


class ExperimentLogger:
    """Logs experiments for comparison."""

    def __init__(self, experiments_dir='experiments_log'):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        self.experiments_file = os.path.join(experiments_dir, 'experiments.json')

        # Load existing experiments
        if os.path.exists(self.experiments_file):
            with open(self.experiments_file, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = []

    def log_experiment(self, name, config, history, best_metrics, elapsed_time):
        """Log a new experiment."""
        exp_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        experiment = {
            'id': exp_id,
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'history': history,
            'best_metrics': best_metrics,
            'final_metrics': {
                'train_auc': history['train_auc'][-1],
                'val_auc': history['val_auc'][-1],
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1],
            },
            'elapsed_time': elapsed_time,
            'time_per_epoch': elapsed_time / len(history['train_loss'])
        }

        self.experiments.append(experiment)

        # Save to JSON
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

        # Save individual experiment
        exp_file = os.path.join(self.experiments_dir, f'{exp_id}.json')
        with open(exp_file, 'w') as f:
            json.dump(experiment, f, indent=2)

        print(f"\n✓ Experiment logged: {exp_id}")
        print(f"  Saved to: {exp_file}")

        return exp_id


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

        logits, _ = model(images, mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    parser = argparse.ArgumentParser(description='MURA training with experiment tracking')
    parser.add_argument('--name', type=str, required=True, help='Experiment name (e.g., "baseline", "high_lr")')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=1000, help='Training samples')
    parser.add_argument('--val_samples', type=int, default=200, help='Validation samples')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer')
    args = parser.parse_args()

    print("="*70)
    print(f"MURA Training - Experiment: {args.name}")
    print("="*70)
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"Optimizer:        {args.optimizer}")
    print(f"Training samples: {args.train_samples}")
    print(f"Val samples:      {args.val_samples}")
    print("="*70 + "\n")

    # Setup
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.create_dirs()
    set_seed(config.SEED)

    device = 'cpu'

    # Initialize experiment logger
    logger = ExperimentLogger()

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

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    criterion = WeightedBCELoss(pos_weight=config.POS_WEIGHT)

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
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")

    best_auc = 0.0
    best_metrics = None
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

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
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'experiment_name': args.name
            }, os.path.join(config.CHECKPOINT_DIR, f'{args.name}_best.pth'))
            print(f"  ✓ New best AUC: {best_auc:.4f}")

    elapsed = time.time() - start_time

    # Log experiment
    exp_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'train_samples': args.train_samples,
        'val_samples': args.val_samples
    }

    exp_id = logger.log_experiment(args.name, exp_config, history, best_metrics, elapsed)

    # Final results
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Experiment ID:    {exp_id}")
    print(f"Best val AUC:     {best_auc:.4f}")
    print(f"Total time:       {elapsed/60:.1f} minutes")
    print(f"Time per epoch:   {elapsed/args.epochs/60:.1f} minutes")
    print(f"\nModel saved:      checkpoints/{args.name}_best.pth")
    print(f"Experiment log:   experiments_log/{exp_id}.json")
    print("="*70)


if __name__ == '__main__':
    main()
