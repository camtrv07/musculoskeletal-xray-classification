"""
Checkpoint management for saving and loading model training state.
Handles best model tracking and old checkpoint cleanup.
"""

import os
import torch
from pathlib import Path


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Features:
        - Saves latest checkpoint after each epoch
        - Saves best checkpoint based on validation metric (AUC)
        - Keeps only N most recent epoch checkpoints
        - Supports resuming training from checkpoint
    """

    def __init__(self, checkpoint_dir, max_checkpoints=5):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir (str): Directory to save checkpoints
            max_checkpoints (int): Maximum number of epoch checkpoints to keep. Default: 5
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_metric = -float('inf')  # Best AUC score

        print(f"Checkpoint manager initialized:")
        print(f"  Directory: {self.checkpoint_dir}")
        print(f"  Max checkpoints: {max_checkpoints}")

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        """
        Save checkpoint with model and training state.

        Args:
            model (nn.Module): Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch (int): Current epoch number
            metrics (dict): Validation metrics
            is_best (bool): Whether this is the best model so far. Default: False
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        print(f"  Saved latest checkpoint: {latest_path}")

        # Save best checkpoint if this is the best model
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.best_metric = metrics['auc']
            print(f"  Saved best checkpoint: {best_path} (AUC: {metrics['auc']:.4f})")

        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
        torch.save(checkpoint, epoch_path)

        # Cleanup old epoch checkpoints
        self._cleanup_old_checkpoints()

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None, device='cuda'):
        """
        Load checkpoint and restore model/training state.

        Args:
            model (nn.Module): Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            checkpoint_path (str): Path to checkpoint file. If None, loads latest. Default: None
            device (str): Device to load tensors to. Default: 'cuda'

        Returns:
            tuple: (epoch, metrics) - Epoch number and metrics from checkpoint
        """
        # Determine which checkpoint to load
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'latest.pth'
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return 0, {}

        print(f"Loading checkpoint from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model state from epoch {checkpoint['epoch']}")

        # Restore optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  Loaded optimizer state")

        # Restore scheduler state if provided
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  Loaded scheduler state")

        # Restore best metric
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']

        epoch = checkpoint['epoch']
        metrics = checkpoint.get('metrics', {})

        print(f"  Resumed from epoch {epoch}")
        if metrics:
            print(f"  Previous metrics: AUC={metrics.get('auc', 0):.4f}, "
                  f"Acc={metrics.get('accuracy', 0):.4f}")

        return epoch, metrics

    def _cleanup_old_checkpoints(self):
        """Remove old epoch checkpoints, keeping only the most recent N."""
        # Get all epoch checkpoints
        epoch_checkpoints = sorted(self.checkpoint_dir.glob('epoch_*.pth'))

        # Remove old checkpoints if we exceed max_checkpoints
        if len(epoch_checkpoints) > self.max_checkpoints:
            for checkpoint in epoch_checkpoints[:-self.max_checkpoints]:
                checkpoint.unlink()
                print(f"  Removed old checkpoint: {checkpoint.name}")

    def get_best_checkpoint_path(self):
        """
        Get path to best checkpoint.

        Returns:
            Path: Path to best checkpoint file
        """
        return self.checkpoint_dir / 'best.pth'

    def get_latest_checkpoint_path(self):
        """
        Get path to latest checkpoint.

        Returns:
            Path: Path to latest checkpoint file
        """
        return self.checkpoint_dir / 'latest.pth'

    def checkpoint_exists(self, checkpoint_type='latest'):
        """
        Check if a checkpoint exists.

        Args:
            checkpoint_type (str): 'latest' or 'best'. Default: 'latest'

        Returns:
            bool: True if checkpoint exists
        """
        if checkpoint_type == 'latest':
            return self.get_latest_checkpoint_path().exists()
        elif checkpoint_type == 'best':
            return self.get_best_checkpoint_path().exists()
        else:
            return (self.checkpoint_dir / checkpoint_type).exists()
