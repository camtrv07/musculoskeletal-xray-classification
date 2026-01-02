"""
Main training orchestration for MURA abnormality detection.
Handles training loop, validation, logging, and checkpointing.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .losses import WeightedBCELoss
from .metrics import MetricsCalculator
from .checkpoint_manager import CheckpointManager


class Trainer:
    """
    Training orchestrator for MURA classifier.

    Handles:
        - Training and validation loops
        - Loss computation and optimization
        - Metrics tracking
        - Checkpointing
        - TensorBoard logging
    """

    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        """
        Initialize trainer.

        Args:
            model (nn.Module): MURA classifier model
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            config: Configuration object
            device (str): Device to train on. Default: 'cuda'
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(self.device)

        # Loss function
        self.criterion = WeightedBCELoss(pos_weight=config.POS_WEIGHT)

        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler: Cosine Annealing with Warm Restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.SCHEDULER_T0,
            T_mult=config.SCHEDULER_T_MULT
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.CHECKPOINT_DIR,
            max_checkpoints=config.MAX_CHECKPOINTS
        )

        # TensorBoard writer
        self.writer = SummaryWriter(config.LOG_DIR)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_auc = 0.0

        print(f"\nTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Optimizer: AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
        print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0={config.SCHEDULER_T0})")
        print(f"  Loss: WeightedBCE (pos_weight={config.POS_WEIGHT})")

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch (int): Current epoch number

        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        metrics_calc = MetricsCalculator()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]', leave=False)

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)  # [batch, max_views, 3, 224, 224]
            mask = batch['mask'].to(self.device)      # [batch, max_views]
            labels = batch['label'].to(self.device)    # [batch]

            # Forward pass
            logits, _ = self.model(images, mask)  # [batch, 1]

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.GRADIENT_CLIP
            )

            # Update weights
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            metrics_calc.update(logits.detach(), labels.detach())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Log to TensorBoard
            if batch_idx % self.config.LOG_INTERVAL == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # Compute epoch metrics
        metrics = metrics_calc.compute()
        metrics['loss'] = total_loss / num_batches

        return metrics

    def validate(self, epoch):
        """
        Validate the model.

        Args:
            epoch (int): Current epoch number

        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        metrics_calc = MetricsCalculator()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        # Progress bar
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]', leave=False)

        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                images = batch['images'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                logits, _ = self.model(images, mask)

                # Compute loss
                loss = self.criterion(logits, labels)

                # Update metrics
                total_loss += loss.item()
                metrics_calc.update(logits, labels)

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute epoch metrics
        metrics = metrics_calc.compute()
        metrics['loss'] = total_loss / num_batches

        return metrics

    def train(self, num_epochs, resume_from=None):
        """
        Main training loop.

        Args:
            num_epochs (int): Total number of epochs to train
            resume_from (str, optional): Path to checkpoint to resume from
        """
        start_epoch = 0

        # Resume from checkpoint if specified
        if resume_from:
            start_epoch, _ = self.checkpoint_manager.load_checkpoint(
                self.model, self.optimizer, self.scheduler,
                checkpoint_path=resume_from, device=self.device
            )
            start_epoch += 1  # Start from next epoch
            self.best_auc = self.checkpoint_manager.best_metric

        print(f"\n{'='*60}")
        print(f"Starting training from epoch {start_epoch} to {num_epochs}")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print("-" * 60)

            # Train for one epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Step scheduler
            self.scheduler.step()

            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Print epoch summary
            self._print_epoch_summary(epoch, train_metrics, val_metrics)

            # Check if this is the best model
            is_best = val_metrics['auc'] > self.best_auc
            if is_best:
                self.best_auc = val_metrics['auc']
                print(f"  New best AUC: {self.best_auc:.4f}")

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, val_metrics, is_best=is_best
            )

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation AUC: {self.best_auc:.4f}")
        print(f"{'='*60}\n")

        # Close TensorBoard writer
        self.writer.close()

    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Log metrics to TensorBoard."""
        # Training metrics
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Train/{key}', value, epoch)

        # Validation metrics
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Val/{key}', value, epoch)

        # Learning rate
        self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

    def _print_epoch_summary(self, epoch, train_metrics, val_metrics):
        """Print formatted epoch summary."""
        print(f"\n  Train - Loss: {train_metrics['loss']:.4f} | "
              f"AUC: {train_metrics['auc']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f}")

        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"Sens: {val_metrics['sensitivity']:.4f} | "
              f"Spec: {val_metrics['specificity']:.4f}")

    def get_model(self):
        """Return the model."""
        return self.model

    def get_best_auc(self):
        """Return the best validation AUC achieved."""
        return self.best_auc
