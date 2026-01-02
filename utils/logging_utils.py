"""
Logging utilities for TensorBoard and console output.
"""

import os
from torch.utils.tensorboard import SummaryWriter


def get_tensorboard_writer(log_dir, experiment_name="default"):
    """
    Create a TensorBoard SummaryWriter.

    Args:
        log_dir (str): Base directory for logs
        experiment_name (str): Name of the experiment

    Returns:
        SummaryWriter: TensorBoard writer instance
    """
    log_path = os.path.join(log_dir, experiment_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    print(f"TensorBoard logs will be saved to: {log_path}")
    print(f"To view logs, run: tensorboard --logdir={log_path}")
    return writer


def log_metrics(writer, metrics, epoch, prefix=""):
    """
    Log metrics to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer
        metrics (dict): Dictionary of metric name -> value
        epoch (int): Current epoch
        prefix (str): Prefix for metric names (e.g., "Train", "Val")
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, epoch)


def print_metrics(metrics, prefix=""):
    """
    Print metrics to console in a formatted way.

    Args:
        metrics (dict): Dictionary of metric name -> value
        prefix (str): Prefix for display (e.g., "Train", "Val")
    """
    metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                             for k, v in metrics.items()])
    print(f"{prefix} - {metric_str}" if prefix else metric_str)
