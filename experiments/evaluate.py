"""
Evaluation script for MURA abnormality detection.

Usage:
    python experiments/evaluate.py --checkpoint checkpoints/best.pth
"""

import sys
import os
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.dataset import get_dataloaders
from models.mura_classifier import MURAClassifier
from evaluation.evaluator import evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MURA abnormality detection model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        choices=['cross_attention', 'concatenation', 'pooling'],
                        help='Type of multi-view fusion (default: cross_attention)')
    parser.add_argument('--save_predictions', type=str, default='results/predictions.csv',
                        help='Path to save predictions CSV (default: results/predictions.csv)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to evaluate on (default: cuda)')

    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("\n" + "="*60)
    print("MURA Abnormality Detection - Evaluation")
    print("="*60)

    # Load data
    print("\nLoading validation data...")
    _, val_loader = get_dataloaders(config)

    # Create model
    print("\nInitializing model...")
    model = MURAClassifier(config, fusion_type=args.fusion_type)
    model.to(args.device)

    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Checkpoint from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Validation AUC: {checkpoint['metrics'].get('auc', 0):.4f}")

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=args.device,
        save_predictions=args.save_predictions
    )

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
