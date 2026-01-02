"""
Visualization script for Grad-CAM.

Usage:
    python experiments/visualize.py --checkpoint checkpoints/best.pth --num_samples 10
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
from evaluation.visualization import visualize_study


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize Grad-CAM for MURA model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        choices=['cross_attention', 'concatenation', 'pooling'],
                        help='Type of multi-view fusion (default: cross_attention)')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to visualize (default: 20)')
    parser.add_argument('--output_dir', type=str, default='results/gradcam',
                        help='Directory to save visualizations (default: results/gradcam)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on (default: cuda)')

    return parser.parse_args()


def main():
    """Main visualization function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("MURA Abnormality Detection - Grad-CAM Visualization")
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

    # Generate visualizations
    print(f"\nGenerating Grad-CAM visualizations for {args.num_samples} samples...")
    print(f"Saving to: {args.output_dir}")

    for i, batch in enumerate(val_loader):
        if i >= args.num_samples:
            break

        # Take first sample from batch
        images = batch['images'][:1]  # [1, max_views, 3, 224, 224]
        mask = batch['mask'][:1]       # [1, max_views]
        labels = batch['label'][:1]    # [1]
        study_path = batch['study_path'][0]

        # Generate safe filename from study path
        filename = f"sample_{i:03d}.png"
        save_path = os.path.join(args.output_dir, filename)

        # Visualize
        visualize_study(
            model=model,
            images=images,
            mask=mask,
            labels=labels,
            study_path=study_path,
            save_path=save_path,
            device=args.device
        )

    print(f"\nVisualization complete! Saved {min(i+1, args.num_samples)} images to {args.output_dir}")


if __name__ == '__main__':
    main()
