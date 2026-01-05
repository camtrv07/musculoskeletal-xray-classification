"""
Simplified dataset for faster CPU training.
Uses a small subset of MURA for quick experiments.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image

from .utils import get_study_images
from .transforms import get_train_transforms, get_val_transforms


class SimpleMURADataset(Dataset):
    """
    Simplified MURA dataset for quick training.
    Same as MURADataset but optimized for speed.
    """

    def __init__(self, csv_path, data_root, transform=None, max_views=4, limit=None):
        """
        Args:
            csv_path: Path to CSV file
            data_root: Root directory
            transform: Image transforms
            max_views: Max views per study
            limit: Limit number of studies (for quick training)
        """
        self.df = pd.read_csv(csv_path, header=None, names=['study_path', 'label'])

        # Limit dataset size if specified
        if limit and limit < len(self.df):
            # Stratified sampling to keep class balance
            pos_samples = self.df[self.df['label'] == 1].sample(n=limit // 2, random_state=42)
            neg_samples = self.df[self.df['label'] == 0].sample(n=limit // 2, random_state=42)
            self.df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42)
            print(f"  Using subset: {len(self.df)} studies (limited from full dataset)")

        self.data_root = data_root
        self.transform = transform
        self.max_views = max_views

        print(f"Dataset loaded:")
        print(f"  Total studies: {len(self.df)}")
        print(f"  Positive: {(self.df['label'] == 1).sum()}")
        print(f"  Negative: {(self.df['label'] == 0).sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx]['study_path']
        label = self.df.iloc[idx]['label']

        # Get all images for this study
        image_paths = get_study_images(study_path, self.data_root)

        # Load and transform images
        images = []
        for img_path in image_paths[:self.max_views]:
            try:
                img = Image.open(img_path).convert('L')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except:
                pass

        # Create mask
        num_views = len(images)
        mask = torch.zeros(self.max_views, dtype=torch.bool)
        mask[:num_views] = True

        # Pad with zeros
        if num_views > 0:
            while len(images) < self.max_views:
                images.append(torch.zeros_like(images[0]))
        else:
            # Fallback for missing images
            dummy = torch.zeros(3, 224, 224)
            images = [dummy] * self.max_views

        images = torch.stack(images)
        label = torch.tensor(label, dtype=torch.float32)

        return {
            'images': images,
            'mask': mask,
            'label': label,
            'study_path': study_path
        }


def get_simple_dataloaders(config, train_limit=1000, val_limit=200):
    """
    Create dataloaders with limited dataset size for faster training.

    Args:
        config: Configuration object
        train_limit: Max training samples (default: 1000, ~7% of dataset)
        val_limit: Max validation samples (default: 200, ~17% of validation)

    Returns:
        train_loader, val_loader
    """
    print("\nCreating simplified dataloaders...")
    print(f"Training subset: {train_limit} studies")
    print(f"Validation subset: {val_limit} studies")

    # Transforms
    train_transform = get_train_transforms(
        image_size=config.IMAGE_SIZE,
        mean=config.MEAN,
        std=config.STD
    )

    val_transform = get_val_transforms(
        image_size=config.IMAGE_SIZE,
        mean=config.MEAN,
        std=config.STD
    )

    # Datasets with limited size
    train_dataset = SimpleMURADataset(
        csv_path=config.TRAIN_CSV,
        data_root=config.DATA_ROOT,
        transform=train_transform,
        max_views=config.MAX_VIEWS,
        limit=train_limit
    )

    val_dataset = SimpleMURADataset(
        csv_path=config.VALID_CSV,
        data_root=config.DATA_ROOT,
        transform=val_transform,
        max_views=config.MAX_VIEWS,
        limit=val_limit
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # No multiprocessing for CPU
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_dataset)} studies, {len(train_loader)} batches")
    print(f"  Valid: {len(val_dataset)} studies, {len(val_loader)} batches")

    return train_loader, val_loader
