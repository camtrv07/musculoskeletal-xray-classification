"""
MURA Dataset for multi-view musculoskeletal abnormality detection.
Handles variable-length studies with padding and masking.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

from .utils import get_study_images
from .transforms import get_train_transforms, get_val_transforms


class MURADataset(Dataset):
    """
    Multi-view MURA dataset.

    Each sample is a study containing 1-4 X-ray images.
    Images are padded to max_views with zero tensors.
    A boolean mask indicates which views are valid vs. padding.

    Returns:
        dict with keys:
            - images: [max_views, 3, image_size, image_size] tensor
            - mask: [max_views] boolean tensor (True=valid, False=padding)
            - label: scalar tensor (0=normal, 1=abnormal)
            - study_path: str (for debugging/visualization)
    """

    def __init__(self, csv_path, data_root, transform=None, max_views=4):
        """
        Initialize MURA dataset.

        Args:
            csv_path (str): Path to CSV file with study paths and labels
            data_root (str): Root directory containing MURA-v1.1
            transform (callable, optional): Transform to apply to each image
            max_views (int): Maximum number of views per study (for padding). Default: 4
        """
        # Load CSV
        self.df = pd.read_csv(csv_path, header=None, names=['study_path', 'label'])
        self.data_root = data_root
        self.transform = transform
        self.max_views = max_views

        print(f"Initialized MURADataset:")
        print(f"  Total studies: {len(self.df)}")
        print(f"  Positive (abnormal): {(self.df['label'] == 1).sum()}")
        print(f"  Negative (normal): {(self.df['label'] == 0).sum()}")

    def __len__(self):
        """Return number of studies in dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single study with all its images.

        Args:
            idx (int): Index of study

        Returns:
            dict: Dictionary containing images, mask, label, and study_path
        """
        # Get study info
        study_path = self.df.iloc[idx]['study_path']
        label = self.df.iloc[idx]['label']

        # Get all image paths for this study
        image_paths = get_study_images(study_path, self.data_root)

        # Load and transform images
        images = []
        for img_path in image_paths[:self.max_views]:  # Take at most max_views images
            try:
                # Load image as grayscale
                img = Image.open(img_path).convert('L')

                # Apply transforms
                if self.transform:
                    img = self.transform(img)

                images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Use a zero tensor as fallback
                if self.transform:
                    # Create a dummy grayscale image
                    dummy_img = Image.new('L', (224, 224), 0)
                    img = self.transform(dummy_img)
                    images.append(img)

        # Create mask (True for valid images, False for padding)
        num_views = len(images)
        mask = torch.zeros(self.max_views, dtype=torch.bool)
        mask[:num_views] = True

        # Pad with zero tensors if needed
        if num_views > 0:
            # Use the shape of the first image for padding
            img_shape = images[0].shape
            while len(images) < self.max_views:
                images.append(torch.zeros_like(images[0]))
        else:
            # Edge case: no valid images
            print(f"Warning: No valid images found for study {study_path}")
            # Create dummy zero tensors
            dummy_img = torch.zeros(3, 224, 224)
            images = [dummy_img] * self.max_views

        # Stack images
        images = torch.stack(images)  # [max_views, 3, H, W]

        # Convert label to float tensor
        label = torch.tensor(label, dtype=torch.float32)

        return {
            'images': images,
            'mask': mask,
            'label': label,
            'study_path': study_path
        }


def get_dataloaders(config):
    """
    Create train and validation dataloaders.

    Args:
        config: Configuration object with dataset parameters

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get transforms
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

    # Create datasets
    train_dataset = MURADataset(
        csv_path=config.TRAIN_CSV,
        data_root=config.DATA_ROOT,
        transform=train_transform,
        max_views=config.MAX_VIEWS
    )

    val_dataset = MURADataset(
        csv_path=config.VALID_CSV,
        data_root=config.DATA_ROOT,
        transform=val_transform,
        max_views=config.MAX_VIEWS
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} studies, {len(train_loader)} batches")
    print(f"  Valid: {len(val_dataset)} studies, {len(val_loader)} batches")

    return train_loader, val_loader
