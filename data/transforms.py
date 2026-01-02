"""
Image transformations and augmentations for medical X-ray images.
Conservative augmentations to preserve medical validity.
"""

import torchvision.transforms as T


def get_train_transforms(image_size=224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Get training transformations with data augmentation.

    Args:
        image_size (int): Target image size. Default: 224
        mean (list): Normalization mean for each channel. Default: [0.5, 0.5, 0.5]
        std (list): Normalization std for each channel. Default: [0.5, 0.5, 0.5]

    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    transforms = T.Compose([
        # Convert grayscale to 3-channel RGB for ViT
        T.Grayscale(num_output_channels=3),

        # Random augmentations (conservative for medical images)
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),

        # Convert to tensor
        T.ToTensor(),

        # Normalize
        T.Normalize(mean=mean, std=std)
    ])

    return transforms


def get_val_transforms(image_size=224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Get validation/test transformations (no augmentation).

    Args:
        image_size (int): Target image size. Default: 224
        mean (list): Normalization mean for each channel. Default: [0.5, 0.5, 0.5]
        std (list): Normalization std for each channel. Default: [0.5, 0.5, 0.5]

    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    transforms = T.Compose([
        # Convert grayscale to 3-channel RGB for ViT
        T.Grayscale(num_output_channels=3),

        # Resize and center crop (no random augmentation)
        T.Resize(256),
        T.CenterCrop(image_size),

        # Convert to tensor
        T.ToTensor(),

        # Normalize
        T.Normalize(mean=mean, std=std)
    ])

    return transforms


def get_test_transforms(image_size=224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Get test transformations (same as validation).

    Args:
        image_size (int): Target image size. Default: 224
        mean (list): Normalization mean for each channel. Default: [0.5, 0.5, 0.5]
        std (list): Normalization std for each channel. Default: [0.5, 0.5, 0.5]

    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return get_val_transforms(image_size, mean, std)
