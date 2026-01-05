"""
Simplified MURA classifier for CPU training.
Uses lightweight CNN instead of ViT for faster training.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Lightweight CNN backbone using pre-trained ResNet18."""

    def __init__(self, pretrained=True):
        super().__init__()
        # Use ResNet18 (much smaller than ViT: 11M vs 86M parameters)
        resnet = models.resnet18(pretrained=pretrained)

        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512  # ResNet18 output dimension

        print(f"Loaded ResNet18 backbone (pretrained={pretrained})")
        print(f"  Output dimension: {self.feature_dim}")

    def forward(self, x):
        """
        Args:
            x: [batch, 3, 224, 224]
        Returns:
            features: [batch, 512]
        """
        features = self.features(x)
        features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        return features


class SimpleFusion(nn.Module):
    """Simple average pooling fusion for multi-view images."""

    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        print(f"Initialized SimpleFusion (average pooling)")

    def forward(self, view_features, mask):
        """
        Average features across valid views.

        Args:
            view_features: [batch, num_views, feature_dim]
            mask: [batch, num_views] - boolean mask
        Returns:
            fused_features: [batch, feature_dim]
        """
        # Mask out invalid views
        mask_expanded = mask.unsqueeze(-1).float()  # [batch, num_views, 1]
        masked_features = view_features * mask_expanded

        # Average across views (only counting valid ones)
        num_valid_views = mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
        fused_features = masked_features.sum(dim=1) / (num_valid_views + 1e-8)

        return fused_features


class SimpleClassifier(nn.Module):
    """
    Simplified multi-view classifier for CPU training.

    Much faster than the full ViT model:
    - ResNet18 instead of ViT (11M vs 86M parameters)
    - Average pooling instead of cross-attention
    - ~10x faster training on CPU
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        # Backbone: ResNet18
        self.backbone = SimpleCNN(pretrained=pretrained)
        feature_dim = self.backbone.feature_dim

        # Fusion: Simple averaging
        self.fusion = SimpleFusion(feature_dim=feature_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "="*60)
        print("Simple MURA Classifier")
        print("="*60)
        print(f"Backbone: ResNet18")
        print(f"Fusion: Average Pooling")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*60 + "\n")

    def forward(self, images, mask):
        """
        Args:
            images: [batch, max_views, 3, 224, 224]
            mask: [batch, max_views]
        Returns:
            logits: [batch, 1]
        """
        batch_size, num_views, C, H, W = images.shape

        # Process all views through shared backbone
        images_flat = images.view(batch_size * num_views, C, H, W)
        features = self.backbone(images_flat)  # [batch*views, 512]

        # Reshape to separate views
        view_features = features.view(batch_size, num_views, -1)

        # Fuse multi-view features
        fused_features = self.fusion(view_features, mask)

        # Classify
        logits = self.classifier(fused_features)

        return logits, None  # No attention weights in simple version


def create_simple_model(device='cpu', pretrained=True):
    """
    Create simplified model for CPU training.

    Args:
        device: Device to move model to
        pretrained: Use pre-trained ResNet18 weights

    Returns:
        SimpleClassifier on specified device
    """
    model = SimpleClassifier(pretrained=pretrained)
    model = model.to(device)

    print(f"Model moved to: {device}")
    return model
