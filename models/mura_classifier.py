"""
Complete MURA classifier combining ViT backbone and multi-view fusion.
End-to-end model for musculoskeletal abnormality detection.
"""

import torch
import torch.nn as nn

from .vit_backbone import ViTBackbone
from .multiview_fusion import CrossAttentionFusion, ConcatenationFusion, PoolingFusion


class MURAClassifier(nn.Module):
    """
    Multi-view Vision Transformer classifier for MURA abnormality detection.

    Architecture:
        1. Shared ViT backbone processes each view independently
        2. Multi-view fusion module combines features from all views
        3. Classification head predicts abnormality probability

    Supports variable-length inputs through attention masking.
    """

    def __init__(self, config, fusion_type='cross_attention'):
        """
        Initialize MURA classifier.

        Args:
            config: Configuration object with model parameters
            fusion_type (str): Type of fusion module ('cross_attention', 'concatenation', or 'pooling')
        """
        super().__init__()

        self.config = config
        self.fusion_type = fusion_type

        # ===== 1. ViT Backbone =====
        self.backbone = ViTBackbone(
            model_name=config.VIT_MODEL,
            freeze_backbone=False,  # Fine-tune all layers
            freeze_layers=6
        )

        # ===== 2. Multi-View Fusion Module =====
        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                hidden_dim=config.HIDDEN_DIM,
                num_queries=config.NUM_QUERY_TOKENS,
                num_heads=config.NUM_ATTENTION_HEADS,
                dropout=config.FUSION_DROPOUT
            )
        elif fusion_type == 'concatenation':
            self.fusion = ConcatenationFusion(
                hidden_dim=config.HIDDEN_DIM,
                max_views=config.MAX_VIEWS,
                dropout=config.FUSION_DROPOUT
            )
        elif fusion_type == 'pooling':
            self.fusion = PoolingFusion(
                hidden_dim=config.HIDDEN_DIM,
                pooling='max'  # Can be 'max', 'avg', or 'both'
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # ===== 3. Classification Head =====
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.Linear(config.HIDDEN_DIM // 2, 1)  # Binary classification (logit)
        )

        # Print model info
        self._print_model_info()

    def forward(self, images, mask):
        """
        Forward pass through the model.

        Args:
            images: [batch, max_views, 3, 224, 224] - Multi-view input images
            mask: [batch, max_views] - Boolean mask (True=valid view, False=padding)

        Returns:
            tuple:
                - logits: [batch, 1] - Abnormality prediction logits
                - attention_weights: [batch, num_queries, max_views] or None - Attention weights (if using cross-attention)
        """
        batch_size, num_views, C, H, W = images.shape

        # ===== Step 1: Extract features from each view using shared ViT =====
        # Reshape to process all views in batch: [batch * num_views, 3, 224, 224]
        images_flat = images.view(batch_size * num_views, C, H, W)

        # Extract CLS tokens from ViT: [batch * num_views, hidden_dim]
        cls_tokens, _ = self.backbone(images_flat)

        # Reshape back to separate views: [batch, num_views, hidden_dim]
        view_features = cls_tokens.view(batch_size, num_views, -1)

        # ===== Step 2: Fuse multi-view features =====
        # fused_features: [batch, hidden_dim]
        # attention_weights: [batch, num_queries, num_views] or None
        fused_features, attention_weights = self.fusion(view_features, mask)

        # ===== Step 3: Classification =====
        # logits: [batch, 1]
        logits = self.classifier(fused_features)

        return logits, attention_weights

    def predict_proba(self, images, mask):
        """
        Get probability predictions (apply sigmoid to logits).

        Args:
            images: [batch, max_views, 3, 224, 224]
            mask: [batch, max_views]

        Returns:
            tuple:
                - probabilities: [batch, 1] - Abnormality probabilities in [0, 1]
                - attention_weights: [batch, num_queries, max_views] or None
        """
        logits, attention_weights = self.forward(images, mask)
        probabilities = torch.sigmoid(logits)
        return probabilities, attention_weights

    def _print_model_info(self):
        """Print model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("MURA Classifier Model Summary")
        print("=" * 60)
        print(f"Fusion Type: {self.fusion_type}")
        print(f"ViT Backbone: {self.config.VIT_MODEL}")
        print(f"Hidden Dimension: {self.config.HIDDEN_DIM}")
        print(f"Max Views: {self.config.MAX_VIEWS}")

        if self.fusion_type == 'cross_attention':
            print(f"Query Tokens: {self.config.NUM_QUERY_TOKENS}")
            print(f"Attention Heads: {self.config.NUM_ATTENTION_HEADS}")

        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print("=" * 60 + "\n")

    def get_num_parameters(self):
        """
        Get parameter counts.

        Returns:
            dict: Dictionary with 'total' and 'trainable' parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def freeze_backbone(self):
        """Freeze ViT backbone parameters (for faster training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Froze ViT backbone parameters")

    def unfreeze_backbone(self):
        """Unfreeze ViT backbone parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Unfroze ViT backbone parameters")


def create_model(config, fusion_type='cross_attention', device='cuda'):
    """
    Factory function to create and initialize MURA classifier.

    Args:
        config: Configuration object
        fusion_type (str): Type of fusion module
        device (str): Device to move model to

    Returns:
        MURAClassifier: Initialized model on specified device
    """
    model = MURAClassifier(config, fusion_type=fusion_type)
    model = model.to(device)

    # Print device info
    if device == 'cuda' and torch.cuda.is_available():
        print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"Model on CPU")

    return model
