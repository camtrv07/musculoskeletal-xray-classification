"""
Vision Transformer (ViT) backbone for feature extraction.
Uses pre-trained ViT from HuggingFace Transformers.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ViTBackbone(nn.Module):
    """
    Pre-trained Vision Transformer feature extractor.

    Loads a pre-trained ViT model and extracts CLS token representations.
    Optionally freezes early layers for faster training.
    """

    def __init__(self, model_name='google/vit-base-patch16-224', freeze_backbone=False, freeze_layers=6):
        """
        Initialize ViT backbone.

        Args:
            model_name (str): HuggingFace model name. Default: 'google/vit-base-patch16-224'
            freeze_backbone (bool): Whether to freeze early layers. Default: False
            freeze_layers (int): Number of early encoder layers to freeze if freeze_backbone=True. Default: 6
        """
        super().__init__()

        print(f"Loading pre-trained ViT model: {model_name}")
        self.vit = ViTModel.from_pretrained(model_name)

        # Optionally freeze early layers for faster training
        if freeze_backbone:
            print(f"Freezing embeddings and first {freeze_layers} encoder layers")

            # Freeze embeddings
            for param in self.vit.embeddings.parameters():
                param.requires_grad = False

            # Freeze first N encoder layers
            for i, layer in enumerate(self.vit.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            # Count frozen vs trainable parameters
            frozen_params = sum(p.numel() for p in self.vit.parameters() if not p.requires_grad)
            trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            print(f"  Frozen parameters: {frozen_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
        else:
            trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            print(f"  All {trainable_params:,} parameters trainable")

    def forward(self, x):
        """
        Forward pass through ViT.

        Args:
            x: [batch, 3, 224, 224] - Input images

        Returns:
            tuple:
                - cls_token: [batch, hidden_dim] - CLS token representation
                - patch_tokens: [batch, num_patches, hidden_dim] - Patch token representations
        """
        # Pass through ViT
        outputs = self.vit(pixel_values=x)

        # Extract hidden states
        # last_hidden_state: [batch, seq_len, hidden_dim]
        # seq_len = 1 (CLS) + num_patches (196 for 14x14 patches)
        last_hidden_state = outputs.last_hidden_state

        # Split CLS token and patch tokens
        cls_token = last_hidden_state[:, 0]  # [batch, hidden_dim]
        patch_tokens = last_hidden_state[:, 1:]  # [batch, num_patches, hidden_dim]

        return cls_token, patch_tokens

    def get_num_parameters(self):
        """Get total number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
