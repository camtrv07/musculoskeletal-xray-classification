"""
Multi-view fusion modules for combining features from multiple radiographic projections.
Implements cross-attention and concatenation-based fusion strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module for multi-view integration.

    Uses learnable query tokens that attend to features from multiple views.
    Mimics how radiologists actively compare multiple projections.
    Supports variable-length inputs through attention masking.
    """

    def __init__(self, hidden_dim=768, num_queries=4, num_heads=8, dropout=0.1):
        """
        Initialize cross-attention fusion module.

        Args:
            hidden_dim (int): Hidden dimension size. Default: 768 (ViT-Base)
            num_queries (int): Number of learnable query tokens. Default: 4
            num_heads (int): Number of attention heads. Default: 8
            dropout (float): Dropout rate. Default: 0.1
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Learnable query tokens (randomly initialized, learned during training)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        # Multi-head cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier tensor manipulation
        )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Aggregation: combine query outputs into single feature vector
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * num_queries, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        print(f"Initialized CrossAttentionFusion:")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num queries: {num_queries}")
        print(f"  Num heads: {num_heads}")
        print(f"  Dropout: {dropout}")

    def forward(self, view_features, mask):
        """
        Fuse features from multiple views using cross-attention.

        Args:
            view_features: [batch, num_views, hidden_dim] - Features from each view
            mask: [batch, num_views] - Boolean mask (True=valid view, False=padding)

        Returns:
            tuple:
                - fused_features: [batch, hidden_dim] - Aggregated multi-view features
                - attention_weights: [batch, num_queries, num_views] - Attention weights
        """
        batch_size, num_views, hidden_dim = view_features.shape

        # Expand query tokens for the batch
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [batch, num_queries, hidden_dim]

        # Create attention mask for cross-attention
        # PyTorch's MultiheadAttention uses key_padding_mask where:
        #   True = ignore this position (padding)
        #   False = attend to this position (valid)
        # Our mask is: True = valid, False = padding
        # So we need to invert it
        key_padding_mask = ~mask  # [batch, num_views]

        # Cross-attention: queries attend to view features
        # query: [batch, num_queries, hidden_dim]
        # key/value: [batch, num_views, hidden_dim]
        attended_features, attention_weights = self.cross_attention(
            query=queries,
            key=view_features,
            value=view_features,
            key_padding_mask=key_padding_mask,  # Mask out padded views
            need_weights=True,
            average_attn_weights=True  # Average across heads for interpretability
        )

        # attended_features: [batch, num_queries, hidden_dim]
        # attention_weights: [batch, num_queries, num_views]

        # Post-attention normalization
        attended_features = self.norm(attended_features)

        # Aggregate query outputs into single feature vector
        # Flatten queries: [batch, num_queries * hidden_dim]
        attended_features_flat = attended_features.flatten(1)

        # Project to hidden_dim: [batch, hidden_dim]
        fused_features = self.aggregation(attended_features_flat)

        return fused_features, attention_weights


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation-based fusion module.

    Concatenates features from all views and projects to single feature vector.
    Simpler and faster than cross-attention, but less interpretable.
    """

    def __init__(self, hidden_dim=768, max_views=4, dropout=0.1):
        """
        Initialize concatenation fusion module.

        Args:
            hidden_dim (int): Hidden dimension size. Default: 768
            max_views (int): Maximum number of views. Default: 4
            dropout (float): Dropout rate. Default: 0.1
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_views = max_views

        # Projection layer: concatenated features -> single feature vector
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * max_views, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        print(f"Initialized ConcatenationFusion:")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Max views: {max_views}")
        print(f"  Dropout: {dropout}")

    def forward(self, view_features, mask):
        """
        Fuse features by concatenation and projection.

        Args:
            view_features: [batch, num_views, hidden_dim] - Features from each view
            mask: [batch, num_views] - Boolean mask (True=valid view, False=padding)

        Returns:
            tuple:
                - fused_features: [batch, hidden_dim] - Fused feature vector
                - None: No attention weights (for consistency with CrossAttentionFusion)
        """
        batch_size, num_views, hidden_dim = view_features.shape

        # Zero out features from padded views
        # Expand mask: [batch, num_views, 1]
        mask_expanded = mask.unsqueeze(-1).float()  # Convert bool to float
        view_features_masked = view_features * mask_expanded

        # Flatten all views: [batch, num_views * hidden_dim]
        concatenated = view_features_masked.flatten(1)

        # Project to hidden_dim: [batch, hidden_dim]
        fused_features = self.projection(concatenated)

        return fused_features, None  # No attention weights


class PoolingFusion(nn.Module):
    """
    Pooling-based fusion module.

    Uses max/average pooling across views. Simplest approach.
    """

    def __init__(self, hidden_dim=768, pooling='max'):
        """
        Initialize pooling fusion module.

        Args:
            hidden_dim (int): Hidden dimension size. Default: 768
            pooling (str): Pooling type ('max', 'avg', or 'both'). Default: 'max'
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooling = pooling

        # Optional projection after pooling
        if pooling == 'both':
            # Concatenate max and avg pooling
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            )
        else:
            self.projection = nn.Identity()

        print(f"Initialized PoolingFusion with {pooling} pooling")

    def forward(self, view_features, mask):
        """
        Fuse features using pooling.

        Args:
            view_features: [batch, num_views, hidden_dim]
            mask: [batch, num_views]

        Returns:
            tuple:
                - fused_features: [batch, hidden_dim]
                - None: No attention weights
        """
        # Zero out padded views
        mask_expanded = mask.unsqueeze(-1).float()
        view_features_masked = view_features * mask_expanded

        if self.pooling == 'max':
            # Max pooling across views
            # Replace -inf with very negative number for masked positions
            view_features_masked = view_features_masked + (mask_expanded - 1) * 1e9
            fused_features, _ = torch.max(view_features_masked, dim=1)

        elif self.pooling == 'avg':
            # Average pooling across views (excluding padding)
            sum_features = torch.sum(view_features_masked, dim=1)  # [batch, hidden_dim]
            num_valid_views = mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
            fused_features = sum_features / (num_valid_views + 1e-8)  # Avoid div by zero

        elif self.pooling == 'both':
            # Combine max and avg pooling
            view_features_for_max = view_features_masked + (mask_expanded - 1) * 1e9
            max_features, _ = torch.max(view_features_for_max, dim=1)

            sum_features = torch.sum(view_features_masked, dim=1)
            num_valid_views = mask.sum(dim=1, keepdim=True).float()
            avg_features = sum_features / (num_valid_views + 1e-8)

            # Concatenate and project
            fused_features = torch.cat([max_features, avg_features], dim=1)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        fused_features = self.projection(fused_features)

        return fused_features, None
