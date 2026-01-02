"""
Utility functions for reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility across different libraries.

    Args:
        seed (int): Random seed value. Default: 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} for reproducibility")
