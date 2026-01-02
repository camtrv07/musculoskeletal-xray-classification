"""
Grad-CAM visualization for Vision Transformer.
Generates saliency maps showing which regions the model focuses on.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


class GradCAMViT:
    """
    Grad-CAM for Vision Transformer models.

    Hooks into the last encoder layer to compute gradient-weighted class activation maps.
    """

    def __init__(self, model):
        """
        Initialize Grad-CAM.

        Args:
            model (nn.Module): MURA classifier model
        """
        self.model = model
        self.gradients = None
        self.activations = None

        # Register hooks on the last ViT encoder layer
        # The path is: model.backbone.vit.encoder.layer[-1]
        target_layer = self.model.backbone.vit.encoder.layer[-1]

        # Forward hook to capture activations
        target_layer.register_forward_hook(self._save_activation)

        # Backward hook to capture gradients
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        # output is a tuple, first element is the hidden states
        self.activations = output[0].detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, image, mask, target_class=1):
        """
        Generate Grad-CAM for a single image.

        Args:
            image: [1, 1, 3, 224, 224] - Single view image (batch=1, views=1)
            mask: [1, 1] - Mask for the image
            target_class (int): Target class (1 for abnormal, 0 for normal). Default: 1

        Returns:
            np.ndarray: Heatmap of shape [224, 224]
        """
        self.model.eval()

        # Forward pass
        logits, _ = self.model(image, mask)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        if target_class == 1:
            target = logits[0, 0]  # Maximize abnormal logit
        else:
            target = -logits[0, 0]  # Maximize normal (minimize abnormal)

        target.backward()

        # Get gradients and activations
        # gradients: [1, seq_len, hidden_dim]
        # activations: [1, seq_len, hidden_dim]
        gradients = self.gradients[0].cpu().numpy()  # [seq_len, hidden_dim]
        activations = self.activations[0].cpu().numpy()  # [seq_len, hidden_dim]

        # Compute weights by global average pooling of gradients
        weights = np.mean(gradients, axis=0)  # [hidden_dim]

        # Weighted combination of activation maps
        cam = np.sum(weights * activations.T, axis=0)  # [seq_len]

        # Remove CLS token (first token)
        cam = cam[1:]  # [196] for 14x14 patches

        # Reshape to spatial grid (14x14 for patch16 on 224x224 image)
        grid_size = int(np.sqrt(len(cam)))
        cam = cam.reshape(grid_size, grid_size)

        # Apply ReLU (only positive contributions)
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size (224x224)
        cam = cv2.resize(cam, (224, 224))

        return cam


def visualize_study(model, images, mask, labels, study_path, save_path, device='cuda'):
    """
    Visualize Grad-CAM for all views in a study.

    Args:
        model: Trained model
        images: [1, max_views, 3, 224, 224] - Study images
        mask: [1, max_views] - View mask
        labels: [1] - True label
        study_path: str - Study identifier
        save_path: str - Path to save visualization
        device: str - Device to run on
    """
    # Move to device
    images = images.to(device)
    mask = mask.to(device)

    # Get number of valid views
    num_views = mask.sum().item()

    # Get model prediction
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(images, mask)
        prob = torch.sigmoid(logits).item()
        pred = int(prob > 0.5)

    # Create Grad-CAM generator
    grad_cam = GradCAMViT(model)

    # Create figure
    fig, axes = plt.subplots(2, num_views, figsize=(4 * num_views, 8))
    if num_views == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_views):
        # Extract single view
        single_view = images[:, i:i+1, :, :, :]  # [1, 1, 3, 224, 224]
        single_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

        # Get original image
        img = images[0, i].cpu().permute(1, 2, 0).numpy()
        # Denormalize
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)

        # Generate Grad-CAM
        cam = grad_cam.generate_cam(single_view, single_mask, target_class=pred)

        # Plot original image
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'View {i+1}')
        axes[0, i].axis('off')

        # Plot Grad-CAM overlay
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].imshow(cam, cmap='jet', alpha=0.4)
        axes[1, i].set_title('Grad-CAM')
        axes[1, i].axis('off')

    # Add title with prediction info
    label_str = "Abnormal" if labels.item() == 1 else "Normal"
    pred_str = "Abnormal" if pred == 1 else "Normal"
    title = f"Study: {study_path}\nTrue: {label_str} | Predicted: {pred_str} (p={prob:.3f})"
    fig.suptitle(title, fontsize=12)

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to: {save_path}")
