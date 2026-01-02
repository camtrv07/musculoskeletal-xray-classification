"""
Model evaluation on validation/test set.
Computes final metrics and saves predictions.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm

from training.metrics import MetricsCalculator


def evaluate_model(model, dataloader, device='cuda', save_predictions=None):
    """
    Evaluate model on a dataset.

    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Data to evaluate on
        device (str): Device to run evaluation on. Default: 'cuda'
        save_predictions (str, optional): Path to save predictions CSV

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    model.to(device)

    metrics_calc = MetricsCalculator()

    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_study_paths = []
    all_attention_weights = []

    print(f"\nEvaluating model on {len(dataloader.dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move data to device
            images = batch['images'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)
            study_paths = batch['study_path']

            # Forward pass
            logits, attention_weights = model(images, mask)

            # Get probabilities
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).int()

            # Update metrics
            metrics_calc.update(logits, labels)

            # Store results
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_study_paths.extend(study_paths)

            if attention_weights is not None:
                all_attention_weights.extend(attention_weights.cpu().numpy())

    # Compute final metrics
    metrics = metrics_calc.compute()

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total Samples: {metrics['num_samples']}")
    print(f"AUC-ROC:      {metrics['auc']:.4f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']:4d}  |  FP: {metrics['fp']:4d}")
    print(f"  FN: {metrics['fn']:4d}  |  TN: {metrics['tn']:4d}")
    print("="*60)

    # Save predictions if requested
    if save_predictions:
        results_df = pd.DataFrame({
            'study_path': all_study_paths,
            'true_label': all_labels,
            'predicted_label': all_predictions,
            'probability': all_probabilities
        })
        results_df.to_csv(save_predictions, index=False)
        print(f"\nPredictions saved to: {save_predictions}")

    return metrics
