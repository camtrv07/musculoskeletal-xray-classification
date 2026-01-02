"""
Evaluation metrics for MURA abnormality detection.
Computes accuracy, AUC-ROC, sensitivity, and specificity.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import torch


class MetricsCalculator:
    """
    Accumulates predictions and computes evaluation metrics.

    Metrics:
        - Accuracy: (TP + TN) / Total
        - AUC-ROC: Area Under the Receiver Operating Characteristic curve
        - Sensitivity (Recall): TP / (TP + FN) - ability to detect abnormal cases
        - Specificity: TN / (TN + FP) - ability to correctly identify normal cases
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()

    def reset(self):
        """Reset all stored predictions and targets."""
        self.predictions = []
        self.probabilities = []
        self.targets = []

    def update(self, logits, targets):
        """
        Add batch predictions and targets.

        Args:
            logits: [batch, 1] - Model logits
            targets: [batch] - Ground truth labels
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits).squeeze(1).cpu().detach().numpy()

        # Convert probabilities to binary predictions (threshold=0.5)
        preds = (probs > 0.5).astype(int)

        # Convert targets to numpy
        targets_np = targets.cpu().numpy()

        # Store
        self.predictions.extend(preds)
        self.probabilities.extend(probs)
        self.targets.extend(targets_np)

    def compute(self):
        """
        Compute all metrics from accumulated predictions.

        Returns:
            dict: Dictionary containing all metrics
        """
        if len(self.targets) == 0:
            return {
                'accuracy': 0.0,
                'auc': 0.0,
                'sensitivity': 0.0,
                'specificity': 0.0,
                'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0
            }

        preds = np.array(self.predictions)
        probs = np.array(self.probabilities)
        targets = np.array(self.targets)

        # Accuracy
        accuracy = accuracy_score(targets, preds)

        # AUC-ROC
        try:
            auc = roc_auc_score(targets, probs)
        except ValueError:
            # Handle case where all targets are same class
            auc = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

        # Sensitivity (Recall for positive class)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity (Recall for negative class)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # F1 score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'num_samples': len(targets)
        }

    def get_optimal_threshold(self):
        """
        Find optimal classification threshold using Youden's J statistic.

        Returns:
            float: Optimal threshold value
        """
        if len(self.targets) == 0:
            return 0.5

        probs = np.array(self.probabilities)
        targets = np.array(self.targets)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(targets, probs)

        # Youden's J statistic: sensitivity + specificity - 1
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold

    def print_metrics(self, prefix=""):
        """
        Print metrics in a formatted way.

        Args:
            prefix (str): Prefix for display (e.g., "Train", "Val")
        """
        metrics = self.compute()

        print(f"\n{prefix} Metrics:" if prefix else "Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC:     {metrics['auc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f} (TP/{metrics['tp']+metrics['fn']} = {metrics['tp']}/{metrics['tp']+metrics['fn']})")
        print(f"  Specificity: {metrics['specificity']:.4f} (TN/{metrics['tn']+metrics['fp']} = {metrics['tn']}/{metrics['tn']+metrics['fp']})")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
