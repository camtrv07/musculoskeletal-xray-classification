"""
Data loading utilities for MURA dataset.
"""

import os
import glob
import pandas as pd


def load_study_paths(csv_path):
    """
    Load study paths and labels from CSV file.

    Args:
        csv_path (str): Path to CSV file (train_labeled_studies.csv or valid_labeled_studies.csv)

    Returns:
        pd.DataFrame: DataFrame with columns ['study_path', 'label']
    """
    df = pd.read_csv(csv_path, header=None, names=['study_path', 'label'])
    print(f"Loaded {len(df)} studies from {csv_path}")
    print(f"  Positive (abnormal): {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"  Negative (normal): {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
    return df


def get_study_images(study_path, data_root=""):
    """
    Get all image paths for a given study.

    Args:
        study_path (str): Relative study path from CSV (e.g., "MURA-v1.1/train/XR_WRIST/patient00001/study1_positive/")
        data_root (str): Root directory containing MURA dataset. If empty, study_path should be absolute.

    Returns:
        list: Sorted list of image file paths
    """
    # Strip "MURA-v1.1/" prefix from study_path if present
    # (CSV paths include it, but data_root already points to MURA-v1.1/)
    if study_path.startswith("MURA-v1.1/"):
        study_path = study_path[len("MURA-v1.1/"):]

    # Construct full path
    if data_root:
        full_path = os.path.join(data_root, study_path)
    else:
        full_path = study_path

    # Get all PNG files in the study directory
    images = sorted(glob.glob(os.path.join(full_path, "*.png")))

    if len(images) == 0:
        print(f"Warning: No images found in {full_path}")

    return images


def compute_class_weights(csv_path):
    """
    Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance.

    Args:
        csv_path (str): Path to training CSV file

    Returns:
        float: pos_weight value (num_negatives / num_positives)
    """
    df = pd.read_csv(csv_path, header=None, names=['study_path', 'label'])
    labels = df['label'].values

    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()

    pos_weight = neg_count / pos_count

    print(f"Class distribution:")
    print(f"  Positive (abnormal): {pos_count} ({pos_count / len(labels) * 100:.1f}%)")
    print(f"  Negative (normal): {neg_count} ({neg_count / len(labels) * 100:.1f}%)")
    print(f"  Calculated pos_weight: {pos_weight:.4f}")

    return pos_weight


def get_body_part_distribution(csv_path):
    """
    Get distribution of studies across different body parts.

    Args:
        csv_path (str): Path to CSV file

    Returns:
        dict: Dictionary mapping body part to count
    """
    df = pd.read_csv(csv_path, header=None, names=['study_path', 'label'])

    body_parts = {}
    for path in df['study_path']:
        # Extract body part from path (e.g., "MURA-v1.1/train/XR_WRIST/..." -> "XR_WRIST")
        parts = path.split('/')
        if len(parts) >= 3:
            body_part = parts[2]  # Assumes format: .../train/XR_BODYPART/...
            body_parts[body_part] = body_parts.get(body_part, 0) + 1

    # Sort by count
    body_parts = dict(sorted(body_parts.items(), key=lambda x: x[1], reverse=True))

    print("Body part distribution:")
    for part, count in body_parts.items():
        print(f"  {part}: {count}")

    return body_parts
