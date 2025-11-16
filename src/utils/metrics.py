"""
Evaluation Metrics Module
=========================
This module provides functions for calculating evaluation metrics.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_proba (array): Predicted probabilities (optional)

    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm

    # Extract TP, TN, FP, FN
    if cm.shape == (2, 2):
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]

        # Calculate additional metrics
        metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        metrics['sensitivity'] = metrics['recall']  # Same as recall

    # ROC AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0

    return metrics


def print_metrics(metrics):
    """
    Print metrics in a formatted way.

    Args:
        metrics (dict): Dictionary of metrics
    """
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)

    print(f"\nAccuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")

    if 'specificity' in metrics:
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")

    if 'roc_auc' in metrics:
        print(f"ROC AUC:     {metrics['roc_auc']:.4f}")

    if 'confusion_matrix' in metrics:
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              Neg    Pos")
        print(f"Actual Neg    {metrics['true_negatives']:4d}   {metrics['false_positives']:4d}")
        print(f"       Pos    {metrics['false_negatives']:4d}   {metrics['true_positives']:4d}")

    print("\n" + "=" * 50)


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Plot confusion matrix as a heatmap.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        labels (list): Class labels
        save_path (str): Path to save the plot
    """
    if labels is None:
        labels = ['Safe', 'Malicious']

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Confusion matrix saved to {save_path}")

    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy curves).

    Args:
        history (dict): Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', marker='s')
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', marker='o')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', marker='s')
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Training history plot saved to {save_path}")

    plt.show()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Plot ROC curve.

    Args:
        y_true (array): True labels
        y_proba (array): Predicted probabilities
        save_path (str): Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] ROC curve saved to {save_path}")

    plt.show()


def compare_models(results_dict, save_path=None):
    """
    Compare performance of multiple models.

    Args:
        results_dict (dict): Dictionary with model names as keys and metrics as values
        save_path (str): Path to save the comparison table
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Header
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)

    # Data rows
    for model_name, metrics in results_dict.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")

    print("=" * 70)

    # Create bar chart
    models = list(results_dict.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics_names):
        values = [results_dict[model][metric] for model in models]
        bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Comparison plot saved to {save_path}")

    plt.show()


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Metrics Module")
    print("=" * 50)

    # Create synthetic predictions
    np.random.seed(42)
    n_samples = 100

    # True labels
    y_true = np.array([0] * 50 + [1] * 50)

    # Predictions (with some errors)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, size=10, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]

    # Probabilities
    y_proba = np.random.uniform(0.1, 0.4, n_samples)
    y_proba[y_true == 1] = np.random.uniform(0.6, 0.9, 50)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    # Print metrics
    print_metrics(metrics)

    # Plot confusion matrix (commented to avoid GUI issues in testing)
    # plot_confusion_matrix(y_true, y_pred)

    # Simulate training history
    history = {
        'train_loss': [0.8, 0.6, 0.45, 0.35, 0.28],
        'val_loss': [0.85, 0.65, 0.55, 0.50, 0.48],
        'train_acc': [0.65, 0.75, 0.82, 0.88, 0.91],
        'val_acc': [0.62, 0.72, 0.78, 0.82, 0.84]
    }

    # Compare models
    model_results = {
        'Text Model': {'accuracy': 0.912, 'precision': 0.90, 'recall': 0.92, 'f1_score': 0.89},
        'Image Model': {'accuracy': 0.875, 'precision': 0.85, 'recall': 0.89, 'f1_score': 0.86},
        'Fusion Model': {'accuracy': 0.893, 'precision': 0.88, 'recall': 0.91, 'f1_score': 0.89}
    }

    compare_models(model_results)

    print("\n[SUCCESS] Metrics module test completed!")
