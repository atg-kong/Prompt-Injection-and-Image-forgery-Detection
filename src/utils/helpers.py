"""
Helper Functions Module
=======================
This module provides utility functions used across the project.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import os
import json
import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[INFO] Random seed set to {seed}")


def create_directory(path):
    """
    Create directory if it doesn't exist.

    Args:
        path (str): Directory path to create

    Returns:
        str: Created directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created directory: {path}")
    return path


def get_device():
    """
    Get the best available device (GPU or CPU).

    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = 'cpu'
        print("[INFO] Using CPU (GPU not available)")

    return device


def save_json(data, path):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        path (str): File path
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[INFO] Data saved to {path}")


def load_json(path):
    """
    Load data from JSON file.

    Args:
        path (str): File path

    Returns:
        dict: Loaded data
    """
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"[INFO] Data loaded from {path}")
    return data


def get_timestamp():
    """
    Get current timestamp string.

    Returns:
        str: Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_time(seconds):
    """
    Format seconds into human-readable time.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def count_parameters(model):
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model):
    """
    Print summary of model parameters.

    Args:
        model: PyTorch model
    """
    total, trainable = count_parameters(model)
    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable:        {total - trainable:,}")
    print("=" * 50)


class Timer:
    """
    Simple timer class for measuring execution time.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = datetime.now()
        return self

    def stop(self):
        """Stop the timer."""
        self.end_time = datetime.now()
        return self

    def elapsed(self):
        """
        Get elapsed time.

        Returns:
            float: Elapsed time in seconds
        """
        if self.start_time is None:
            return 0

        if self.end_time is None:
            end = datetime.now()
        else:
            end = self.end_time

        return (end - self.start_time).total_seconds()

    def elapsed_formatted(self):
        """
        Get formatted elapsed time.

        Returns:
            str: Formatted time string
        """
        return format_time(self.elapsed())


class AverageMeter:
    """
    Keeps track of average values.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with new value.

        Args:
            val: New value
            n (int): Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def batch_generator(data, batch_size):
    """
    Generate batches from data.

    Args:
        data: Input data (list or array)
        batch_size (int): Batch size

    Yields:
        Batches of data
    """
    n_samples = len(data)
    for i in range(0, n_samples, batch_size):
        yield data[i:i + batch_size]


def early_stopping_check(val_losses, patience=5, min_delta=0.001):
    """
    Check if training should stop early.

    Args:
        val_losses (list): List of validation losses
        patience (int): Number of epochs to wait
        min_delta (float): Minimum improvement

    Returns:
        bool: True if should stop
    """
    if len(val_losses) < patience + 1:
        return False

    recent = val_losses[-patience:]
    best_recent = min(recent)
    prev_best = min(val_losses[:-patience])

    # If no improvement
    if best_recent >= prev_best - min_delta:
        return True

    return False


def log_experiment(config, results, log_file='experiment_log.json'):
    """
    Log experiment configuration and results.

    Args:
        config (dict): Experiment configuration
        results (dict): Experiment results
        log_file (str): Path to log file
    """
    # Load existing log if exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log = json.load(f)
    else:
        log = {'experiments': []}

    # Add new experiment
    experiment = {
        'timestamp': get_timestamp(),
        'config': config,
        'results': results
    }
    log['experiments'].append(experiment)

    # Save log
    save_json(log, log_file)


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Helper Functions")
    print("=" * 50)

    # Test seed setting
    set_seed(42)
    print(f"Random int: {random.randint(0, 100)}")
    print(f"Numpy random: {np.random.rand():.4f}")

    # Test device
    device = get_device()

    # Test timer
    print("\nTesting timer:")
    timer = Timer()
    timer.start()
    import time
    time.sleep(1.5)
    timer.stop()
    print(f"Elapsed: {timer.elapsed_formatted()}")

    # Test average meter
    print("\nTesting average meter:")
    meter = AverageMeter()
    for val in [0.5, 0.4, 0.35, 0.32, 0.30]:
        meter.update(val)
        print(f"Value: {val:.2f}, Average: {meter.avg:.4f}")

    # Test batch generator
    print("\nTesting batch generator:")
    data = list(range(10))
    for i, batch in enumerate(batch_generator(data, batch_size=3)):
        print(f"Batch {i}: {batch}")

    # Test early stopping
    print("\nTesting early stopping:")
    val_losses = [0.8, 0.7, 0.65, 0.64, 0.64, 0.64, 0.64, 0.64]
    should_stop = early_stopping_check(val_losses, patience=5)
    print(f"Should stop early: {should_stop}")

    # Test format time
    print(f"\nFormat 125 seconds: {format_time(125)}")
    print(f"Format 7200 seconds: {format_time(7200)}")

    # Test timestamp
    print(f"\nCurrent timestamp: {get_timestamp()}")

    print("\n[SUCCESS] Helper functions test completed!")
