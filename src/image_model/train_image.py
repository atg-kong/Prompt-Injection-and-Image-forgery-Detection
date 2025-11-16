"""
Training Script for Image Forgery Detection Model
==================================================
This script trains the EfficientNet-based image forgery detector.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

# Import our custom modules
from model import ImageForgeryDetector, save_model
from image_dataset import ImageForgeryDataset, AugmentedImageDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        current_loss = total_loss / (progress_bar.n + 1)
        current_acc = correct / total
        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.

    Args:
        model: The model to evaluate
        dataloader: Data loader
        criterion: Loss function
        device: Device

    Returns:
        tuple: (loss, accuracy, predictions, true_labels)
    """
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_loss = total_loss / (progress_bar.n + 1)
            current_acc = correct / total
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


def train_model(
    train_csv_path,
    val_csv_path,
    image_dir=None,
    save_dir='./saved_models',
    num_epochs=15,
    batch_size=32,
    learning_rate=1e-4,
    image_size=224,
    use_augmentation=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Main training function for image model.

    Args:
        train_csv_path (str): Path to training CSV
        val_csv_path (str): Path to validation CSV
        image_dir (str): Base directory for images
        save_dir (str): Directory to save models
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        image_size (int): Target image size
        use_augmentation (bool): Whether to use data augmentation
        device (str): Device to train on

    Returns:
        dict: Training history
    """
    print("=" * 60)
    print("IMAGE FORGERY DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Image Size: {image_size}x{image_size}")
    print(f"Data Augmentation: {use_augmentation}")
    print("=" * 60)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Create datasets
    print("\n[STEP 1] Loading datasets...")

    if use_augmentation:
        train_dataset = AugmentedImageDataset(train_csv_path, image_dir, image_size)
    else:
        train_dataset = ImageForgeryDataset(train_csv_path, image_dir, image_size=image_size)

    val_dataset = ImageForgeryDataset(val_csv_path, image_dir, image_size=image_size)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    print("\n[STEP 2] Creating model...")
    model = ImageForgeryDetector(num_classes=2, dropout_rate=0.2)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    # We use Adam with a smaller learning rate for transfer learning
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    # Reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Track best model
    best_val_acc = 0.0
    best_epoch = 0

    # Training loop
    print("\n[STEP 3] Starting training...")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print results
        print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = os.path.join(save_dir, 'best_image_model.pt')
            save_model(model, best_model_path)
            print(f"[NEW BEST] Model saved with accuracy: {best_val_acc:.4f}")

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_image_model.pt')
    save_model(model, final_model_path)

    # Save history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(save_dir, 'image_training_history.csv')
    history_df.to_csv(history_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    print("=" * 60)

    return history


def main():
    """Main function to run training."""
    # Configuration
    TRAIN_CSV = '../../data/processed/train_images.csv'
    VAL_CSV = '../../data/processed/val_images.csv'
    IMAGE_DIR = '../../data/processed/images'
    SAVE_DIR = '../../saved_models/image_model'

    # Training parameters
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 224
    USE_AUGMENTATION = True

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Train
    history = train_model(
        train_csv_path=TRAIN_CSV,
        val_csv_path=VAL_CSV,
        image_dir=IMAGE_DIR,
        save_dir=SAVE_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        image_size=IMAGE_SIZE,
        use_augmentation=USE_AUGMENTATION,
        device=device
    )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
