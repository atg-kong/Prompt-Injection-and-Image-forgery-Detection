"""
Training Script for Prompt Injection Detection Model
=====================================================
This script trains the BERT-based prompt injection detector.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

# Import our custom modules
from model import PromptInjectionDetector, save_model
from text_dataset import PromptInjectionDataset


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on

    Returns:
        tuple: (average_loss, accuracy)
    """
    # Set model to training mode
    model.train()

    # Initialize metrics
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # Progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Update learning rate
        scheduler.step()

        # Update metrics
        total_loss += loss.item()

        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        # Update progress bar
        current_loss = total_loss / (progress_bar.n + 1)
        current_acc = correct_predictions / total_samples
        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.4f}'
        })

    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation/test data.

    Args:
        model: The model to evaluate
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        tuple: (average_loss, accuracy, predictions, true_labels)
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # Store predictions and labels for detailed analysis
    all_predictions = []
    all_labels = []

    # Progress bar
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    # Disable gradient computation
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Store for later analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            current_loss = total_loss / (progress_bar.n + 1)
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}'
            })

    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


def train_model(
    train_csv_path,
    val_csv_path,
    save_dir='./saved_models',
    num_epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    max_length=512,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Main training function.

    Args:
        train_csv_path (str): Path to training CSV
        val_csv_path (str): Path to validation CSV
        save_dir (str): Directory to save models
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        max_length (int): Maximum sequence length
        device (str): Device to train on

    Returns:
        dict: Training history
    """
    print("=" * 60)
    print("PROMPT INJECTION DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 60)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create datasets
    print("\n[STEP 1] Loading datasets...")
    train_dataset = PromptInjectionDataset(train_csv_path, max_length=max_length)
    val_dataset = PromptInjectionDataset(val_csv_path, max_length=max_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    print("\n[STEP 2] Creating model...")
    model = PromptInjectionDetector(num_classes=2, dropout_rate=0.3)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Define loss function
    # CrossEntropyLoss is good for classification
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    # AdamW is recommended for transformers (Adam with weight decay)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Define learning rate scheduler
    # Linear warmup then linear decay
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.1)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

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

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch results
        print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = os.path.join(save_dir, 'best_text_model.pt')
            save_model(model, best_model_path)
            print(f"[NEW BEST] Model saved with accuracy: {best_val_acc:.4f}")

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_text_model.pt')
    save_model(model, final_model_path)

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(save_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)

    # Print summary
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
    """
    Main function to run training.
    """
    # Configuration
    # These paths should point to your actual dataset files
    TRAIN_CSV = '../../data/processed/train_text.csv'
    VAL_CSV = '../../data/processed/val_text.csv'
    SAVE_DIR = '../../saved_models/text_model'

    # Training parameters
    # These were tuned during our experiments
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Train the model
    history = train_model(
        train_csv_path=TRAIN_CSV,
        val_csv_path=VAL_CSV,
        save_dir=SAVE_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        device=device
    )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
