"""
Image Dataset for Forgery Detection
====================================
This module defines the PyTorch Dataset class for loading image data.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class ImageForgeryDataset(Dataset):
    """
    PyTorch Dataset for image forgery detection.

    This dataset loads images and their labels from a CSV file.
    The CSV should have 'image_path' and 'label' columns.
    """

    def __init__(self, csv_path, image_dir=None, transform=None, image_size=224):
        """
        Initialize the dataset.

        Args:
            csv_path (str): Path to CSV file with image paths and labels
            image_dir (str): Base directory for images (if paths are relative)
            transform: Custom transforms (if None, will use default)
            image_size (int): Target image size
        """
        print(f"[INFO] Loading image dataset from {csv_path}")

        # Load CSV
        self.data = pd.read_csv(csv_path)

        # Check required columns
        if 'image_path' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError("CSV must contain 'image_path' and 'label' columns")

        self.image_dir = image_dir

        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        # Get image paths and labels
        self.image_paths = self.data['image_path'].tolist()
        self.labels = self.data['label'].tolist()

        # Print dataset info
        print(f"[INFO] Dataset loaded successfully!")
        print(f"       - Total samples: {len(self.image_paths)}")
        print(f"       - Class distribution:")
        label_counts = self.data['label'].value_counts()
        for label, count in label_counts.items():
            label_name = "Authentic" if label == 0 else "Forged"
            print(f"         * {label_name} (Label {label}): {count} samples")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image_tensor, label)
        """
        # Get image path
        img_path = self.image_paths[idx]

        # If image_dir is specified, join paths
        if self.image_dir is not None:
            img_path = os.path.join(self.image_dir, img_path)

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Could not load image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        image_tensor = self.transform(image)

        # Get label
        label = self.labels[idx]

        return image_tensor, torch.tensor(label, dtype=torch.long)


class AugmentedImageDataset(ImageForgeryDataset):
    """
    Image dataset with data augmentation for training.

    Applies random transformations to increase dataset diversity.
    """

    def __init__(self, csv_path, image_dir=None, image_size=224):
        """
        Initialize the augmented dataset.

        Args:
            csv_path (str): Path to CSV file
            image_dir (str): Base directory for images
            image_size (int): Target image size
        """
        # Define augmentation transforms
        augment_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),  # Resize slightly larger
            transforms.RandomCrop(image_size),                      # Random crop
            transforms.RandomHorizontalFlip(p=0.5),                 # Random flip
            transforms.RandomRotation(degrees=10),                  # Small rotation
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),                                                      # Color variations
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        super().__init__(csv_path, image_dir, augment_transform, image_size)

        print("[INFO] Using data augmentation for training")


def create_data_loaders(
    train_csv,
    val_csv,
    test_csv=None,
    image_dir=None,
    batch_size=32,
    image_size=224,
    use_augmentation=True
):
    """
    Create DataLoaders for training, validation, and test sets.

    Args:
        train_csv (str): Path to training CSV
        val_csv (str): Path to validation CSV
        test_csv (str): Path to test CSV (optional)
        image_dir (str): Base directory for images
        batch_size (int): Batch size
        image_size (int): Target image size
        use_augmentation (bool): Whether to use augmentation for training

    Returns:
        tuple: DataLoaders
    """
    print("\n" + "=" * 50)
    print("Creating Image Data Loaders")
    print("=" * 50)

    # Create training dataset (with or without augmentation)
    print("\nLoading training data...")
    if use_augmentation:
        train_dataset = AugmentedImageDataset(train_csv, image_dir, image_size)
    else:
        train_dataset = ImageForgeryDataset(train_csv, image_dir, image_size=image_size)

    # Create validation dataset (no augmentation)
    print("\nLoading validation data...")
    val_dataset = ImageForgeryDataset(val_csv, image_dir, image_size=image_size)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n[INFO] Training DataLoader: {len(train_loader)} batches")
    print(f"[INFO] Validation DataLoader: {len(val_loader)} batches")

    if test_csv is not None:
        print("\nLoading test data...")
        test_dataset = ImageForgeryDataset(test_csv, image_dir, image_size=image_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        print(f"[INFO] Test DataLoader: {len(test_loader)} batches")
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Image Forgery Dataset")
    print("=" * 50)

    # Create sample CSV for testing
    sample_data = {
        'image_path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
        'label': [0, 1, 0, 1]
    }
    test_df = pd.DataFrame(sample_data)
    test_csv = 'test_image_dataset.csv'
    test_df.to_csv(test_csv, index=False)

    print(f"\nCreated test CSV with {len(test_df)} samples")

    # Create sample images
    os.makedirs('test_images', exist_ok=True)
    for i in range(1, 5):
        # Create dummy image
        img = Image.new('RGB', (256, 256), color=(i*60, i*40, i*30))
        img.save(f'test_images/img{i}.jpg')

    print("Created sample images")

    # Create dataset
    dataset = ImageForgeryDataset(test_csv, image_dir='test_images')

    print(f"\nDataset length: {len(dataset)}")

    # Test __getitem__
    print("\nSample 0:")
    image, label = dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label}")

    # Test DataLoader
    print("\nTesting DataLoader:")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for i, (images, labels) in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels}")
        if i == 0:
            break

    # Clean up
    os.remove(test_csv)
    import shutil
    shutil.rmtree('test_images')

    print("\n[SUCCESS] Dataset test completed!")
