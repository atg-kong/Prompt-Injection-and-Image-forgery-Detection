"""
Image Model for Forgery Detection
==================================
This module defines the EfficientNet-based classifier for detecting image forgery.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class ImageForgeryDetector(nn.Module):
    """
    EfficientNet-based image classifier for forgery detection.

    This model uses a pre-trained EfficientNet-B0 and adds custom classification
    layers on top to detect forged/manipulated images.

    Architecture:
        EfficientNet-B0 (pretrained) -> Global Avg Pool -> Dropout -> FC1 -> ReLU -> FC2
    """

    def __init__(self, num_classes=2, dropout_rate=0.2):
        """
        Initialize the Image Forgery Detector model.

        Args:
            num_classes (int): Number of output classes (default: 2)
            dropout_rate (float): Dropout probability for regularization
        """
        super(ImageForgeryDetector, self).__init__()

        # Load pre-trained EfficientNet-B0
        # B0 is the smallest version, good for our academic project
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # Get the number of features from EfficientNet
        # For B0, this is 1280
        self.num_features = self.efficientnet._fc.in_features

        # Remove the original classification layer
        # We'll add our own custom layers
        self.efficientnet._fc = nn.Identity()

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # First fully connected layer
        # Reduces dimension from 1280 to 512
        self.fc1 = nn.Linear(self.num_features, 512)

        # ReLU activation
        self.relu = nn.ReLU()

        # Second fully connected layer
        # Maps to final number of classes
        self.fc2 = nn.Linear(512, num_classes)

        print(f"[INFO] ImageForgeryDetector initialized with:")
        print(f"       - EfficientNet features: {self.num_features}")
        print(f"       - Dropout rate: {dropout_rate}")
        print(f"       - Number of classes: {num_classes}")

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Logits for each class (batch_size, num_classes)
        """
        # Extract features using EfficientNet
        # This gives us a 1280-dimensional feature vector
        features = self.efficientnet(x)  # Shape: (batch_size, 1280)

        # Apply dropout
        x = self.dropout(features)

        # First fully connected layer
        x = self.fc1(x)  # Shape: (batch_size, 512)

        # ReLU activation
        x = self.relu(x)

        # Apply dropout again
        x = self.dropout(x)

        # Final classification layer
        logits = self.fc2(x)  # Shape: (batch_size, num_classes)

        return logits

    def extract_features(self, x):
        """
        Extract features from images without classification.

        Useful for feature visualization or other downstream tasks.

        Args:
            x (torch.Tensor): Input images

        Returns:
            torch.Tensor: Feature vectors
        """
        return self.efficientnet(x)

    def predict_proba(self, x):
        """
        Get probability predictions for the input.

        Args:
            x (torch.Tensor): Input images

        Returns:
            torch.Tensor: Probabilities for each class
        """
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities

    def predict(self, x):
        """
        Get class predictions for the input.

        Args:
            x (torch.Tensor): Input images

        Returns:
            torch.Tensor: Predicted class labels
        """
        probabilities = self.predict_proba(x)
        predictions = torch.argmax(probabilities, dim=1)

        return predictions


class ImagePreprocessor:
    """
    Helper class for preprocessing images before feeding to the model.

    Handles resizing, normalization, and tensor conversion.
    """

    def __init__(self, image_size=224):
        """
        Initialize the image preprocessor.

        Args:
            image_size (int): Target image size (default: 224 for EfficientNet)
        """
        from torchvision import transforms

        self.image_size = image_size

        # Define preprocessing transforms
        # These are standard transforms for ImageNet pre-trained models
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize to fixed size
            transforms.ToTensor(),  # Convert to tensor (0-1 range)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

        print(f"[INFO] ImagePreprocessor initialized with size={image_size}x{image_size}")

    def preprocess(self, image):
        """
        Preprocess a single image.

        Args:
            image: PIL Image or path to image file

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        from PIL import Image

        # Load image if path is given
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # Apply transforms
        tensor = self.transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)

        return tensor

    def preprocess_batch(self, images):
        """
        Preprocess a batch of images.

        Args:
            images (list): List of PIL Images or paths

        Returns:
            torch.Tensor: Batch of preprocessed images
        """
        from PIL import Image

        tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            tensors.append(self.transform(img))

        # Stack into batch
        batch = torch.stack(tensors)  # Shape: (batch_size, 3, 224, 224)

        return batch


def load_model(model_path, device='cpu'):
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the saved model file
        device (str): Device to load the model on

    Returns:
        ImageForgeryDetector: Loaded model
    """
    model = ImageForgeryDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded from {model_path}")

    return model


def save_model(model, model_path):
    """
    Save a model to disk.

    Args:
        model (ImageForgeryDetector): Model to save
        model_path (str): Path where to save the model
    """
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path}")


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Image Forgery Detector Model")
    print("=" * 50)

    # Create model
    model = ImageForgeryDetector()

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Create dummy input
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images

    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")

    # Get probabilities
    probs = torch.softmax(output, dim=1)
    print(f"Probabilities: {probs}")
    print(f"Predictions: {torch.argmax(probs, dim=1)}")

    print("\n[SUCCESS] Model test completed!")
