"""
Text Model for Prompt Injection Detection
==========================================
This module defines the BERT-based classifier for detecting prompt injection attacks.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class PromptInjectionDetector(nn.Module):
    """
    BERT-based text classifier for prompt injection detection.

    This model uses a pre-trained BERT model and adds custom classification
    layers on top to detect prompt injection attempts in text.

    Architecture:
        BERT Base (pretrained) -> Dropout -> FC1 (768->256) -> ReLU -> FC2 (256->2)
    """

    def __init__(self, num_classes=2, dropout_rate=0.3):
        """
        Initialize the Prompt Injection Detector model.

        Args:
            num_classes (int): Number of output classes (default: 2 for binary classification)
            dropout_rate (float): Dropout probability for regularization
        """
        super(PromptInjectionDetector, self).__init__()

        # Load pre-trained BERT model
        # We use bert-base-uncased as it's smaller and sufficient for our task
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Get the hidden size from BERT (768 for base model)
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Dropout layer to prevent overfitting
        # We found 0.3 works well during our experiments
        self.dropout = nn.Dropout(dropout_rate)

        # First fully connected layer
        # Reduces dimension from 768 to 256
        self.fc1 = nn.Linear(self.hidden_size, 256)

        # ReLU activation function
        self.relu = nn.ReLU()

        # Second fully connected layer
        # Maps from 256 to number of classes (2 in our case)
        self.fc2 = nn.Linear(256, num_classes)

        # Note: We don't add softmax here because CrossEntropyLoss in PyTorch
        # already applies LogSoftmax internally

        print(f"[INFO] PromptInjectionDetector initialized with:")
        print(f"       - BERT hidden size: {self.hidden_size}")
        print(f"       - Dropout rate: {dropout_rate}")
        print(f"       - Number of classes: {num_classes}")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tokenized input text (batch_size, seq_length)
            attention_mask (torch.Tensor): Attention mask for padding (batch_size, seq_length)

        Returns:
            torch.Tensor: Logits for each class (batch_size, num_classes)
        """
        # Pass input through BERT
        # BERT returns multiple outputs, we need the pooled output (CLS token representation)
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get the [CLS] token representation
        # This is a 768-dimensional vector that represents the entire sequence
        pooled_output = bert_output.pooler_output  # Shape: (batch_size, 768)

        # Apply dropout for regularization
        x = self.dropout(pooled_output)

        # Pass through first fully connected layer
        x = self.fc1(x)  # Shape: (batch_size, 256)

        # Apply ReLU activation
        x = self.relu(x)

        # Apply dropout again
        x = self.dropout(x)

        # Pass through second fully connected layer to get final logits
        logits = self.fc2(x)  # Shape: (batch_size, num_classes)

        return logits

    def predict_proba(self, input_ids, attention_mask):
        """
        Get probability predictions for the input.

        Args:
            input_ids (torch.Tensor): Tokenized input text
            attention_mask (torch.Tensor): Attention mask

        Returns:
            torch.Tensor: Probabilities for each class
        """
        # Set model to evaluation mode
        self.eval()

        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1)

        return probabilities

    def predict(self, input_ids, attention_mask):
        """
        Get class predictions for the input.

        Args:
            input_ids (torch.Tensor): Tokenized input text
            attention_mask (torch.Tensor): Attention mask

        Returns:
            torch.Tensor: Predicted class labels
        """
        probabilities = self.predict_proba(input_ids, attention_mask)
        # Return the class with highest probability
        predictions = torch.argmax(probabilities, dim=1)

        return predictions


class TextPreprocessor:
    """
    Helper class for preprocessing text before feeding to the model.

    This handles tokenization, padding, and truncation of input text.
    """

    def __init__(self, max_length=512):
        """
        Initialize the text preprocessor.

        Args:
            max_length (int): Maximum sequence length (default: 512, BERT's max)
        """
        # Load BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

        print(f"[INFO] TextPreprocessor initialized with max_length={max_length}")

    def preprocess(self, text):
        """
        Preprocess a single text string.

        Args:
            text (str): Input text to preprocess

        Returns:
            dict: Dictionary containing input_ids and attention_mask
        """
        # Clean the text first
        text = self._clean_text(text)

        # Tokenize the text
        # This converts the text into token IDs that BERT can understand
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',      # Pad to max_length
            truncation=True,           # Truncate if longer than max_length
            return_tensors='pt'        # Return PyTorch tensors
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def preprocess_batch(self, texts):
        """
        Preprocess a batch of text strings.

        Args:
            texts (list): List of input texts

        Returns:
            dict: Dictionary containing batched input_ids and attention_mask
        """
        # Clean all texts
        texts = [self._clean_text(t) for t in texts]

        # Tokenize all texts at once
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def _clean_text(self, text):
        """
        Clean the input text.

        Args:
            text (str): Raw input text

        Returns:
            str: Cleaned text
        """
        # Convert to string if not already
        text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text


def load_model(model_path, device='cpu'):
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the saved model file
        device (str): Device to load the model on ('cpu' or 'cuda')

    Returns:
        PromptInjectionDetector: Loaded model
    """
    # Create model instance
    model = PromptInjectionDetector()

    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move model to specified device
    model = model.to(device)

    # Set to evaluation mode
    model.eval()

    print(f"[INFO] Model loaded from {model_path}")

    return model


def save_model(model, model_path):
    """
    Save a model to disk.

    Args:
        model (PromptInjectionDetector): Model to save
        model_path (str): Path where to save the model
    """
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path}")


# Example usage (for testing)
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Prompt Injection Detector Model")
    print("=" * 50)

    # Create model
    model = PromptInjectionDetector()

    # Create preprocessor
    preprocessor = TextPreprocessor(max_length=128)

    # Test with sample text
    test_text = "This is a sample text to test the model."

    # Preprocess
    inputs = preprocessor.preprocess(test_text)

    print(f"\nInput text: '{test_text}'")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Attention mask shape: {inputs['attention_mask'].shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])

    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits: {logits}")

    # Get probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities: {probs}")
    print(f"Predicted class: {torch.argmax(probs, dim=1).item()}")

    print("\n[SUCCESS] Model test completed!")
