"""
CLIP Module for Text-Image Consistency Checking
================================================
This module uses OpenAI's CLIP model to verify text-image consistency.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import torch
import clip
from PIL import Image
import numpy as np


class CLIPChecker:
    """
    CLIP-based text-image consistency checker.

    Uses OpenAI's CLIP model to compute similarity between
    text descriptions and images.
    """

    def __init__(self, model_name='ViT-B/32', device=None):
        """
        Initialize the CLIP checker.

        Args:
            model_name (str): CLIP model variant to use
            device (str): Device to run on ('cpu' or 'cuda')
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[INFO] Loading CLIP model '{model_name}' on {self.device}...")

        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        # Set model to evaluation mode
        self.model.eval()

        print(f"[INFO] CLIP model loaded successfully!")

    def compute_similarity(self, image, text):
        """
        Compute similarity between an image and text.

        Args:
            image: PIL Image, numpy array, or path to image
            text (str): Text description

        Returns:
            float: Similarity score between 0 and 1
        """
        # Load image if path is given
        if isinstance(image, str):
            image = Image.open(image)

        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize text
        text_input = clip.tokenize([text]).to(self.device)

        # Compute features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            similarity = (image_features @ text_features.T).item()

        # Convert to 0-1 range (CLIP similarity can be negative)
        # Typically ranges from -1 to 1, but we normalize to 0-1
        similarity = (similarity + 1) / 2

        return similarity

    def check_consistency(self, image, text, threshold=0.5):
        """
        Check if text description is consistent with image.

        Args:
            image: Image input
            text (str): Text description
            threshold (float): Similarity threshold (default 0.5)

        Returns:
            dict: Consistency check results
        """
        similarity = self.compute_similarity(image, text)

        is_consistent = similarity >= threshold

        # Determine consistency level
        if similarity >= 0.8:
            consistency_level = 'HIGH'
        elif similarity >= 0.6:
            consistency_level = 'MEDIUM'
        elif similarity >= 0.4:
            consistency_level = 'LOW'
        else:
            consistency_level = 'VERY LOW'

        return {
            'similarity': similarity,
            'is_consistent': is_consistent,
            'consistency_level': consistency_level,
            'threshold_used': threshold
        }

    def rank_captions(self, image, captions):
        """
        Rank multiple captions by their similarity to the image.

        Useful for finding the best matching description.

        Args:
            image: Image input
            captions (list): List of text captions

        Returns:
            list: List of (caption, similarity) tuples, sorted by similarity
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize all captions
        text_inputs = clip.tokenize(captions).to(self.device)

        # Compute features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            similarities = (image_features @ text_features.T).squeeze().cpu().numpy()

        # Normalize to 0-1
        similarities = (similarities + 1) / 2

        # Create ranked list
        ranked = sorted(zip(captions, similarities), key=lambda x: x[1], reverse=True)

        return ranked

    def detect_mismatch(self, image, claimed_description, alternatives=None, threshold=0.4):
        """
        Detect if the claimed description is a mismatch for the image.

        This can help detect misinformation where images are paired with
        misleading captions.

        Args:
            image: Image input
            claimed_description (str): The description claimed for the image
            alternatives (list): Alternative descriptions to compare
            threshold (float): Similarity threshold below which is a mismatch

        Returns:
            dict: Mismatch detection results
        """
        # Get similarity with claimed description
        claimed_similarity = self.compute_similarity(image, claimed_description)

        result = {
            'claimed_description': claimed_description,
            'claimed_similarity': claimed_similarity,
            'is_mismatch': claimed_similarity < threshold
        }

        # If alternatives provided, compare
        if alternatives:
            alt_similarities = {}
            for alt in alternatives:
                alt_similarities[alt] = self.compute_similarity(image, alt)

            # Check if any alternative is significantly better
            best_alt = max(alt_similarities.items(), key=lambda x: x[1])

            result['alternatives'] = alt_similarities
            result['best_alternative'] = best_alt[0]
            result['best_alternative_similarity'] = best_alt[1]

            # If alternative is much better, likely a mismatch
            if best_alt[1] - claimed_similarity > 0.2:
                result['is_mismatch'] = True
                result['reason'] = f"Alternative '{best_alt[0]}' matches better"

        return result

    def get_embedding(self, image=None, text=None):
        """
        Get CLIP embeddings for image and/or text.

        Args:
            image: Image input (optional)
            text: Text input (optional)

        Returns:
            dict: Embeddings for provided inputs
        """
        result = {}

        if image is not None:
            if isinstance(image, str):
                image = Image.open(image)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            result['image_embedding'] = image_features.cpu().numpy()

        if text is not None:
            text_input = clip.tokenize([text]).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            result['text_embedding'] = text_features.cpu().numpy()

        return result


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Testing CLIP Consistency Checker")
    print("=" * 50)

    # Create checker
    checker = CLIPChecker()

    # Create test image
    from PIL import ImageDraw

    # Simple image of a "blue square"
    img = Image.new('RGB', (224, 224), color='blue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 200, 200], fill='blue', outline='white')

    test_path = 'test_clip_image.png'
    img.save(test_path)

    print("\nCreated test image: Blue square")

    # Test similarity with different descriptions
    descriptions = [
        "a blue square",
        "a red circle",
        "a green triangle",
        "blue color",
        "a dog sitting"
    ]

    print("\nSimilarity scores:")
    for desc in descriptions:
        sim = checker.compute_similarity(test_path, desc)
        print(f"  '{desc}': {sim:.4f}")

    # Test consistency check
    print("\nConsistency check:")
    result = checker.check_consistency(test_path, "a blue square")
    print(f"  Description: 'a blue square'")
    print(f"  Similarity: {result['similarity']:.4f}")
    print(f"  Is consistent: {result['is_consistent']}")
    print(f"  Level: {result['consistency_level']}")

    # Test ranking
    print("\nCaption ranking:")
    ranked = checker.rank_captions(test_path, descriptions)
    for i, (caption, sim) in enumerate(ranked):
        print(f"  {i+1}. '{caption}': {sim:.4f}")

    # Test mismatch detection
    print("\nMismatch detection:")
    mismatch = checker.detect_mismatch(
        test_path,
        "a red circle",
        alternatives=["a blue square", "blue color"]
    )
    print(f"  Claimed: '{mismatch['claimed_description']}'")
    print(f"  Claimed similarity: {mismatch['claimed_similarity']:.4f}")
    print(f"  Is mismatch: {mismatch['is_mismatch']}")
    if 'best_alternative' in mismatch:
        print(f"  Best alternative: '{mismatch['best_alternative']}'")
        print(f"  Best alternative similarity: {mismatch['best_alternative_similarity']:.4f}")

    # Clean up
    import os
    os.remove(test_path)

    print("\n[SUCCESS] CLIP module test completed!")
