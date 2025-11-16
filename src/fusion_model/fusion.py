"""
Fusion Model for Combining Detection Results
=============================================
This module combines results from text, image, OCR, and CLIP modules.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle


class FusionModel:
    """
    Fusion model that combines results from multiple detection modules.

    Uses both rule-based logic and machine learning (Logistic Regression)
    to make final decisions.
    """

    def __init__(self):
        """Initialize the fusion model."""
        # Logistic Regression for ML-based fusion
        self.ml_model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

        # Rule-based thresholds (tuned during experiments)
        self.thresholds = {
            'text_injection': 0.7,      # High confidence for injection
            'image_forgery': 0.7,       # High confidence for forgery
            'clip_mismatch': 0.3,       # Low similarity indicates mismatch
            'ocr_injection': 0.6        # OCR detected injection threshold
        }

        print("[INFO] FusionModel initialized")
        print(f"       Thresholds: {self.thresholds}")

    def train(self, features, labels):
        """
        Train the ML-based fusion model.

        Args:
            features (np.array): Feature array of shape (n_samples, n_features)
                                 Features: [text_score, image_score, clip_score, ocr_score]
            labels (np.array): Binary labels (0=Safe, 1=Malicious)
        """
        print(f"[INFO] Training fusion model on {len(labels)} samples...")

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train logistic regression
        self.ml_model.fit(features_scaled, labels)

        self.is_trained = True

        # Print coefficients for interpretability
        print("[INFO] Fusion model trained successfully!")
        print("       Feature importance (coefficients):")
        feature_names = ['text_injection', 'image_forgery', 'clip_similarity', 'ocr_injection']
        for name, coef in zip(feature_names, self.ml_model.coef_[0]):
            print(f"         - {name}: {coef:.4f}")
        print(f"       Intercept: {self.ml_model.intercept_[0]:.4f}")

    def predict_rule_based(self, text_score, image_score, clip_score, ocr_score=0.0):
        """
        Make prediction using rule-based logic.

        Args:
            text_score (float): Prompt injection probability (0-1)
            image_score (float): Image forgery probability (0-1)
            clip_score (float): CLIP similarity score (0-1)
            ocr_score (float): OCR injection score (0-1)

        Returns:
            dict: Prediction results
        """
        reasons = []
        is_malicious = False

        # Rule 1: High text injection probability
        if text_score > self.thresholds['text_injection']:
            is_malicious = True
            reasons.append(f"High prompt injection probability: {text_score:.3f}")

        # Rule 2: High image forgery probability
        if image_score > self.thresholds['image_forgery']:
            is_malicious = True
            reasons.append(f"High image forgery probability: {image_score:.3f}")

        # Rule 3: Low CLIP similarity (text-image mismatch)
        if clip_score < self.thresholds['clip_mismatch']:
            is_malicious = True
            reasons.append(f"Low text-image consistency: {clip_score:.3f}")

        # Rule 4: OCR detected injection
        if ocr_score > self.thresholds['ocr_injection']:
            is_malicious = True
            reasons.append(f"Hidden text injection detected: {ocr_score:.3f}")

        # Rule 5: Multiple moderate signals
        moderate_count = 0
        if text_score > 0.5:
            moderate_count += 1
        if image_score > 0.5:
            moderate_count += 1
        if clip_score < 0.4:
            moderate_count += 1
        if ocr_score > 0.4:
            moderate_count += 1

        if moderate_count >= 3 and not is_malicious:
            is_malicious = True
            reasons.append(f"Multiple moderate threat signals: {moderate_count}/4")

        return {
            'prediction': 1 if is_malicious else 0,
            'is_malicious': is_malicious,
            'reasons': reasons,
            'method': 'rule_based'
        }

    def predict_ml(self, text_score, image_score, clip_score, ocr_score=0.0):
        """
        Make prediction using ML model.

        Args:
            text_score (float): Prompt injection probability
            image_score (float): Image forgery probability
            clip_score (float): CLIP similarity score
            ocr_score (float): OCR injection score

        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            print("[WARNING] ML model not trained, falling back to rule-based")
            return self.predict_rule_based(text_score, image_score, clip_score, ocr_score)

        # Prepare features
        features = np.array([[text_score, image_score, clip_score, ocr_score]])
        features_scaled = self.scaler.transform(features)

        # Get prediction and probability
        prediction = self.ml_model.predict(features_scaled)[0]
        probability = self.ml_model.predict_proba(features_scaled)[0]

        return {
            'prediction': int(prediction),
            'is_malicious': bool(prediction == 1),
            'probability': float(probability[1]),  # Probability of malicious
            'method': 'ml_based'
        }

    def predict(self, text_score, image_score, clip_score, ocr_score=0.0, method='combined'):
        """
        Make final prediction using combined approach.

        Args:
            text_score (float): Prompt injection probability
            image_score (float): Image forgery probability
            clip_score (float): CLIP similarity score
            ocr_score (float): OCR injection score
            method (str): 'rule_based', 'ml_based', or 'combined'

        Returns:
            dict: Final prediction results
        """
        if method == 'rule_based':
            return self.predict_rule_based(text_score, image_score, clip_score, ocr_score)
        elif method == 'ml_based':
            return self.predict_ml(text_score, image_score, clip_score, ocr_score)
        else:  # combined
            # Get both predictions
            rule_result = self.predict_rule_based(text_score, image_score, clip_score, ocr_score)
            ml_result = self.predict_ml(text_score, image_score, clip_score, ocr_score)

            # Combined logic:
            # If ML is confident (prob > 0.8 or < 0.2), trust ML
            # Otherwise, use rule-based as override
            if self.is_trained:
                ml_prob = ml_result.get('probability', 0.5)

                if ml_prob > 0.8 or ml_prob < 0.2:
                    # ML is confident
                    final_prediction = ml_result['prediction']
                    final_malicious = ml_result['is_malicious']
                    confidence_source = 'ml_confident'
                else:
                    # Use rule-based for edge cases
                    if rule_result['is_malicious']:
                        final_prediction = 1
                        final_malicious = True
                        confidence_source = 'rule_override'
                    else:
                        final_prediction = ml_result['prediction']
                        final_malicious = ml_result['is_malicious']
                        confidence_source = 'ml_uncertain'
            else:
                final_prediction = rule_result['prediction']
                final_malicious = rule_result['is_malicious']
                confidence_source = 'rule_only'

            # Calculate confidence score
            confidence = self._calculate_confidence(
                text_score, image_score, clip_score, ocr_score, final_malicious
            )

            return {
                'prediction': final_prediction,
                'is_malicious': final_malicious,
                'label': 'MALICIOUS' if final_malicious else 'SAFE',
                'confidence': confidence,
                'confidence_source': confidence_source,
                'rule_based_result': rule_result,
                'ml_based_result': ml_result,
                'individual_scores': {
                    'text_injection': text_score,
                    'image_forgery': image_score,
                    'clip_similarity': clip_score,
                    'ocr_injection': ocr_score
                },
                'method': 'combined'
            }

    def _calculate_confidence(self, text_score, image_score, clip_score, ocr_score, is_malicious):
        """
        Calculate confidence score for the prediction.

        Args:
            All individual scores and final prediction

        Returns:
            float: Confidence score (0-1)
        """
        if is_malicious:
            # Confidence based on how strong the malicious signals are
            max_threat_score = max(text_score, image_score, 1 - clip_score, ocr_score)
            confidence = max_threat_score
        else:
            # Confidence based on how low the threat scores are
            threat_scores = [text_score, image_score, 1 - clip_score, ocr_score]
            avg_threat = np.mean(threat_scores)
            confidence = 1 - avg_threat

        return float(np.clip(confidence, 0, 1))

    def save(self, path):
        """
        Save the fusion model to disk.

        Args:
            path (str): Path to save the model
        """
        model_data = {
            'ml_model': self.ml_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'thresholds': self.thresholds
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"[INFO] Fusion model saved to {path}")

    def load(self, path):
        """
        Load the fusion model from disk.

        Args:
            path (str): Path to the saved model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.ml_model = model_data['ml_model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.thresholds = model_data['thresholds']

        print(f"[INFO] Fusion model loaded from {path}")


def generate_report(fusion_result):
    """
    Generate a human-readable report from fusion results.

    Args:
        fusion_result (dict): Result from FusionModel.predict()

    Returns:
        str: Formatted report
    """
    report = []
    report.append("=" * 50)
    report.append("MULTIMODAL DETECTION REPORT")
    report.append("=" * 50)

    # Final verdict
    label = fusion_result.get('label', 'UNKNOWN')
    confidence = fusion_result.get('confidence', 0)

    if label == 'MALICIOUS':
        report.append(f"\n[VERDICT] {label}")
    else:
        report.append(f"\n[VERDICT] {label}")

    report.append(f"Confidence: {confidence:.2%}")

    # Individual scores
    scores = fusion_result.get('individual_scores', {})
    report.append("\n--- Individual Analysis Scores ---")
    report.append(f"  Text Injection Probability: {scores.get('text_injection', 0):.3f}")
    report.append(f"  Image Forgery Probability:  {scores.get('image_forgery', 0):.3f}")
    report.append(f"  Text-Image Consistency:     {scores.get('clip_similarity', 0):.3f}")
    report.append(f"  OCR Hidden Text Score:      {scores.get('ocr_injection', 0):.3f}")

    # Rule-based reasoning
    rule_result = fusion_result.get('rule_based_result', {})
    if rule_result.get('reasons'):
        report.append("\n--- Detection Reasons ---")
        for reason in rule_result['reasons']:
            report.append(f"  - {reason}")

    report.append("\n" + "=" * 50)

    return '\n'.join(report)


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Fusion Model")
    print("=" * 50)

    # Create fusion model
    fusion = FusionModel()

    # Test rule-based prediction
    print("\nTest 1: Clear prompt injection")
    result = fusion.predict(
        text_score=0.85,    # High injection probability
        image_score=0.2,    # Low forgery
        clip_score=0.7,     # Good consistency
        ocr_score=0.1       # Low OCR threat
    )
    print(generate_report(result))

    print("\nTest 2: Forged image with mismatched caption")
    result = fusion.predict(
        text_score=0.15,    # Low injection
        image_score=0.82,   # High forgery
        clip_score=0.25,    # Low consistency (mismatch)
        ocr_score=0.1       # Low OCR threat
    )
    print(generate_report(result))

    print("\nTest 3: Safe content")
    result = fusion.predict(
        text_score=0.1,     # Low all scores
        image_score=0.15,
        clip_score=0.85,    # High consistency
        ocr_score=0.05
    )
    print(generate_report(result))

    # Train ML model with synthetic data
    print("\nTraining ML model with synthetic data...")
    np.random.seed(42)

    # Generate synthetic training data
    n_samples = 200
    features = []
    labels = []

    for _ in range(n_samples // 2):
        # Safe samples
        features.append([
            np.random.uniform(0, 0.4),    # Low text injection
            np.random.uniform(0, 0.4),    # Low image forgery
            np.random.uniform(0.6, 1.0),  # High consistency
            np.random.uniform(0, 0.3)     # Low OCR threat
        ])
        labels.append(0)

        # Malicious samples
        threat_type = np.random.choice(['text', 'image', 'mismatch', 'ocr'])
        if threat_type == 'text':
            features.append([
                np.random.uniform(0.7, 1.0),
                np.random.uniform(0.1, 0.4),
                np.random.uniform(0.5, 0.8),
                np.random.uniform(0.1, 0.3)
            ])
        elif threat_type == 'image':
            features.append([
                np.random.uniform(0.1, 0.3),
                np.random.uniform(0.7, 1.0),
                np.random.uniform(0.3, 0.6),
                np.random.uniform(0.1, 0.3)
            ])
        elif threat_type == 'mismatch':
            features.append([
                np.random.uniform(0.1, 0.4),
                np.random.uniform(0.3, 0.6),
                np.random.uniform(0.1, 0.3),
                np.random.uniform(0.1, 0.3)
            ])
        else:  # ocr
            features.append([
                np.random.uniform(0.1, 0.4),
                np.random.uniform(0.2, 0.5),
                np.random.uniform(0.4, 0.7),
                np.random.uniform(0.6, 1.0)
            ])
        labels.append(1)

    features = np.array(features)
    labels = np.array(labels)

    # Train
    fusion.train(features, labels)

    # Test after training
    print("\nTest 4: After ML training - Edge case")
    result = fusion.predict(
        text_score=0.55,    # Moderate scores
        image_score=0.48,
        clip_score=0.45,
        ocr_score=0.42
    )
    print(generate_report(result))

    print("\n[SUCCESS] Fusion model test completed!")
