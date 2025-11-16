# Chapter 7: Algorithms Used

## 7.1 Introduction

This chapter explains the key algorithms and models we used in our project. We tried to explain each algorithm in simple terms so that students can understand them easily.

## 7.2 BERT (Bidirectional Encoder Representations from Transformers)

### 7.2.1 What is BERT?

BERT is a pre-trained language model developed by Google in 2018. It revolutionized natural language processing (NLP) by learning to understand text in both directions (left-to-right and right-to-left).

### 7.2.2 How BERT Works

**Key Concepts:**

1. **Transformer Architecture:** BERT uses the Transformer architecture which uses attention mechanisms to understand relationships between words.

2. **Bidirectional Learning:** Unlike traditional models that read text in one direction, BERT reads in both directions simultaneously.

3. **Pre-training:** BERT is pre-trained on a large corpus of text (Wikipedia + Books) using two tasks:
   - **Masked Language Model (MLM):** Predict missing words
   - **Next Sentence Prediction (NSP):** Predict if sentences follow each other

4. **Fine-tuning:** We take the pre-trained BERT and train it further on our specific task (prompt injection detection).

### 7.2.3 BERT Architecture

```
Input: [CLS] This is a prompt [SEP]
         │     │   │  │   │    │
         ▼     ▼   ▼  ▼   ▼    ▼
     ┌─────────────────────────────┐
     │     Token Embeddings        │
     └─────────────────────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │   Segment Embeddings        │
     └─────────────────────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │   Position Embeddings       │
     └─────────────────────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │   Transformer Encoder       │
     │    (12 layers × 12 heads)   │
     └─────────────────────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │    [CLS] Token Output       │
     │      (768 dimensions)       │
     └─────────────────────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │    Classification Head      │
     │       (768 → 2)             │
     └─────────────────────────────┘
```

### 7.2.4 Self-Attention Mechanism

The key innovation in BERT is **self-attention**, which calculates how much each word should attend to every other word.

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q = Query matrix
- K = Key matrix
- V = Value matrix
- d_k = dimension of keys

**Simple Explanation:**
- For each word, BERT asks "How relevant are other words to understanding this word?"
- It assigns weights to all words based on relevance
- This allows understanding context better

### 7.2.5 Why BERT for Prompt Injection?

We chose BERT because:
1. Understands context very well
2. Can detect subtle patterns in text
3. Pre-trained on vast amounts of text
4. Easy to fine-tune for classification
5. Good performance on small datasets

### 7.2.6 Our BERT Configuration

```python
Configuration:
- Model: bert-base-uncased
- Parameters: 110 million
- Hidden Size: 768
- Attention Heads: 12
- Layers: 12
- Max Sequence Length: 512
```

## 7.3 EfficientNet

### 7.3.1 What is EfficientNet?

EfficientNet is a family of convolutional neural networks developed by Google in 2019. It achieves better accuracy with fewer parameters by using **compound scaling**.

### 7.3.2 The Problem EfficientNet Solves

Traditional CNN scaling methods:
- Increase depth (more layers)
- Increase width (more channels)
- Increase resolution (larger images)

But doing only one doesn't give best results.

**EfficientNet's Solution:** Scale all three dimensions together in a balanced way.

### 7.3.3 Compound Scaling

```
depth = α^φ
width = β^φ
resolution = γ^φ

Where: α × β² × γ² ≈ 2
```

For EfficientNet-B0:
- α = 1.2 (depth)
- β = 1.1 (width)
- γ = 1.15 (resolution)

This gives optimal scaling for each model variant (B0 to B7).

### 7.3.4 MBConv Block (Mobile Inverted Bottleneck)

The basic building block of EfficientNet:

```
Input Features
      │
      ▼
┌─────────────┐
│  1x1 Conv   │
│  (Expand)   │
└─────────────┘
      │
      ▼
┌─────────────┐
│  Depthwise  │
│    Conv     │
└─────────────┘
      │
      ▼
┌─────────────┐
│  Squeeze &  │
│  Excitation │
└─────────────┘
      │
      ▼
┌─────────────┐
│  1x1 Conv   │
│  (Project)  │
└─────────────┘
      │
      ▼
┌─────────────┐
│  Skip Conn. │
│  (Add Input)│
└─────────────┘
      │
      ▼
Output Features
```

### 7.3.5 Why EfficientNet for Image Forgery?

1. **Efficient:** Good accuracy with fewer parameters
2. **Pre-trained:** Trained on ImageNet (14M images)
3. **Feature Rich:** Learns hierarchical features
4. **Scalable:** Can choose B0-B7 based on resources
5. **Fast:** Faster inference than other models

### 7.3.6 Our EfficientNet Configuration

```python
Configuration:
- Model: EfficientNet-B0
- Parameters: 5.3 million
- Input Size: 224 × 224 × 3
- Output Features: 1280
- Classifier: 1280 → 512 → 2
```

## 7.4 CLIP (Contrastive Language-Image Pre-training)

### 7.4.1 What is CLIP?

CLIP is a model by OpenAI that learns to understand images and text together. It can measure how well an image matches a text description.

### 7.4.2 How CLIP is Trained

CLIP was trained on 400 million (image, text) pairs from the internet using **contrastive learning**.

```
Training Objective:
- Maximize similarity of correct (image, text) pairs
- Minimize similarity of incorrect pairs
```

### 7.4.3 CLIP Architecture

```
         Image                    Text
           │                       │
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │   Vision    │         │    Text     │
    │  Transformer│         │ Transformer │
    │   (ViT)     │         │             │
    └─────────────┘         └─────────────┘
           │                       │
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │  Projection │         │  Projection │
    │    Layer    │         │    Layer    │
    └─────────────┘         └─────────────┘
           │                       │
           ▼                       ▼
    Image Embedding           Text Embedding
    (512-dim)                  (512-dim)
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
                ┌─────────────┐
                │   Cosine    │
                │  Similarity │
                └─────────────┘
                       │
                       ▼
                 Score (0 to 1)
```

### 7.4.4 Cosine Similarity

Formula:
```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

Where:
- A = Image embedding vector
- B = Text embedding vector
- · = Dot product
- ||.|| = Magnitude

**Interpretation:**
- Score close to 1.0 = High similarity (text matches image)
- Score close to 0.0 = Low similarity (text doesn't match image)

### 7.4.5 Why CLIP for Our Project?

1. Detects mismatched text-image pairs
2. Zero-shot capability (no training needed)
3. Robust to different image types
4. Understands semantic meaning
5. Easy to integrate

## 7.5 OCR - Tesseract Algorithm

### 7.5.1 What is OCR?

Optical Character Recognition (OCR) converts images of text into machine-readable text.

### 7.5.2 Tesseract Process

```
Image Input
      │
      ▼
┌─────────────┐
│  Adaptive   │
│ Thresholding│
└─────────────┘
      │
      ▼
┌─────────────┐
│   Line      │
│  Detection  │
└─────────────┘
      │
      ▼
┌─────────────┐
│   Word      │
│ Segmentation│
└─────────────┘
      │
      ▼
┌─────────────┐
│  Character  │
│ Recognition │
│   (LSTM)    │
└─────────────┘
      │
      ▼
┌─────────────┐
│  Language   │
│   Model     │
└─────────────┘
      │
      ▼
  Text Output
```

### 7.5.3 Key Steps

1. **Preprocessing:** Convert to grayscale, remove noise
2. **Binarization:** Convert to black and white
3. **Layout Analysis:** Find text regions
4. **Character Recognition:** Identify each character using neural network
5. **Post-processing:** Apply language model to correct errors

## 7.6 Fusion - Logistic Regression

### 7.6.1 What is Logistic Regression?

Logistic Regression is a simple machine learning algorithm for binary classification. It predicts the probability of an input belonging to a class.

### 7.6.2 Logistic Function (Sigmoid)

```
P(y=1|x) = 1 / (1 + e^(-z))

where z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

### 7.6.3 In Our Fusion Model

Input features (x):
- x₁ = Text injection probability
- x₂ = Image forgery probability
- x₃ = CLIP similarity score
- x₄ = OCR injection probability (if available)

Output:
- P(malicious) = σ(w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b)

### 7.6.4 Why Logistic Regression?

1. Simple and interpretable
2. Fast to train and predict
3. Good for combining probabilities
4. Low risk of overfitting
5. Weights show feature importance

## 7.7 Cross-Entropy Loss Function

Used for training both text and image models:

```
L = -1/N Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

Where:
- N = Number of samples
- yᵢ = True label (0 or 1)
- pᵢ = Predicted probability

This loss function penalizes wrong predictions, helping the model learn correct patterns.

## 7.8 Optimization Algorithms

### 7.8.1 AdamW (for BERT)

Adam with Weight Decay:
- Combines momentum and adaptive learning rates
- Helps prevent overfitting
- Works well with transformers

```
m = β₁m + (1-β₁)g
v = β₂v + (1-β₂)g²
w = w - lr × m / √v - lr × wd × w
```

### 7.8.2 Adam (for EfficientNet)

Standard Adam optimizer:
- Good for CNNs
- Adaptive learning rates
- Fast convergence

## 7.9 Evaluation Metrics

### 7.9.1 Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### 7.9.2 Precision

```
Precision = TP / (TP + FP)
```

How many predicted positives are actually positive?

### 7.9.3 Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

How many actual positives did we find?

### 7.9.4 F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Harmonic mean of precision and recall.

## 7.10 Summary

This chapter explained the key algorithms:

1. **BERT** - For understanding text and detecting injections
2. **EfficientNet** - For analyzing images and detecting forgery
3. **CLIP** - For checking text-image consistency
4. **Tesseract OCR** - For extracting text from images
5. **Logistic Regression** - For fusing all results

Each algorithm was chosen for its specific strengths and suitability for our academic project.
