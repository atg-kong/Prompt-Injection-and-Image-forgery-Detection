# Chapter 6: System Architecture

## 6.1 Overview

Our system follows a **modular architecture** where each component handles a specific task. The modules are loosely coupled, meaning they can work independently but are connected through a central pipeline. This makes the system easy to understand, test, and modify.

## 6.2 High-Level Architecture

The system consists of five main layers:

1. **Input Layer** - Receives text and/or image input
2. **Processing Layer** - Individual analysis modules
3. **Feature Extraction Layer** - Extracts relevant features
4. **Fusion Layer** - Combines all results
5. **Output Layer** - Provides final decision and report

```
┌─────────────────────────────────────────────┐
│              INPUT LAYER                     │
│    (Text Input + Image Input + Caption)      │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│           PROCESSING LAYER                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │
│  │  Text   │  │  Image  │  │     OCR     │ │
│  │ Module  │  │ Module  │  │   Module    │ │
│  └─────────┘  └─────────┘  └─────────────┘ │
│                                              │
│           ┌─────────────┐                   │
│           │    CLIP     │                   │
│           │   Module    │                   │
│           └─────────────┘                   │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         FEATURE EXTRACTION LAYER            │
│   (Scores, Probabilities, Embeddings)       │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│            FUSION LAYER                      │
│   (Logistic Regression + Rule-Based Logic)  │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│            OUTPUT LAYER                      │
│   (Final Decision + Confidence + Report)    │
└─────────────────────────────────────────────┘
```

## 6.3 Detailed Module Architecture

### 6.3.1 Text Processing Module

**Purpose:** Detect prompt injection in text input

**Components:**
```
Text Input
    │
    ▼
┌──────────────────┐
│  Preprocessing   │
│  - Lowercase     │
│  - Clean text    │
│  - Tokenize      │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  BERT Encoder    │
│  (768-dim output)│
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Classification   │
│     Head         │
│  (768→256→2)     │
└──────────────────┘
    │
    ▼
Injection Probability
```

**Input:** Raw text string
**Output:** Probability of prompt injection (0.0 to 1.0)

**Model Details:**
- Base: bert-base-uncased (110M parameters)
- Custom layers: 2 fully connected layers
- Activation: ReLU
- Output: Softmax for probability

### 6.3.2 Image Processing Module

**Purpose:** Detect image forgery/manipulation

**Components:**
```
Image Input
    │
    ▼
┌──────────────────┐
│  Preprocessing   │
│  - Resize 224x224│
│  - Normalize     │
│  - ToTensor      │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│   EfficientNet   │
│      B0          │
│  (1280-dim feat) │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Classification   │
│     Head         │
│ (1280→512→2)    │
└──────────────────┘
    │
    ▼
Forgery Probability
```

**Input:** Image file (any size)
**Output:** Probability of forgery (0.0 to 1.0)

**Model Details:**
- Base: EfficientNet-B0 (5.3M parameters)
- Pre-trained on ImageNet
- Custom classifier head
- Output: Binary classification

### 6.3.3 OCR Module

**Purpose:** Extract text hidden in images

**Components:**
```
Image Input
    │
    ▼
┌──────────────────┐
│  Preprocessing   │
│  - Grayscale     │
│  - Threshold     │
│  - Denoise       │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│   Pytesseract    │
│   OCR Engine     │
└──────────────────┘
    │
    ▼
Extracted Text
    │
    ▼
┌──────────────────┐
│ Text Analysis    │
│ (Check for       │
│  injections)     │
└──────────────────┘
```

**Input:** Image file
**Output:** Extracted text + analysis results

**Process:**
1. Convert image to grayscale
2. Apply adaptive thresholding
3. Run OCR with Tesseract
4. Clean extracted text
5. Optionally analyze for injections

### 6.3.4 CLIP Module

**Purpose:** Check text-image consistency

**Components:**
```
Image Input          Text Caption
    │                     │
    ▼                     ▼
┌──────────┐        ┌──────────┐
│   CLIP   │        │   CLIP   │
│  Image   │        │   Text   │
│  Encoder │        │  Encoder │
└──────────┘        └──────────┘
    │                     │
    ▼                     ▼
Image Embedding     Text Embedding
    │                     │
    └─────────┬───────────┘
              │
              ▼
       ┌──────────────┐
       │   Cosine     │
       │  Similarity  │
       └──────────────┘
              │
              ▼
      Similarity Score
```

**Input:** Image + text description
**Output:** Similarity score (0.0 to 1.0)

**Model Details:**
- Uses OpenAI CLIP (ViT-B/32)
- Projects both modalities to same space
- Cosine similarity measures match
- High score = good match

### 6.3.5 Fusion Module

**Purpose:** Combine all results for final decision

**Two-Stage Fusion:**

```
        Input Features
              │
    ┌─────────┴──────────┬───────────────┐
    │                    │               │
Text Score          Image Score      CLIP Score
    │                    │               │
    └─────────┬──────────┴───────────────┘
              │
              ▼
┌─────────────────────────┐
│   Logistic Regression   │
│    (ML-based fusion)    │
└─────────────────────────┘
              │
              ▼
         ML Prediction
              │
              ▼
┌─────────────────────────┐
│   Rule-Based Override   │
│   - If any score > 0.9  │
│   - If CLIP < 0.2       │
│   - Special patterns    │
└─────────────────────────┘
              │
              ▼
       Final Decision
```

**Input Features:**
- Text injection probability
- Image forgery probability
- CLIP similarity score
- OCR text injection score (if text found)

**Output:** Final binary classification + confidence

## 6.4 Data Flow Diagram

Refer to `diagrams/data_flow_diagram.mmd` for the complete DFD.

**Summary:**
1. User provides input (text, image, or both)
2. Input is validated and preprocessed
3. Each module processes its respective input
4. Results are stored in intermediate data stores
5. Fusion layer collects all results
6. Final decision is made and reported to user

## 6.5 Class Diagram

```
┌────────────────────────┐
│   MultimodalDetector   │
├────────────────────────┤
│ - text_model           │
│ - image_model          │
│ - ocr_module           │
│ - clip_module          │
│ - fusion_model         │
├────────────────────────┤
│ + analyze(text, image) │
│ + load_models()        │
│ + generate_report()    │
└────────────────────────┘
          │
    ┌─────┴─────┬─────────┬──────────┐
    │           │         │          │
    ▼           ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│TextDet.│ │ImageDet│ │OCRMod. │ │CLIPMod.│
└────────┘ └────────┘ └────────┘ └────────┘
```

## 6.6 Sequence Diagram

**Typical Analysis Flow:**

```
User          System        TextModel     ImageModel    CLIP      Fusion
  │              │              │             │          │          │
  │─ Input ─────>│              │             │          │          │
  │              │─ Text ──────>│             │          │          │
  │              │<── Score ────│             │          │          │
  │              │─── Image ────────────────->│          │          │
  │              │<──── Score ────────────────│          │          │
  │              │──── Image+Text ───────────────────────>│          │
  │              │<───── Score ──────────────────────────│          │
  │              │────────────────────All Scores──────────────────>│
  │              │<───────────────────Final Decision──────────────│
  │<─ Report ────│              │             │          │          │
  │              │              │             │          │          │
```

## 6.7 Deployment Architecture

For our academic project, we use a simple local deployment:

```
┌─────────────────────────────────────┐
│          Local Machine               │
│  ┌─────────────────────────────┐    │
│  │      Python Runtime         │    │
│  │  ┌─────────────────────┐   │    │
│  │  │  Detection System   │   │    │
│  │  │   (All Modules)     │   │    │
│  │  └─────────────────────┘   │    │
│  └─────────────────────────────┘    │
│                                      │
│  ┌─────────────────────────────┐    │
│  │    Model Files (.pt/.bin)   │    │
│  └─────────────────────────────┘    │
│                                      │
│  ┌─────────────────────────────┐    │
│  │    Data Files (CSV/Images)  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

## 6.8 Security Considerations

Even though this is an academic project, we considered:

1. **Input Validation:** Check for malformed inputs
2. **File Safety:** Only accept specific image formats
3. **No External Calls:** All processing is local
4. **No Data Storage:** We don't store user inputs
5. **Model Safety:** Pre-trained models from trusted sources

## 6.9 Design Patterns Used

1. **Factory Pattern:** For creating different model instances
2. **Strategy Pattern:** For different fusion strategies
3. **Pipeline Pattern:** For sequential processing
4. **Singleton Pattern:** For model loading (load once, use multiple times)

## 6.10 Trade-offs and Design Decisions

| Decision | Alternative | Why We Chose |
|----------|-------------|--------------|
| Late Fusion | Early/Hybrid Fusion | Simpler, more interpretable |
| EfficientNet-B0 | ResNet, VGG | Better accuracy/size ratio |
| BERT-base | RoBERTa, DistilBERT | Well-documented, good performance |
| Logistic Regression | Neural Network | Transparent, less overfitting |
| Local Processing | Cloud API | Privacy, no internet needed |

## 6.11 Scalability Considerations

Our current system is designed for single-user, batch processing. For scaling (future work):

- **Horizontal Scaling:** Multiple instances
- **Batch Processing:** GPU batching for speed
- **Caching:** Store model predictions
- **API Server:** REST API for multiple clients

## 6.12 Summary

Our system architecture is designed to be:
- **Modular:** Easy to understand and modify
- **Extensible:** Can add new modules easily
- **Interpretable:** Can see why decisions are made
- **Efficient:** Uses pre-trained models for fast development
- **Academic-Friendly:** Suitable for learning and teaching

The architecture diagrams (system workflow, architecture, and data flow) are available in the `diagrams/` folder in Mermaid format.
