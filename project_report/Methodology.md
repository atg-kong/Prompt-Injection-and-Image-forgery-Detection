# Chapter 4: Methodology

## 4.1 Overview of Our Approach

Our methodology follows a systematic approach to solve the problem of detecting prompt injection and image forgery. We divided our work into clear phases that align with the 6-month project timeline.

**High-Level Methodology:**

1. Data Collection and Generation
2. Preprocessing and Preparation
3. Individual Model Development
4. Integration and Fusion
5. Testing and Evaluation
6. Documentation and Presentation

## 4.2 Development Timeline

### Month 1: Research and Planning

**Activities:**
- Studied research papers on prompt injection and image forgery
- Explored available datasets
- Set up development environment
- Installed required libraries (PyTorch, Transformers, OpenCV)
- Created project repository structure

**What We Learned:**
- Understood the problem domain thoroughly
- Realized that public datasets are scarce
- Decided to create synthetic datasets

**Challenges Faced:**
- Initially overwhelmed by the complexity
- Had to narrow down scope to manageable size

### Month 2: Dataset Creation

**Activities:**
- Created synthetic prompt injection dataset (1000 samples)
- Created synthetic image forgery dataset (500 images)
- Implemented data augmentation techniques
- Split data into train/validation/test sets

**Prompt Injection Dataset:**
- 500 safe examples (normal prompts)
- 500 injection examples (malicious prompts)
- Covered various injection techniques

**Image Forgery Dataset:**
- 250 original images
- 250 forged images (copy-move, splicing, text manipulation)
- Used PIL and OpenCV for creating forgeries

**What We Learned:**
- Creating realistic datasets is challenging
- Need diversity in examples for better generalization

**Challenges Faced:**
- Ensuring dataset is balanced
- Making forgeries realistic enough
- Avoiding overfitting to specific patterns

### Month 3: Text Model Development

**Activities:**
- Implemented BERT-based text classifier
- Created text preprocessing pipeline
- Trained the model for prompt injection detection
- Performed hyperparameter tuning
- Logged training metrics

**Model Architecture:**
```
BERT Base (Pretrained)
    ↓
Dropout (0.3)
    ↓
Linear Layer (768 → 256)
    ↓
ReLU Activation
    ↓
Linear Layer (256 → 2)
    ↓
Softmax Output
```

**Training Details:**
- Optimizer: AdamW
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 10
- Loss Function: Cross Entropy

**Results Achieved:**
- Training Accuracy: 93.5%
- Validation Accuracy: 91.2%
- F1-Score: 0.89

**What We Learned:**
- Pre-trained models are very powerful
- Fine-tuning is easier than training from scratch
- Regularization (dropout) helps prevent overfitting

**Challenges Faced:**
- GPU memory constraints with larger batch sizes
- Choosing the right learning rate
- Preventing overfitting on small dataset

### Month 4: Image Model Development

**Activities:**
- Implemented EfficientNet-B0 for image classification
- Created image preprocessing pipeline
- Added OCR module using Pytesseract
- Trained image forgery detector
- Improved dataset with more diverse examples

**EfficientNet Architecture:**
```
EfficientNet-B0 (Pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dropout (0.2)
    ↓
Linear Layer (1280 → 512)
    ↓
ReLU Activation
    ↓
Linear Layer (512 → 2)
    ↓
Softmax Output
```

**OCR Module:**
- Uses Pytesseract for text extraction
- Preprocessing: grayscale conversion, thresholding
- Extracts text from images for further analysis

**Training Details:**
- Optimizer: Adam
- Learning Rate: 1e-4
- Batch Size: 32
- Epochs: 15
- Image Size: 224x224

**Results Achieved:**
- Training Accuracy: 89.8%
- Validation Accuracy: 87.5%
- F1-Score: 0.86

**What We Learned:**
- Transfer learning is very effective
- Image augmentation helps generalization
- OCR is useful but not always accurate

**Challenges Faced:**
- Training time is longer than text models
- Need more diverse forgery examples
- OCR struggles with low-quality images

### Month 5: CLIP and Fusion Model

**Activities:**
- Integrated CLIP for text-image similarity
- Developed fusion model combining all signals
- Experimented with different fusion strategies
- Performed cross-validation
- Compared individual vs combined performance

**CLIP Integration:**
```python
# Pseudo-code for CLIP similarity
image_features = clip_model.encode_image(image)
text_features = clip_model.encode_text(caption)
similarity = cosine_similarity(image_features, text_features)
```

**Fusion Model Architecture:**

We tried two approaches:

**Approach 1: Rule-Based Fusion**
```
If text_injection_prob > 0.7 → MALICIOUS
Else if image_forgery_prob > 0.7 → MALICIOUS
Else if clip_similarity < 0.3 → SUSPICIOUS
Else → SAFE
```

**Approach 2: Logistic Regression Fusion**
```
Input Features:
- text_injection_probability
- image_forgery_probability
- clip_similarity_score
- ocr_text_injection_probability

Output: Binary classification (Safe/Malicious)
```

**Final Choice:** Combined approach using both rule-based and ML

**Results Achieved:**
- Fusion Model Accuracy: 89.3%
- Better than individual models alone
- Good balance between precision and recall

**What We Learned:**
- Combining multiple signals improves robustness
- Rule-based logic adds interpretability
- CLIP similarity is a good indicator of mismatched content

**Challenges Faced:**
- Deciding optimal thresholds for rules
- Handling cases where models disagree
- Balancing false positives and false negatives

### Month 6: Evaluation and Documentation

**Activities:**
- Final testing on held-out test set
- Generated confusion matrices
- Created performance comparison tables
- Wrote complete documentation
- Prepared presentation and demo
- Created visualizations and graphs

**Final Evaluation Metrics:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Text Model | 91.2% | 0.90 | 0.92 | 0.89 |
| Image Model | 87.5% | 0.85 | 0.89 | 0.86 |
| Fusion Model | 89.3% | 0.88 | 0.91 | 0.89 |

## 4.3 Data Preprocessing

### 4.3.1 Text Preprocessing

```python
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Tokenize using BERT tokenizer
    tokens = tokenizer(text, max_length=512,
                      padding='max_length',
                      truncation=True)
    return tokens
```

### 4.3.2 Image Preprocessing

```python
def preprocess_image(image_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    # Resize
    img = img.resize((224, 224))
    # Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(img)
```

## 4.4 Model Training Procedure

### 4.4.1 General Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_accuracy = evaluate(model, val_loader)

    # Save best model
    if val_accuracy > best_accuracy:
        save_model(model)
```

### 4.4.2 Hyperparameter Selection

We used a simple grid search for hyperparameters:

| Hyperparameter | Values Tried | Best Value |
|----------------|--------------|------------|
| Learning Rate | 1e-3, 1e-4, 2e-5 | 2e-5 (text), 1e-4 (image) |
| Batch Size | 8, 16, 32 | 16 (text), 32 (image) |
| Dropout | 0.1, 0.2, 0.3 | 0.3 (text), 0.2 (image) |
| Epochs | 5, 10, 15, 20 | 10 (text), 15 (image) |

## 4.5 Evaluation Methodology

### 4.5.1 Metrics Used

1. **Accuracy:** Overall correct predictions
2. **Precision:** True positives / (True positives + False positives)
3. **Recall:** True positives / (True positives + False negatives)
4. **F1-Score:** Harmonic mean of precision and recall
5. **Confusion Matrix:** Detailed breakdown of predictions

### 4.5.2 Cross-Validation

We used 5-fold cross-validation:
- Split data into 5 parts
- Train on 4 parts, validate on 1 part
- Repeat 5 times
- Average the results

This ensures our results are reliable and not dependent on a specific split.

## 4.6 Tools and Libraries Used

| Tool/Library | Version | Purpose |
|--------------|---------|---------|
| Python | 3.8+ | Programming language |
| PyTorch | 1.10+ | Deep learning framework |
| Transformers | 4.20+ | Pre-trained BERT model |
| EfficientNet-PyTorch | 0.7+ | EfficientNet model |
| Pytesseract | 0.3+ | OCR |
| OpenAI CLIP | - | Text-image similarity |
| Scikit-learn | 1.0+ | Fusion model, metrics |
| Pandas | 1.3+ | Data handling |
| Matplotlib | 3.5+ | Visualization |
| PIL | 8.0+ | Image processing |
| OpenCV | 4.5+ | Image manipulation |

## 4.7 Code Organization

```
src/
├── text_model/
│   ├── model.py           # BERT classifier definition
│   ├── train_text.py      # Training script
│   └── text_dataset.py    # Dataset class
├── image_model/
│   ├── model.py           # EfficientNet classifier
│   ├── train_image.py     # Training script
│   └── image_dataset.py   # Dataset class
├── ocr_module/
│   └── ocr_extract.py     # OCR functionality
├── clip_checker/
│   └── clip_module.py     # CLIP similarity
├── fusion_model/
│   └── fusion.py          # Final fusion logic
└── utils/
    ├── metrics.py         # Evaluation metrics
    └── helpers.py         # Helper functions
```

## 4.8 Testing Strategy

### 4.8.1 Unit Testing
- Tested each module independently
- Verified preprocessing functions
- Checked model outputs

### 4.8.2 Integration Testing
- Tested complete pipeline end-to-end
- Verified data flow between modules
- Checked fusion layer logic

### 4.8.3 Performance Testing
- Measured inference time
- Checked memory usage
- Tested with different input sizes

## 4.9 Mistakes We Made and Corrections

Throughout the project, we made several mistakes that we corrected:

1. **Initially forgot to normalize images** → Added proper normalization
2. **Used too high learning rate** → Reduced to prevent divergence
3. **Didn't balance the dataset** → Added class weights
4. **Overfitted on training data** → Added dropout and early stopping
5. **OCR was too slow** → Added caching for repeated images
6. **CLIP similarity threshold was wrong** → Tuned based on validation set

## 4.10 Summary

Our methodology followed a structured 6-month approach:
- Month 1: Research and planning
- Month 2: Dataset creation
- Month 3: Text model development
- Month 4: Image model development
- Month 5: Integration and fusion
- Month 6: Evaluation and documentation

We used pre-trained models (BERT, EfficientNet, CLIP) and fine-tuned them for our specific task. Our fusion approach combines rule-based logic with machine learning for robust detection.
