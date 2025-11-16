# Chapter 5: System Requirements

## 5.1 Hardware Requirements

### 5.1.1 Minimum Requirements (For Running Inference Only)

| Component | Specification |
|-----------|--------------|
| Processor | Intel Core i5 or equivalent |
| RAM | 8 GB |
| Storage | 10 GB free space |
| GPU | Not required (CPU inference possible) |

### 5.1.2 Recommended Requirements (For Training)

| Component | Specification |
|-----------|--------------|
| Processor | Intel Core i7 or AMD Ryzen 7 |
| RAM | 16 GB or more |
| Storage | 50 GB SSD |
| GPU | NVIDIA GTX 1080 or better (8GB+ VRAM) |
| GPU Alternative | Google Colab (Free GPU) |

### 5.1.3 What We Used

For our project development, we used:

- **Laptop:** Dell Inspiron 15
  - Processor: Intel Core i7-10750H
  - RAM: 16 GB DDR4
  - Storage: 512 GB SSD
  - GPU: NVIDIA GTX 1650 (4GB)

- **Cloud Computing:** Google Colab Pro
  - Used for training deep learning models
  - Free GPU (Tesla T4 or K80)
  - 12 GB GPU RAM

## 5.2 Software Requirements

### 5.2.1 Operating System

| OS | Version | Status |
|----|---------|--------|
| Windows | 10/11 | Supported |
| Ubuntu | 18.04+ | Recommended |
| macOS | 10.15+ | Supported |
| Google Colab | - | Used for training |

We developed on **Ubuntu 20.04 LTS**.

### 5.2.2 Programming Language

- **Python 3.8 or higher**
  - Object-oriented programming
  - Rich ecosystem for ML/DL
  - Easy to learn and use

### 5.2.3 Required Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| torch | 1.10.0+ | Deep learning framework |
| torchvision | 0.11.0+ | Computer vision utilities |
| transformers | 4.20.0+ | BERT and pre-trained models |
| efficientnet-pytorch | 0.7.0+ | EfficientNet model |
| clip | latest | OpenAI CLIP model |
| pytesseract | 0.3.8+ | OCR engine |
| scikit-learn | 1.0.0+ | ML utilities and metrics |
| pandas | 1.3.0+ | Data manipulation |
| numpy | 1.21.0+ | Numerical computing |
| matplotlib | 3.5.0+ | Plotting and visualization |
| seaborn | 0.11.0+ | Statistical visualization |
| pillow | 8.0.0+ | Image processing |
| opencv-python | 4.5.0+ | Advanced image processing |
| tqdm | 4.60.0+ | Progress bars |
| jupyter | 1.0.0+ | Interactive notebooks |

### 5.2.4 External Software

| Software | Version | Purpose |
|----------|---------|---------|
| Tesseract OCR | 4.1+ | Backend for pytesseract |
| Git | 2.30+ | Version control |
| CUDA Toolkit | 11.0+ | GPU acceleration (optional) |
| cuDNN | 8.0+ | Deep learning on GPU (optional) |

### 5.2.5 Installation Commands

**For Ubuntu/Linux:**
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr

# Install Python libraries
pip install torch torchvision
pip install transformers
pip install efficientnet-pytorch
pip install git+https://github.com/openai/CLIP.git
pip install pytesseract
pip install scikit-learn pandas numpy matplotlib seaborn
pip install pillow opencv-python tqdm jupyter
```

**For Windows:**
```bash
# Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH

# Install Python libraries (same as above)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers efficientnet-pytorch
# ... rest same as Linux
```

## 5.3 Functional Requirements

These are the features our system must have:

### FR1: Text Input Processing
- **Description:** System must accept text input for analysis
- **Input:** String of text (up to 512 tokens)
- **Output:** Classification (Injection / No Injection)

### FR2: Image Input Processing
- **Description:** System must accept image input for analysis
- **Input:** Image file (JPG, PNG, BMP)
- **Output:** Classification (Forged / Authentic)

### FR3: OCR Text Extraction
- **Description:** System must extract text from images
- **Input:** Image file
- **Output:** Extracted text string

### FR4: Text-Image Consistency Check
- **Description:** System must verify if text matches image content
- **Input:** Image and text description
- **Output:** Similarity score (0 to 1)

### FR5: Fusion Decision
- **Description:** System must combine all results for final decision
- **Input:** Results from all modules
- **Output:** Final classification (Safe / Malicious)

### FR6: Result Reporting
- **Description:** System must generate detailed report
- **Input:** All analysis results
- **Output:** Formatted report with scores and reasoning

## 5.4 Non-Functional Requirements

### NFR1: Performance
- Text analysis should complete within 2 seconds
- Image analysis should complete within 5 seconds
- Total pipeline should complete within 10 seconds

### NFR2: Accuracy
- Text model accuracy > 85%
- Image model accuracy > 80%
- Fusion model accuracy > 85%

### NFR3: Usability
- Clear error messages
- Simple command-line interface
- Well-documented code

### NFR4: Maintainability
- Modular code structure
- Proper comments
- Easy to extend

### NFR5: Reliability
- Should handle invalid inputs gracefully
- Should not crash on edge cases
- Should provide consistent results

### NFR6: Portability
- Should run on multiple operating systems
- Should work with different Python versions (3.8+)

## 5.5 Data Requirements

### 5.5.1 Training Data

**Text Dataset:**
- Minimum 500 safe text samples
- Minimum 500 injection attempt samples
- Each sample: 10-500 words

**Image Dataset:**
- Minimum 250 authentic images
- Minimum 250 forged images
- Image size: Any (will be resized to 224x224)
- Format: JPG, PNG, BMP

### 5.5.2 Storage Requirements

| Data Type | Size Estimate |
|-----------|--------------|
| Text Dataset | ~10 MB |
| Image Dataset | ~500 MB |
| Pre-trained Models | ~2 GB |
| Trained Models | ~500 MB |
| Logs and Results | ~100 MB |
| **Total** | **~3-4 GB** |

## 5.6 Interface Requirements

### 5.6.1 User Interface

Our system provides:
- Command-line interface (CLI)
- Jupyter notebook interface
- Python API for integration

**CLI Example:**
```bash
python main.py --text "Your text here" --image "path/to/image.jpg"
```

**Output:**
```
=== Detection Results ===
Text Injection Score: 0.85
Image Forgery Score: 0.23
CLIP Similarity: 0.91
Final Decision: MALICIOUS (Prompt Injection Detected)
```

### 5.6.2 API Interface

```python
from detector import MultimodalDetector

detector = MultimodalDetector()
result = detector.analyze(text="...", image_path="...")
print(result.is_safe)
print(result.details)
```

## 5.7 Constraints

1. **Limited GPU memory** - Cannot use very large batch sizes
2. **Internet required** - For downloading pre-trained models
3. **English only** - System trained on English text
4. **Image size limit** - Very large images need resizing
5. **No real-time processing** - Batch processing only

## 5.8 Dependencies

External services/resources required:

1. **Hugging Face Model Hub** - For downloading BERT model
2. **PyTorch Model Zoo** - For EfficientNet weights
3. **OpenAI CLIP** - For text-image similarity
4. **Tesseract OCR Engine** - For text extraction from images

## 5.9 Summary

This chapter outlined all hardware, software, functional, and non-functional requirements for our project. We ensured that our requirements are realistic and achievable within the 6-month timeframe and available resources. The system is designed to be portable and can run on modest hardware, making it suitable for academic purposes.
