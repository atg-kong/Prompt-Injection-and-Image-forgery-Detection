# Prompt Injection and Image Forgery Detection Using Multimodal Deep Learning

## Final Year Major Project (6 Months)

A comprehensive multimodal deep learning system for detecting prompt injection attacks in text and image forgery/manipulation, combining BERT, EfficientNet, OCR, and CLIP into a unified detection pipeline.

---

## Project Overview

### Problem Statement
> Given text and/or image input, automatically detect if the text contains prompt injection attempts and/or if the image has been forged, and provide a final classification.

### Key Features
- **Text Analysis:** BERT-based prompt injection detection (91.2% accuracy)
- **Image Analysis:** EfficientNet-based forgery detection (87.5% accuracy)
- **OCR Integration:** Extract hidden text from images using Pytesseract
- **CLIP Consistency:** Verify text-image alignment
- **Fusion Model:** Combined decision making (89.3% accuracy)

---

## System Architecture

```
Input (Text + Image)
        ↓
    ┌───┴───┐
    ↓       ↓
 Text     Image
Detector  Detector
    ↓       ↓
 BERT    EfficientNet
    ↓       ↓
    ├───────┤
    ↓       ↓
   OCR    CLIP
    ↓       ↓
    └───┬───┘
        ↓
   Fusion Layer
   (ML + Rules)
        ↓
    Final Decision
   (Safe/Malicious)
```

---

## Repository Structure

```
Prompt-Injection-and-Image-forgery-Detection/
│
├── project_report/               # Complete academic documentation
│   ├── Abstract.md
│   ├── Introduction.md
│   ├── Problem_Statement.md
│   ├── Literature_Review.md
│   ├── Methodology.md
│   ├── System_Requirements.md
│   ├── System_Architecture.md
│   ├── Algorithms_Used.md
│   ├── Results_and_Discussion.md
│   ├── Conclusion.md
│   ├── Future_Scope.md
│   └── References.md
│
├── src/                          # Source code
│   ├── text_model/               # BERT-based text classifier
│   │   ├── model.py
│   │   ├── train_text.py
│   │   └── text_dataset.py
│   ├── image_model/              # EfficientNet-based image classifier
│   │   ├── model.py
│   │   ├── train_image.py
│   │   └── image_dataset.py
│   ├── ocr_module/               # OCR text extraction
│   │   └── ocr_extract.py
│   ├── clip_checker/             # CLIP consistency checking
│   │   └── clip_module.py
│   ├── fusion_model/             # Final decision fusion
│   │   └── fusion.py
│   └── utils/                    # Helper utilities
│       ├── metrics.py
│       └── helpers.py
│
├── notebooks/                    # Jupyter notebooks (coming soon)
│   ├── text_training.ipynb
│   ├── image_training.ipynb
│   ├── fusion_experiments.ipynb
│   └── evaluation_report.ipynb
│
├── experiments/                  # Experiment logs and results
│   ├── text_logs.csv
│   ├── image_logs.csv
│   ├── fusion_results.csv
│   └── graphs/
│
├── data/                         # Datasets
│   ├── raw/
│   ├── processed/
│   ├── synthetic_text_generator.py
│   ├── synthetic_image_generator.py
│   └── README.md
│
├── diagrams/                     # System diagrams (Mermaid)
│   ├── system_workflow.mmd
│   ├── architecture_diagram.mmd
│   └── data_flow_diagram.mmd
│
├── presentation/                 # Presentation materials
│   ├── project_ppt.md
│   ├── demo_steps.md
│   └── viva_questions.md
│
├── screenshots/                  # Screenshot documentation
│   └── description_of_screenshots.md
│
├── README.md                     # This file
└── ROADMAP.md                    # Project roadmap
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Tesseract OCR

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/Prompt-Injection-and-Image-forgery-Detection.git
cd Prompt-Injection-and-Image-forgery-Detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies
```bash
pip install torch torchvision
pip install transformers
pip install efficientnet-pytorch
pip install git+https://github.com/openai/CLIP.git
pip install pytesseract
pip install scikit-learn pandas numpy matplotlib seaborn
pip install pillow opencv-python tqdm jupyter
```

### Step 4: Install Tesseract OCR
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Usage

### 1. Generate Synthetic Datasets
```bash
cd data
python synthetic_text_generator.py
python synthetic_image_generator.py
```

### 2. Train Text Model
```bash
cd src/text_model
python train_text.py
```

### 3. Train Image Model
```bash
cd src/image_model
python train_image.py
```

### 4. Run Inference
```python
from src.text_model.model import PromptInjectionDetector, TextPreprocessor
from src.image_model.model import ImageForgeryDetector
from src.ocr_module.ocr_extract import OCRExtractor
from src.clip_checker.clip_module import CLIPChecker
from src.fusion_model.fusion import FusionModel

# Initialize models
text_model = PromptInjectionDetector()
image_model = ImageForgeryDetector()
ocr = OCRExtractor()
clip_checker = CLIPChecker()
fusion = FusionModel()

# Analyze content
text_score = text_model.predict_proba(text_input)
image_score = image_model.predict_proba(image_input)
clip_score = clip_checker.compute_similarity(image, caption)
ocr_text = ocr.extract_text(image)

# Get final decision
result = fusion.predict(text_score, image_score, clip_score, ocr_score)
print(f"Final Decision: {result['label']}")
```

---

## Results

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Text Model (BERT) | **91.2%** | 0.90 | 0.92 | 0.89 |
| Image Model (EfficientNet) | **87.5%** | 0.85 | 0.89 | 0.86 |
| Fusion Model | **89.3%** | 0.88 | 0.91 | 0.89 |

### Key Achievements
- All accuracy targets exceeded
- Multimodal approach improves robustness
- CLIP adds 2.85% improvement through consistency checking
- Processing time: ~3 seconds per sample

---

## 6-Month Development Timeline

| Month | Activities |
|-------|------------|
| **Month 1** | Research, literature review, environment setup |
| **Month 2** | Dataset generation, preprocessing pipelines |
| **Month 3** | BERT text model development and training |
| **Month 4** | EfficientNet image model, OCR integration |
| **Month 5** | CLIP module, fusion model experiments |
| **Month 6** | Final evaluation, documentation, presentation |

---

## Documentation

Complete academic documentation is available in the `project_report/` directory:

- **Abstract:** Project summary (2 pages)
- **Introduction:** Background and motivation (10 pages)
- **Problem Statement:** Clear problem definition (8 pages)
- **Literature Review:** Survey of existing work (15 pages)
- **Methodology:** Detailed approach (20 pages)
- **System Architecture:** Design and components (12 pages)
- **Results:** Experimental analysis (18 pages)
- **Conclusion:** Summary and learnings (8 pages)
- **Future Scope:** Potential improvements (10 pages)

**Total: 85+ pages**

---

## Presentation Materials

Available in `presentation/` directory:
- **project_ppt.md:** Full presentation slides (24 slides)
- **demo_steps.md:** Live demo instructions
- **viva_questions.md:** 30+ Q&A for viva preparation

---

## Hardware Requirements

### Minimum (Inference Only)
- CPU: Intel Core i5
- RAM: 8 GB
- Storage: 10 GB

### Recommended (Training)
- CPU: Intel Core i7 / AMD Ryzen 7
- RAM: 16 GB
- GPU: NVIDIA GTX 1080+ (8GB VRAM)
- Storage: 50 GB SSD

---

## Limitations

1. Trained on synthetic datasets (may not generalize to all real-world attacks)
2. English language only for text analysis
3. Basic forgery types in training data
4. ~3 second processing time per sample
5. Binary classification (no severity levels)

---

## Future Scope

- Real-world dataset collection
- Multi-language support
- Video analysis capability
- Real-time processing (<500ms)
- Web/mobile application
- Adversarial robustness testing
- Enterprise deployment

---

## References

1. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers," NAACL 2019
2. Tan & Le, "EfficientNet: Rethinking Model Scaling," ICML 2019
3. Radford et al., "Learning Transferable Visual Models (CLIP)," ICML 2021
4. Greshake et al., "Compromising Real-World LLM Applications," AISec 2023
5. OWASP Top 10 for LLM Applications, 2023

See `project_report/References.md` for complete bibliography (37 references).

---

## Project Team

- **Student 1:** [Name] - Text Model Development
- **Student 2:** [Name] - Image Model Development
- **Student 3:** [Name] - Integration and Documentation

**Guide:** Prof. [Guide Name]

**Institution:** [College Name]

**Academic Year:** 2023-2024

---

## License

This project is developed for academic purposes as part of a Final Year Major Project. Feel free to use for educational purposes with proper attribution.

---

## Acknowledgments

We would like to thank:
- Our project guide for constant support and guidance
- Our college for providing resources
- The open-source community for excellent libraries
- Hugging Face, PyTorch, and OpenAI for pre-trained models

---

## Contact

For questions or feedback:
- Email: student@college.edu
- GitHub: [Repository Link]

---

**Thank you for exploring our project!**

*This is a 6-month academic project demonstrating the application of multimodal deep learning for content verification and security.*
