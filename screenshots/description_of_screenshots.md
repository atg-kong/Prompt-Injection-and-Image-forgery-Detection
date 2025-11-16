# Screenshots for Project Documentation

## Overview

This document describes the screenshots that should be captured for the project report and presentation. These screenshots serve as evidence of the work done and help visualize the system's functionality.

---

## 1. Development Environment Screenshots

### Screenshot 1.1: Project Structure
**Description:** File explorer view showing the complete project directory structure
**What to capture:**
- Root folder with all subdirectories
- Expand to show src/, data/, notebooks/, experiments/ folders
- Shows organized codebase

### Screenshot 1.2: IDE/Code Editor
**Description:** VS Code or PyCharm showing the code
**What to capture:**
- Open one of the model files (e.g., text_model/model.py)
- Show syntax highlighting
- Show well-commented code

### Screenshot 1.3: Terminal/Command Prompt
**Description:** Terminal showing Python environment
**What to capture:**
- `pip list` output showing installed libraries
- Python version
- GPU availability check (`torch.cuda.is_available()`)

---

## 2. Dataset Screenshots

### Screenshot 2.1: Text Dataset CSV
**Description:** Excel or CSV viewer showing text dataset
**What to capture:**
- First 10-15 rows of synthetic_text_dataset.csv
- Show 'text' and 'label' columns
- Mix of safe (0) and injection (1) samples

### Screenshot 2.2: Sample Injection Texts
**Description:** Close-up of injection examples
**What to capture:**
- 3-4 examples of injection attempts
- Show different attack types
- Highlight suspicious keywords

### Screenshot 2.3: Image Dataset
**Description:** Folder view of images
**What to capture:**
- Grid view of original and forged images
- Side-by-side comparison
- Different forgery types

### Screenshot 2.4: Dataset Statistics
**Description:** Python output showing dataset info
**What to capture:**
- Total samples
- Class distribution
- Train/val/test split ratios

---

## 3. Training Screenshots

### Screenshot 3.1: Text Model Training Log
**Description:** Console output during training
**What to capture:**
- Epoch progress with loss and accuracy
- Training and validation metrics
- Best model save notification
- Total training time

**Example:**
```
Epoch 1/10
Training - Loss: 0.6823, Accuracy: 0.6542
Validation - Loss: 0.6145, Accuracy: 0.6780
...
Epoch 10/10
Training - Loss: 0.0823, Accuracy: 0.9548
Validation - Loss: 0.2545, Accuracy: 0.9120
[NEW BEST] Model saved with accuracy: 0.9120
```

### Screenshot 3.2: Image Model Training Log
**Description:** Similar to text model training
**What to capture:**
- 15 epoch training progress
- Learning rate scheduling
- Final accuracy

### Screenshot 3.3: GPU Memory Usage
**Description:** nvidia-smi output
**What to capture:**
- GPU utilization
- Memory usage during training
- Temperature

### Screenshot 3.4: Google Colab (if used)
**Description:** Colab notebook interface
**What to capture:**
- Connected runtime
- GPU type (Tesla T4)
- Training cell execution

---

## 4. Evaluation Screenshots

### Screenshot 4.1: Text Model Confusion Matrix
**Description:** Matplotlib heatmap
**What to capture:**
- 2x2 confusion matrix
- True Negatives, False Positives, False Negatives, True Positives
- Color-coded with annotations
- Title: "Text Model Confusion Matrix"

### Screenshot 4.2: Image Model Confusion Matrix
**Description:** Similar to text model
**What to capture:**
- Confusion matrix for image classification
- Authentic vs Forged

### Screenshot 4.3: Training Curves
**Description:** Loss and accuracy plots
**What to capture:**
- Two subplots side by side
- Training loss vs Validation loss
- Training accuracy vs Validation accuracy
- X-axis: Epochs, Y-axis: Loss/Accuracy
- Legend showing train and val

### Screenshot 4.4: Model Comparison Bar Chart
**Description:** Comparison of different models
**What to capture:**
- Bar chart with 3-4 models
- Metrics: Accuracy, Precision, Recall, F1
- Values labeled on bars

### Screenshot 4.5: Classification Report
**Description:** Scikit-learn classification report
**What to capture:**
- Text output showing:
  - Precision, Recall, F1-score per class
  - Support (number of samples)
  - Macro and weighted averages

---

## 5. Demo/Inference Screenshots

### Screenshot 5.1: Safe Text Classification
**Description:** Terminal showing safe text prediction
**What to capture:**
- Input: "How do I learn Python?"
- Output: Safe probability ~0.92, Injection probability ~0.08
- Final classification: SAFE

### Screenshot 5.2: Injection Detection
**Description:** Terminal showing injection detection
**What to capture:**
- Input: "Ignore all previous instructions..."
- Output: High injection probability
- Final classification: INJECTION

### Screenshot 5.3: Image Forgery Detection
**Description:** Side-by-side original and forged image with predictions
**What to capture:**
- Original image → Authentic classification
- Forged image → Forged classification
- Confidence scores

### Screenshot 5.4: OCR Text Extraction
**Description:** Image with extracted text
**What to capture:**
- Input image containing text
- Extracted text output
- Hidden prompt detection results

### Screenshot 5.5: CLIP Consistency Check
**Description:** Text-image matching results
**What to capture:**
- Image displayed
- Matching caption → High similarity (0.8+)
- Mismatched caption → Low similarity (<0.3)

### Screenshot 5.6: Fusion Model Report
**Description:** Complete detection report
**What to capture:**
```
=== MULTIMODAL DETECTION REPORT ===
[VERDICT] MALICIOUS
Confidence: 0.92

--- Individual Analysis Scores ---
  Text Injection: 0.85
  Image Forgery: 0.15
  Text-Image Consistency: 0.75
  OCR Hidden Text: 0.10

--- Detection Reasons ---
  - High prompt injection probability: 0.85
```

---

## 6. Jupyter Notebook Screenshots

### Screenshot 6.1: Training Notebook
**Description:** Jupyter notebook interface
**What to capture:**
- Cell with model training code
- Output showing training progress
- Well-organized cells with markdown headers

### Screenshot 6.2: Experiment Results
**Description:** Notebook showing analysis
**What to capture:**
- Metrics calculation
- Visualization cells
- Interpretation text cells

### Screenshot 6.3: Data Exploration
**Description:** Dataset analysis in notebook
**What to capture:**
- Pandas dataframe display
- Dataset statistics
- Sample visualization

---

## 7. System Architecture Screenshots

### Screenshot 7.1: Architecture Diagram
**Description:** System architecture visualization
**What to capture:**
- Mermaid diagram rendered
- Shows all modules and connections
- Clear flow from input to output

### Screenshot 7.2: Data Flow Diagram
**Description:** DFD showing data movement
**What to capture:**
- Input/Output entities
- Processing modules
- Data stores

---

## 8. Results Screenshots

### Screenshot 8.1: Final Results Table
**Description:** Summary of all model performances
**What to capture:**
- Table with all models
- Accuracy, Precision, Recall, F1
- Highlight best results

### Screenshot 8.2: Experiment Logs
**Description:** CSV file with experiment results
**What to capture:**
- fusion_results.csv
- Different configurations tested
- Performance metrics

### Screenshot 8.3: ROC Curve
**Description:** ROC curve plot
**What to capture:**
- True Positive Rate vs False Positive Rate
- AUC score
- Diagonal baseline

---

## How to Capture Screenshots

### Windows:
- Use Snipping Tool or Win + Shift + S
- PrtScn for full screen

### Mac:
- Cmd + Shift + 4 for selection
- Cmd + Shift + 3 for full screen

### Linux:
- Use gnome-screenshot
- Or install Flameshot

### Best Practices:
1. High resolution (at least 1080p)
2. Clear, readable text
3. Crop to relevant area
4. Add borders if needed
5. Consistent naming convention
6. PNG format for quality

---

## Screenshot Naming Convention

```
screenshot_[category]_[number]_[description].png

Examples:
screenshot_training_01_text_model_progress.png
screenshot_results_02_confusion_matrix.png
screenshot_demo_03_injection_detection.png
```

---

## Screenshots Checklist

- [ ] Development environment (3 screenshots)
- [ ] Dataset (4 screenshots)
- [ ] Training process (4 screenshots)
- [ ] Evaluation results (5 screenshots)
- [ ] Demo/Inference (6 screenshots)
- [ ] Jupyter notebooks (3 screenshots)
- [ ] Architecture diagrams (2 screenshots)
- [ ] Final results (3 screenshots)

**Total: ~30 screenshots**

---

## Notes for Report

When including screenshots in the report:
1. Add figure numbers (Figure 1, Figure 2, etc.)
2. Add descriptive captions
3. Reference in text ("As shown in Figure 5...")
4. Ensure readability when printed
5. Keep consistent size
6. Add annotations if needed
