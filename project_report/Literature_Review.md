# Chapter 3: Literature Review

## 3.1 Introduction to Literature Review

In this chapter, we review the existing research and work done in the areas of prompt injection detection, image forgery detection, and multimodal learning. This review helped us understand what has already been done and what gaps exist that our project can fill.

## 3.2 Prompt Injection Detection

### 3.2.1 What is Prompt Injection?

Prompt injection was first discussed widely in 2022-2023 with the rise of ChatGPT and other Large Language Models (LLMs). The term was coined by security researchers who noticed that users could manipulate AI responses by inserting special commands.

**Key Papers and Resources:**

1. **"Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" (Greshake et al., 2023)**
   - This paper introduced the concept of indirect prompt injection
   - Showed how attackers can inject prompts through external data sources
   - Demonstrated attacks on real applications like Bing Chat

2. **"Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition" (Schulhoff et al., 2023)**
   - Organized a competition to find prompt injection vulnerabilities
   - Collected thousands of prompt injection examples
   - Showed the creativity of attackers in crafting injections

3. **OWASP Top 10 for LLM Applications (2023)**
   - Listed prompt injection as the #1 vulnerability in LLM applications
   - Provided guidelines for detection and prevention
   - Categorized different types of prompt injections

### 3.2.2 Existing Detection Methods

Several methods have been proposed for detecting prompt injections:

**1. Rule-Based Detection:**
- Uses keyword matching and regular expressions
- Looks for phrases like "ignore previous instructions", "system prompt", etc.
- Simple but easily bypassed

**2. Perplexity-Based Detection:**
- Measures how "surprising" the input text is to the model
- High perplexity may indicate injection attempts
- Proposed by Simon Willison and others

**3. Machine Learning-Based Detection:**
- Uses classifiers trained on labeled data
- BERT and other transformers for text classification
- More robust but needs training data

### 3.2.3 Limitations in Existing Work

- Most solutions focus on specific types of injections
- Limited public datasets available
- Attackers constantly evolve their techniques
- Few solutions integrate with other security checks

## 3.3 Image Forgery Detection

### 3.3.1 Types of Image Forgery

Image forgery has been studied for many years. The main types include:

1. **Copy-Move Forgery** - Copying and pasting parts within same image
2. **Splicing** - Combining parts from different images
3. **Inpainting** - Removing objects and filling the area
4. **Enhancement** - Modifying colors, contrast, etc.

### 3.3.2 Key Research Papers

1. **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (Tan & Le, 2019)**
   - Introduced EfficientNet architecture
   - Better accuracy with fewer parameters
   - We used this for our image classifier

2. **"TruFor: Leveraging all-round clues for trustworthy image forgery detection" (Guillaro et al., 2023)**
   - State-of-the-art forgery detection
   - Uses multiple visual cues
   - Inspired our approach

3. **"Learning Rich Features for Image Manipulation Detection" (Zhou et al., 2018)**
   - CNN-based approach
   - Learns features that indicate manipulation
   - Good baseline for comparison

4. **"ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries" (Wu et al., 2019)**
   - End-to-end trainable network
   - Can localize forged regions
   - More complex than our approach

### 3.3.3 Traditional vs Deep Learning Methods

**Traditional Methods:**
- Error Level Analysis (ELA)
- Noise analysis
- JPEG compression artifacts
- Metadata analysis

**Deep Learning Methods:**
- CNNs for feature extraction
- Transfer learning from ImageNet
- Attention mechanisms
- Encoder-decoder architectures

Our project uses deep learning (EfficientNet) as it provides better accuracy on diverse forgeries.

## 3.4 OCR for Text Extraction

### 3.4.1 OCR Technology

Optical Character Recognition (OCR) has been around for decades but has improved significantly with deep learning.

**Popular OCR Tools:**
1. **Tesseract** - Open source, developed by Google
2. **EasyOCR** - Deep learning-based
3. **Google Cloud Vision** - Commercial API
4. **AWS Textract** - Commercial service

We chose **Pytesseract** (Python wrapper for Tesseract) because:
- Free and open source
- Good accuracy for printed text
- Easy to integrate
- Widely used in academia

### 3.4.2 OCR in Security Applications

OCR is used in security for:
- Detecting hidden text in images (steganography)
- Extracting text from screenshots for analysis
- Identifying text-based manipulation in images

## 3.5 CLIP and Multimodal Learning

### 3.5.1 What is CLIP?

CLIP (Contrastive Language-Image Pre-training) was introduced by OpenAI in 2021.

**Key Paper:**
"Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)

CLIP learns to:
- Understand images and text together
- Match images with their descriptions
- Compute similarity between text and images

### 3.5.2 How CLIP Works

1. Image is passed through a Vision Transformer (ViT) encoder
2. Text is passed through a Transformer text encoder
3. Both are projected to a common embedding space
4. Cosine similarity measures how well they match

### 3.5.3 Applications of CLIP

- Zero-shot image classification
- Image-text matching
- Content moderation
- Fake content detection

We use CLIP to check if the text description matches the actual image content. This helps detect misinformation where images are paired with misleading captions.

## 3.6 Multimodal Deep Learning

### 3.6.1 What is Multimodal Learning?

Multimodal learning combines multiple types of data (modalities) like:
- Text
- Images
- Audio
- Video

The idea is that combining information from multiple sources provides better understanding than using a single source.

### 3.6.2 Fusion Strategies

Different ways to combine multimodal information:

1. **Early Fusion:**
   - Combine raw features at input level
   - Simple but may lose modality-specific patterns

2. **Late Fusion:**
   - Process each modality separately
   - Combine final predictions
   - What we use in our project

3. **Hybrid Fusion:**
   - Combine at multiple levels
   - More complex but potentially better

### 3.6.3 Related Work

**"Multimodal Machine Learning: A Survey and Taxonomy" (Baltrusaitis et al., 2019)**
- Comprehensive survey of multimodal learning
- Discusses challenges like representation, translation, alignment
- Good theoretical foundation

## 3.7 Gaps in Existing Literature

After reviewing the literature, we identified these gaps:

1. **Limited Combined Solutions:** Most work focuses on either text OR images, not both together

2. **Lack of Prompt Injection Datasets:** Very few public datasets for prompt injection detection

3. **No OCR Integration:** Few systems extract and analyze text hidden in images

4. **Missing Consistency Checks:** CLIP is rarely used for detecting mismatched text-image pairs in security contexts

5. **Complex Solutions:** Most research proposes complex architectures not suitable for learning purposes

## 3.8 How Our Project Fills These Gaps

Our project addresses these gaps by:

1. **Combining text and image analysis** in a single pipeline
2. **Creating synthetic datasets** for both prompt injection and image forgery
3. **Integrating OCR** to extract hidden text from images
4. **Using CLIP** for text-image consistency checking
5. **Keeping it simple** with well-documented code suitable for students

## 3.9 Comparison Table

| Feature | Existing Solutions | Our Solution |
|---------|-------------------|--------------|
| Text Analysis | ✓ | ✓ |
| Image Analysis | ✓ | ✓ |
| Combined Approach | Limited | ✓ |
| OCR Integration | Rare | ✓ |
| CLIP Consistency | Rare | ✓ |
| Simple Implementation | No | ✓ |
| Academic-Friendly | No | ✓ |
| Synthetic Dataset Generation | Limited | ✓ |

## 3.10 Summary

In this chapter, we reviewed existing literature on prompt injection detection, image forgery detection, OCR, CLIP, and multimodal learning. We identified gaps in current solutions and explained how our project addresses these gaps. The next chapter will present the detailed methodology we followed.
