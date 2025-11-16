# Project Some Final ****

How to proceed : 
1. Model will take text/image as input and classify it as forged or prompt injected
2. we will be using pre trained models like BERT/RoBERTa, these are NLP models pre trained for prompt injection
3. for image forgery use -->TruFor, EfficientNet classifier(this will be done later after prompt is done).
4. OCR module --> it extracts hidden texts from images  
5. CLIP to check if the caption / text matches the image.
6. then we will pipeline this 
    Input (Text + Image)
   ↓
OCR → Text Detector
   ↓
Image Detector
   ↓
CLIP Consistency
   ↓
Fusion Layer (Rule-based / ML)
   ↓
Output: Safe / Forged
 this will be pipeline flow

7.Then use dataset and training
8. Evaluation
