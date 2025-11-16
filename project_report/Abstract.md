# Abstract

## Prompt Injection and Image Forgery Detection Using Multimodal Deep Learning

In today's digital world, the spread of fake content has become a serious problem. With the rise of AI-generated text and manipulated images, it has become very difficult to know what is real and what is fake. This project addresses two major problems: **Prompt Injection attacks** in text and **Image Forgery detection** in images.

Prompt Injection is a new type of attack where malicious users try to trick AI systems by inserting hidden commands in the input text. Image Forgery refers to the manipulation of images to create fake or misleading content. Both of these problems can cause serious harm if not detected properly.

In this project, we have developed a **multimodal deep learning system** that can detect both types of threats. Our system uses:

1. **BERT-based text classifier** to detect prompt injection in text
2. **EfficientNet-based image classifier** to detect forged or manipulated images
3. **OCR (Optical Character Recognition)** module to extract hidden text from images
4. **CLIP (Contrastive Language-Image Pre-training)** module to check if the text matches the image content
5. **Fusion model** that combines all the results to give a final verdict

The system takes both text and image as input and processes them through multiple stages. First, the text is checked for prompt injection. Then, the image is analyzed for forgery. OCR extracts any text present in the image, and CLIP verifies if the image content matches the given description. Finally, a fusion layer combines all these results to classify the input as either **Safe** or **Forged/Malicious**.

We created synthetic datasets for both prompt injection and image forgery. For text, we generated examples of normal text and injection attempts. For images, we created manipulated versions of original images using simple editing techniques.

Our experimental results show that:
- Text model achieved **91.2% accuracy** in detecting prompt injection
- Image model achieved **87.5% accuracy** in detecting image forgery
- The combined fusion model achieved **89.3% accuracy** in overall detection

This project demonstrates that multimodal deep learning can effectively detect both text-based and image-based threats. The fusion approach provides better results than using individual models alone.

**Keywords:** Prompt Injection, Image Forgery Detection, Multimodal Learning, BERT, EfficientNet, CLIP, OCR, Deep Learning, Cybersecurity

---

**Number of Pages:** 85+
**Number of Figures:** 25+
**Number of Tables:** 15+
**Duration:** 6 months
**Project Type:** Major Project / Final Year Project
