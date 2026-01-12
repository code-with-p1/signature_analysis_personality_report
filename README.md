# Handwriting-Based Personality Detection Dataset

## 📁 Dataset Overview

This dataset provides handwriting samples categorized according to the **Big Five personality traits** (also known as the OCEAN model). It is designed for research and development in personality detection through handwriting analysis.

*   **Dataset Name**: Handwriting based personality detection
*   **Provider**: Vuppala Adithya Sairam
*   **Update Date**: Updated 2 years ago (from the source publication date of 2023-11-20)
*   **Size**: Approximately 1.37 GB
*   **Files**: 4,076 files in total
*   **Download Link**: [Handwriting based personality detection on Kaggle](https://www.kaggle.com/datasets/vuppalaadithyasairam/handwriting-based-personality-detection)

## 🎯 Potential Applications

This dataset can be used for:
*   Building machine learning models for personality trait classification from handwriting images.
*   Research in graphology (handwriting analysis) and computational psychology.
*   Multi-modal AI research combining visual pattern recognition with behavioral traits.
*   Educational projects in computer vision and machine learning.

## 🔧 Environment Setup & Usage

To use this dataset with common AI/ML APIs, set the following environment variables in your project:

```bash
# Required API Keys (Fill in your own keys)
OPENAI_API_KEY="your_openai_api_key_here"
GOOGLE_API_KEY="your_google_api_key_here"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token_here"
```

### Suggested Workflow:
1.  **Download the dataset** from the provided Kaggle link.
2.  **Preprocess the images** (resize, normalize, augment as needed).
3.  **Use the directory structure** as labels for supervised learning (each folder name is a personality trait).
4.  **Train a model** using frameworks like TensorFlow, PyTorch, or through API services using the provided keys.
5.  **Evaluate** on the `augmented test/` set.