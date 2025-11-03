# Advanced Waste Classification for Smart Recycling using Deep Learning

### Problem Statement

Improper waste disposal and inefficient segregation at the source are significant barriers to effective recycling. A substantial amount of recyclable and hazardous material ends up in landfills due to contamination and a lack of detailed sorting, leading to increased environmental pollution and wasted resources. Manual sorting processes for such a wide variety of materials are impractical, slow, and prone to error. This project aims to address this challenge by developing an intelligent system that can automatically classify a wide range of household waste, thereby promoting better recycling practices and contributing to a more sustainable environment.

### Project Objective

The primary objective of this project is to design, build, and train a Convolutional Neural Network (CNN) model capable of accurately identifying and classifying images of waste into **12 distinct categories**. These categories include `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, and `white-glass`. This model will serve as the core component for a more advanced and nuanced automated waste segregation system.

### Dataset

This project utilizes the **Garbage Classification V2** dataset available on Kaggle.

*   **Source:** [https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
*   **Description:** This is an expanded dataset containing images across 12 different classes of waste materials. The detailed categorization, which includes different colors of glass, batteries, and clothing, allows for the development of a more precise and practical sorting model compared to simpler datasets.


### Methodology

1.  **Data Preprocessing:** The image dataset will be preprocessed by resizing all images to a uniform dimension, normalizing pixel values, and augmenting the data (e.g., rotation, flipping) to improve model generalization.
2.  **Model Architecture:** A Convolutional Neural Network (CNN) will be implemented.
3.  **Training & Evaluation:** The model will be trained on the preprocessed dataset and its performance will be evaluated using key metrics such as accuracy, precision, recall, and a confusion matrix on a held-out test set to analyze its performance on each of the 12 classes.
   

*   **Language:** Python
*   **Libraries:** TensorFlow/Keras, Scikit-learn, OpenCV, Matplotlib
*   **Environment:** Google Colab 
