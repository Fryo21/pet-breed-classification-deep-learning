# Pet Breed Image Classification using Deep Learning

This project explores fine-grained image classification using the **Oxford-IIIT Pet Dataset**, focusing on identifying pet breeds from images using deep learning. Multiple convolutional neural network (CNN) models were designed, trained, and evaluated to compare custom architectures against transfer-learning approaches.

The project was completed as part of an MSc-level Deep Learning coursework and demonstrates a full **end-to-end machine learning pipeline**, from data preprocessing to model evaluation and analysis.

---

## Project Overview

The primary objective is to classify images of pets into one of **37 distinct breeds** (cats and dogs). Due to visual similarity between breeds, the task presents real-world challenges such as fine-grained feature extraction, class imbalance, and generalisation.

To address this, the project investigates multiple modelling strategies:
- Binary classification (Cat vs Dog)
- Breed classification within each species
- Full multi-class breed classification
- Comparison between custom CNNs and transfer learning

---

## Dataset

- **Dataset:** Oxford-IIIT Pet Dataset  
- **Classes:** 37 pet breeds (12 cats, 25 dogs)
- **Annotations:** Class labels and segmentation masks (classification task used labels only)
- **Source:**  
  https://academictorrents.com/details/b18bbd9ba03d50b0f7f479acc9f4228a408cecc

Images were normalised, resized, and split into training, validation, and test sets. Data augmentation was applied to improve generalisation and reduce overfitting.

---

## Models Implemented

A total of **four deep learning models** were built and evaluated:

### 1. Pet Family Classifier (Binary)
- Task: Cat vs Dog
- Model: Custom CNN
- Purpose: Establish a baseline and validate preprocessing pipeline

### 2. Cat Breed Classifier
- Task: 12-class cat breed classification
- Model: Custom CNN
- Focus: Fine-grained visual feature learning

### 3. Dog Breed Classifier
- Task: 25-class dog breed classification
- Model: Transfer learning with **MobileNetV2**
- Purpose: Improve accuracy and efficiency using pretrained features

### 4. Combined Breed Classifier
- Task: 37-class classification (all breeds)
- Model: Transfer learning
- Challenge: High inter-class similarity and increased class complexity

---

## Training & Evaluation

### Training
- Framework: **TensorFlow / Keras**
- Optimiser: Adam
- Loss: Categorical / Binary Cross-Entropy (depending on task)
- Regularisation: Data augmentation, dropout

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrices

Performance was compared across all models to analyse trade-offs between complexity, accuracy, and computational cost.

---

## Key Results & Insights

- Transfer learning significantly outperformed custom CNNs for fine-grained breed classification.
- Species-specific classifiers (cats vs dogs) achieved higher accuracy than a single unified model.
- MobileNetV2 provided a strong balance between performance and efficiency.
- Certain breeds consistently exhibited higher confusion due to visual similarity.
- Data augmentation improved generalisation but could not fully compensate for class imbalance.

---

## Limitations

- No object detection or semantic segmentation was implemented.
- Some breed classes remain difficult to distinguish due to limited visual variance.
- Training was performed offline; models were not deployed in a production environment.

---

## Future Work

Potential extensions include:
- Object detection or semantic segmentation using bounding box or mask annotations
- Few-shot learning for under-represented breeds
- GAN-based data augmentation
- Model compression for mobile deployment
- Real-time inference via a web-based interface

---

## Repository Structure

