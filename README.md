# AI-Powered-Potato-Leaf-Disease-Detection-and-Severity-Estimation
AI-powered system for potato leaf disease classification and severity estimation using deep learning (VGG16), image segmentation (U-2-Net), K-means clustering, and Grad-CAM visual explanations. Designed for real-time deployment and practical agricultural applications.

# AI-Powered Potato Leaf Disease Detection and Severity Estimation

This repository contains the implementation of a computer vision project for detecting and estimating the severity of potato leaf diseases using AI techniques. The system classifies leaf images into three categories (Healthy, Early Blight, Late Blight), estimates disease severity using K-means clustering, and provides visual explainability with Grad-CAM.

This project was developed as part of ICT619 Artificial Intelligence course at Murdoch University.

## Project Overview

Traditional manual methods of disease detection are time-consuming, subjective, and unsuitable for large-scale farming. This project provides an automated and scalable solution using deep learning and clustering techniques to assist farmers in timely decision-making.

Key components:
- VGG16 CNN model for disease classification
- U-2-Net (via rembg) for background removal and leaf isolation
- K-means clustering for severity estimation
- Grad-CAM for visual explainability


## Objectives

- Classify potato leaves as Healthy, Early Blight, or Late Blight
- Estimate severity of disease on infected leaves
- Visualize decision-making regions using Grad-CAM
- Build a solution ready for real-time deployment in low-resource environments

## Dataset

- Source: [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- Classes used: Healthy, Early Blight, Late Blight
- Preprocessing:
  - Resizing (256x256)
  - Normalization
  - Data Augmentation (flip, rotate, zoom)
  - Background removal using `rembg`

## AI Techniques Used

- **VGG16**: Transfer learning for classification, fine-tuned for this task
- **K-means Clustering**: For segmenting infected vs healthy areas to estimate severity
- **Grad-CAM**: For heatmaps that explain CNN decisions
- **U-2-Net (via rembg)**: Removes image background to improve model focus

## Results

- **Classification Accuracy**: 95% (after background removal)
- **Severity Estimation Range**: 21.01% to 47.83%
- Grad-CAM confirms the model focuses on disease regions
- Improved interpretability and accuracy compared to baseline CNN

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Grad-CAM heatmaps
- Visual overlays for K-means segmentation

## Innovation Highlights

- Integrates classification, severity estimation, and explainability in one pipeline
- Designed for use in field environments using lightweight, fast models
- Can be integrated into mobile apps or IoT field cameras

## How to Run

1. Install required libraries:

pip install tensorflow keras opencv-python matplotlib rembg scikit-learn

