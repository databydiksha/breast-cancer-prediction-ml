# Breast-Cancer-Prediction-ML
A healthcare-based machine learning system that predicts breast cancer diagnosis using supervised learning models, featuring data preprocessing, model training, evaluation, and performance optimization.
# Breast Cancer Prediction Using Machine Learning

## Overview

This project is a Machine Learning classification system that predicts whether a breast tumor is **malignant or benign** based on diagnostic features. It is built using Python and trained on the Breast Cancer Wisconsin Dataset.

The goal is to demonstrate a complete end-to-end ML pipeline including data preprocessing, model training, evaluation, and model persistence.

---

## Problem Statement

Early detection of breast cancer is critical for effective treatment. This project aims to build a predictive model that can assist in identifying tumors accurately using medical diagnostic data.

---

## Dataset

- **Name:** Breast Cancer Wisconsin Diagnostic Dataset  
- **Source:** sklearn / UCI Machine Learning Repository  
- **Type:** Classification  
- **Classes:**
  - Malignant (M)
  - Benign (B)

- **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, etc.)

---

## Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Pickle (Model Saving)

---

## Machine Learning Workflow

### 1. Data Preprocessing
- Handling missing values
- Label encoding (M → 1, B → 0)
- Feature scaling using StandardScaler

### 2. Exploratory Data Analysis (EDA)
- Correlation heatmaps
- Class distribution visualization
- Feature importance analysis

### 3. Model Building
Models tested:
- Logistic Regression  
- K-Nearest Neighbors  
- Support Vector Machine  
- Random Forest Classifier  

### 4. Model Evaluation
Metrics used:
- Accuracy Score  
- Confusion Matrix  
- Precision  
- Recall  
- F1 Score  

### 5. Final Model
The best performing model was selected based on accuracy and generalization ability.

---

## Model Saving

The trained model is saved using Pickle:

```python
import pickle

pickle.dump(model, open("breast_cancer_model.pkl", "wb"))
