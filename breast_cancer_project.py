import warnings
warnings.filterwarnings("ignore")

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv(r"C:\Users\allab\OneDrive\Documents\GitHub Projects\Breast Cancer Predictive Model\data.csv.csv")

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

print("\nShape of Dataset:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================
# DATA CLEANING
# ==========================
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)

# Convert Target Column
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# EDA
# ==========================

# Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="diagnosis", data=df)
plt.title("Benign vs Malignant Cases")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ==========================
# FEATURES & TARGET
# ==========================
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# FEATURE SCALING
# ==========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# LOGISTIC REGRESSION
# ==========================
lr = LogisticRegression()

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

print("\nLOGISTIC REGRESSION ACCURACY")
print(accuracy_score(y_test, lr_pred))

# ==========================
# RANDOM FOREST
# ==========================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRANDOM FOREST ACCURACY")
print(accuracy_score(y_test, rf_pred))

# ==========================
# SVM
# ==========================
svm = SVC(
    kernel="rbf",
    probability=True
)

svm.fit(X_train_scaled, y_train)

svm_pred = svm.predict(X_test_scaled)

print("\nSVM ACCURACY")
print(accuracy_score(y_test, svm_pred))

# ==========================
# MODEL COMPARISON
# ==========================
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "SVM"
    ],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, svm_pred)
    ]
})

print("\nMODEL COMPARISON")
print(results.sort_values(by="Accuracy", ascending=False))

# ==========================
# CONFUSION MATRIX
# ==========================
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================
# CLASSIFICATION REPORT
# ==========================
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, rf_pred))

# ==========================
# FEATURE IMPORTANCE
# ==========================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 FEATURES")
print(importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.show()

# ==========================
# SAVE MODEL
# ==========================
joblib.dump(rf, joblib.dump(
    rf,
    r"C:\Users\allab\OneDrive\Documents\GitHub Projects\Breast Cancer Predictive Model\breast_cancer_model.pkl"
))

print("\nModel saved successfully!")
print("File Name: breast_cancer_model.pkl")
