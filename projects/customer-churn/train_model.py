# ============================================
# CUSTOMER CHURN PREDICTION SYSTEM
# CodSoft Internship - Task 3
# Part 1: Data Preprocessing & Model Training
# ============================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib

# Use non-GUI backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------
# Create static folder
# --------------------------------------------

os.makedirs("static", exist_ok=True)

# --------------------------------------------
# Load Dataset
# --------------------------------------------

df = pd.read_csv(
    r"C:\Users\KASHITHRA J\OneDrive\Desktop\Task3_CustomerChurn\Churn_Modelling.csv"
)

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(df.head())

# --------------------------------------------
# Remove unnecessary columns
# --------------------------------------------

df.drop(
    columns=[
        "RowNumber",
        "CustomerId",
        "Surname"
    ],
    inplace=True
)

print("\nColumns Used:")
print(df.columns)

# --------------------------------------------
# Encode Categorical Columns
# --------------------------------------------

geo_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

df["Geography"] = geo_encoder.fit_transform(df["Geography"])
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# --------------------------------------------
# Split Features and Target
# --------------------------------------------

X = df.drop("Exited", axis=1)
y = df["Exited"]

feature_names = X.columns

# --------------------------------------------
# Feature Scaling
# --------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --------------------------------------------
# Train Test Split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# --------------------------------------------
# Logistic Regression
# --------------------------------------------

print("\nTraining Logistic Regression...")

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_pred)

print("Done.")

# --------------------------------------------
# Random Forest
# --------------------------------------------

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Done.")

# --------------------------------------------
# Select Best Model
# --------------------------------------------

if rf_accuracy >= log_accuracy:
    best_model = rf_model
    best_name = "Random Forest"
    prediction = rf_pred
else:
    best_model = log_model
    best_name = "Logistic Regression"
    prediction = log_pred

print("\nBest Model Selected :", best_name)

# ============================================
# PART 2 : MODEL EVALUATION
# ============================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
roc = roc_auc_score(y_test, prediction)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, prediction))

cm = confusion_matrix(y_test, prediction)

print("\nConfusion Matrix")
print(cm)
plt.figure(figsize=(5,5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

labels = ["No Churn","Churn"]

plt.xticks([0,1], labels)

plt.yticks([0,1], labels)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j],
                 ha="center",
                 va="center",
                 fontsize=14)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("static/confusion_matrix.png")

plt.close()

print("Confusion Matrix Saved")
# ============================================
# FEATURE IMPORTANCE GRAPH
# ============================================

if best_name == "Random Forest":

    importance = best_model.feature_importances_

    plt.figure(figsize=(10,6))

    plt.barh(feature_names, importance)

    plt.title("Feature Importance")

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig("static/feature_importance.png")

    plt.close()

    print("Feature Importance Graph Saved")
# ============================================
# MODEL COMPARISON GRAPH
# ============================================

plt.figure(figsize=(6,5))

models = ["Logistic Regression", "Random Forest"]

scores = [log_accuracy, rf_accuracy]

bars = plt.bar(models, scores)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

for bar in bars:

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.003,
        f"{bar.get_height():.3f}",
        ha='center'
    )

plt.tight_layout()

plt.savefig("static/model_comparison.png")

plt.close()

print("Model Comparison Graph Saved")
# ============================================
# CONFUSION MATRIX GRAPH
# ============================================

plt.figure(figsize=(5,5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

labels = ["No Churn","Churn"]

plt.xticks([0,1], labels)

plt.yticks([0,1], labels)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j],
                 ha="center",
                 va="center",
                 fontsize=14)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("static/confusion_matrix.png")

plt.close()

print("Confusion Matrix Saved")
# ============================================
# SAVE MODEL
# ============================================

joblib.dump(best_model, "churn_model.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(geo_encoder, "geo_encoder.pkl")

joblib.dump(gender_encoder, "gender_encoder.pkl")

print("\nModel Saved Successfully")