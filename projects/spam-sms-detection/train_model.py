# ==========================================
# SMART SPAM SMS DETECTION SYSTEM
# CodSoft AI Internship Project
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# STEP 1 : LOAD DATASET
# ==========================================

dataset_path = r"C:\Users\KASHITHRA J\OneDrive\Desktop\Task4_Spam_SMS_Detection\spam.csv"

df = pd.read_csv(dataset_path, encoding="latin-1")

# Keep only useful columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("=" * 60)
print("DATASET PREVIEW")
print("=" * 60)
print(df.head())

print("\nDataset Shape :", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nClass Distribution")
print(df['label'].value_counts())

# ==========================================
# STEP 2 : DATA PREPROCESSING
# ==========================================

# Convert labels into numbers

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nLabel Conversion Completed")

# ==========================================
# STEP 3 : TF-IDF FEATURE EXTRACTION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X = vectorizer.fit_transform(df['message'])

y = df['label']

print("\nTF-IDF Feature Extraction Completed")

print("Number of Features :", len(vectorizer.get_feature_names_out()))

# ==========================================
# STEP 4 : TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ==========================================
# STEP 5 : CREATE MODELS
# ==========================================

models = {

    "Naive Bayes":
        MultinomialNB(),

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Support Vector Machine":
        LinearSVC()

}

results = []

best_accuracy = 0

best_model = None

best_model_name = ""

# ==========================================
# STEP 6 : TRAIN & EVALUATE ALL MODELS
# ==========================================

for name, model in models.items():

    print("\n")
    print("=" * 70)
    print("MODEL :", name)
    print("=" * 70)

    # Train Model
    model.fit(X_train, y_train)

    # Prediction
    prediction = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report")
    print("-" * 50)

    print(classification_report(
        y_test,
        prediction,
        target_names=["Ham","Spam"],
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, prediction)

    print("\nConfusion Matrix")

    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Ham","Spam"]
    )

    disp.plot(cmap="Blues")

    plt.title(f"Confusion Matrix - {name}")

    plt.show()

    # Store Results
    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    # Save Best Model
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

# ==========================================
# STEP 7 : MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison)

print("\nBest Model Selected :", best_model_name)
print("Best Accuracy :", round(best_accuracy,4))

# ==========================================
# STEP 8 : ACCURACY COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title("Accuracy Comparison of ML Algorithms")

plt.xlabel("Algorithms")

plt.ylabel("Accuracy")

plt.xticks(rotation=10)

for index, value in enumerate(comparison["Accuracy"]):

    plt.text(
        index,
        value,
        f"{value:.3f}",
        ha="center"
    )

plt.show()

# ==========================================
# STEP 9 : PRECISION COMPARISON
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(
    comparison["Model"],
    comparison["Precision"]
)

plt.title("Precision Comparison")

plt.xlabel("Algorithms")

plt.ylabel("Precision")

plt.xticks(rotation=10)

plt.show()

# ==========================================
# STEP 10 : RECALL COMPARISON
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(
    comparison["Model"],
    comparison["Recall"]
)

plt.title("Recall Comparison")

plt.xlabel("Algorithms")

plt.ylabel("Recall")

plt.xticks(rotation=10)

plt.show()

# ==========================================
# STEP 11 : F1 SCORE COMPARISON
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(
    comparison["Model"],
    comparison["F1 Score"]
)

plt.title("F1 Score Comparison")

plt.xlabel("Algorithms")

plt.ylabel("F1 Score")

plt.xticks(rotation=10)

plt.show()