# INTELLIGENT CREDIT CARD FRAUD DETECTION SYSTEM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# LOAD DATASET

train_path = r"C:\Users\KASHITHRA J\OneDrive\Documents\archive\fraudTrain.csv"
test_path = r"C:\Users\KASHITHRA J\OneDrive\Documents\archive\fraudTest.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("Training Shape :", train_df.shape)
print("Testing Shape  :", test_df.shape)

# SAMPLE DATA FOR FASTER EXECUTION

train_df = train_df.sample(5000, random_state=42)
test_df = test_df.sample(2000, random_state=42)

print("\nAfter Sampling")
print("Training Shape :", train_df.shape)
print("Testing Shape  :", test_df.shape)

# FEATURE ENGINEERING
# DISTANCE BETWEEN CUSTOMER & MERCHANT

train_df["distance"] = np.sqrt(
    (train_df["lat"] - train_df["merch_lat"])**2 +
    (train_df["long"] - train_df["merch_long"])**2
)

test_df["distance"] = np.sqrt(
    (test_df["lat"] - test_df["merch_lat"])**2 +
    (test_df["long"] - test_df["merch_long"])**2
)

# FRAUD DISTRIBUTION

plt.figure(figsize=(6,4))
sns.countplot(x="is_fraud", data=train_df)
plt.title("Fraud vs Legitimate Transactions")
plt.show(block=False)
plt.pause(2)
plt.close()

# FRAUD ANALYSIS

fraud_data = train_df[train_df["is_fraud"] == 1]

plt.figure(figsize=(10,5))
fraud_data["category"].value_counts().head(10).plot(kind="bar")
plt.title("Top Fraud Categories")
plt.show(block=False)
plt.pause(2)
plt.close()

plt.figure(figsize=(10,5))
fraud_data["state"].value_counts().head(10).plot(kind="bar")
plt.title("Top Fraud States")
plt.show(block=False)
plt.pause(2)
plt.close()

plt.figure(figsize=(10,5))
fraud_data["merchant"].value_counts().head(10).plot(kind="bar")
plt.title("Top Fraud Merchants")
plt.show(block=False)
plt.pause(2)
plt.close()

# DROP UNNECESSARY COLUMNS

drop_columns = [
    "trans_date_trans_time",
    "first",
    "last",
    "street",
    "dob",
    "trans_num"
]

for col in drop_columns:

    if col in train_df.columns:
        train_df.drop(col, axis=1, inplace=True)

    if col in test_df.columns:
        test_df.drop(col, axis=1, inplace=True)

# LABEL ENCODING

categorical_columns = train_df.select_dtypes(
    include=["object", "string", "category"]
).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    combined = pd.concat([
        train_df[column],
        test_df[column]
    ]).astype(str)

    encoder.fit(combined)

    train_df[column] = encoder.transform(
        train_df[column].astype(str)
    )

    test_df[column] = encoder.transform(
        test_df[column].astype(str)
    )

# FEATURES AND TARGET

X_train = train_df.drop("is_fraud", axis=1)
y_train = train_df["is_fraud"]

X_test = test_df.drop("is_fraud", axis=1)
y_test = test_df["is_fraud"]

# LOGISTIC REGRESSION

print("Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=500,
    solver='liblinear'
)
print("Starting Logistic Regression Training...")
lr_model.fit(X_train, y_train)
print("Logistic Regression Training Completed")

print("Making Predictions...")
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print("Logistic Regression Prediction Completed")

# DECISION TREE

print("Training Decision Tree...")

dt_model = DecisionTreeClassifier(
    random_state=42
)

print("Starting Decision Tree Training...")
dt_model.fit(X_train, y_train)
print("Decision Tree Training Completed")

dt_pred = dt_model.predict(X_test)
print("Decision Tree Prediction Completed")

dt_acc = accuracy_score(y_test, dt_pred)

# RANDOM FOREST

print("Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

import time

start = time.time()

print("Starting Random Forest Training...")

rf_model.fit(X_train, y_train)

print("Random Forest Training Completed")

print("Random Forest Time:",
      round(time.time() - start, 2),
      "seconds")

rf_pred = rf_model.predict(X_test)

print("Random Forest Prediction Completed")

rf_acc = accuracy_score(y_test, rf_pred)# ROC AUC SCORE

rf_prob = rf_model.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, rf_prob)

# MODEL COMPARISON

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
]

scores = [
    lr_acc * 100,
    dt_acc * 100,
    rf_acc * 100
]

plt.figure(figsize=(8,5))
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")

for i, score in enumerate(scores):
    plt.text(i, score + 0.3,
             str(round(score,2)))

plt.show(block=False)
plt.pause(2)
plt.close()

# RANDOM FOREST REPORT

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

print("\nROC-AUC Score :", round(roc_score,4))

# CONFUSION MATRIX

cm = confusion_matrix(y_test, rf_pred)

labels = np.array([
    [
        f"True Negative\n{cm[0,0]}",
        f"False Positive\n{cm[0,1]}"
    ],
    [
        f"False Negative\n{cm[1,0]}",
        f"True Positive\n{cm[1,1]}"
    ]
])

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=labels,
    fmt="",
    cmap="Greens",
    linewidths=2,
    linecolor="black",
    cbar=True,
    annot_kws={
        "size":14,
        "weight":"bold"
    }
)

plt.title(
    "Fraud Detection Confusion Matrix",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Predicted Class",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Actual Class",
    fontsize=14,
    fontweight="bold"
)

plt.xticks(
    [0.5, 1.5],
    ["Legitimate", "Fraud"],
    fontsize=12
)

plt.yticks(
    [0.5, 1.5],
    ["Legitimate", "Fraud"],
    fontsize=12,
    rotation=0
)

plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()

# FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features")
print(feature_importance.head(15))

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"].head(15),
    feature_importance["Importance"].head(15)
)

plt.gca().invert_yaxis()
plt.title("Top 15 Important Features")
plt.show(block=False)
plt.pause(2)
plt.close()

# FRAUD RISK SCORE

sample_transaction = X_test.iloc[[0]]

fraud_probability = rf_model.predict_proba(
    sample_transaction
)[0][1]

risk_score = round(fraud_probability * 100, 2)

if risk_score < 30:
    risk_level = "LOW RISK"
elif risk_score < 70:
    risk_level = "MEDIUM RISK"
else:
    risk_level = "HIGH RISK"

print("FRAUD RISK ANALYSIS")

print("Fraud Probability :", round(fraud_probability,4))
print("Risk Score        :", risk_score)
print("Risk Category     :", risk_level)

# SAVE MODEL

joblib.dump(
    rf_model,
    "fraud_detection_model.pkl"
)

print("\nModel Saved Successfully")

# FINAL SUMMARY

print("PROJECT SUMMARY")

print("Logistic Regression Accuracy :",
      round(lr_acc*100,2), "%")

print("Decision Tree Accuracy :",
      round(dt_acc*100,2), "%")

print("Random Forest Accuracy :",
      round(rf_acc*100,2), "%")

print("ROC-AUC Score :",
      round(roc_score,4))

print("\nBest Model : Random Forest")