# ============================================
# CUSTOMER CHURN PREDICTION SYSTEM
# Flask Application
# ============================================

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# ============================================
# Load Model and Encoders
# ============================================

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
geo_encoder = joblib.load("geo_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

# ============================================
# Home Page
# ============================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================
# Prediction
# ============================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -----------------------------
        # Get User Input
        # -----------------------------

        credit_score = int(request.form["credit_score"])
        geography = request.form["geography"]
        gender = request.form["gender"]
        age = int(request.form["age"])
        tenure = int(request.form["tenure"])
        balance = float(request.form["balance"])
        products = int(request.form["products"])
        credit_card = int(request.form["creditcard"])
        active = int(request.form["active"])
        salary = float(request.form["salary"])

        # -----------------------------
        # Encode
        # -----------------------------

        geography = geo_encoder.transform([geography])[0]
        gender = gender_encoder.transform([gender])[0]

        # -----------------------------
        # Create DataFrame
        # -----------------------------

        data = pd.DataFrame([[

            credit_score,
            geography,
            gender,
            age,
            tenure,
            balance,
            products,
            credit_card,
            active,
            salary

        ]], columns=[

            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"

        ])

        # -----------------------------
        # Scale
        # -----------------------------

        scaled = scaler.transform(data)

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(scaled)[0]

        probability = model.predict_proba(scaled)[0][1]

        probability = round(probability * 100, 2)

        # -----------------------------
        # Risk Level
        # -----------------------------

        if probability >= 75:

            risk = "HIGH 🔴"

        elif probability >= 40:

            risk = "MEDIUM 🟠"

        else:

            risk = "LOW 🟢"

        # -----------------------------
        # Prediction Text
        # -----------------------------

        if prediction == 1:

            result = "⚠ Customer Will Churn"

        else:

            result = "✅ Customer Will Stay"

        # -----------------------------
        # Recommendation
        # -----------------------------

        if probability >= 75:

            recommendation = (
                "Offer premium discounts, assign a relationship manager "
                "and contact the customer immediately."
            )

        elif probability >= 40:

            recommendation = (
                "Offer loyalty rewards and personalized offers."
            )

        else:

            recommendation = (
                "Maintain customer satisfaction through regular engagement."
            )

        # -----------------------------
        # Save Prediction History
        # -----------------------------

        history = pd.DataFrame({

            "Date":[datetime.now().strftime("%d-%m-%Y %H:%M")],
            "Probability":[probability],
            "Risk":[risk],
            "Prediction":[result]

        })

        if os.path.exists("prediction_history.csv"):

            history.to_csv(
                "prediction_history.csv",
                mode="a",
                header=False,
                index=False
            )

        else:

            history.to_csv(
                "prediction_history.csv",
                index=False
            )

        # -----------------------------
        # Return Output
        # -----------------------------

        return render_template(

            "index.html",

            prediction=result,

            probability=probability,

            risk=risk,

            recommendation=recommendation

        )

    except Exception as e:

        return render_template(

            "index.html",

            prediction="Error",

            probability=0,

            risk="-",

            recommendation=str(e)

        )


# ============================================
# Run
# ============================================

if __name__ == "__main__":

    app.run(debug=True)