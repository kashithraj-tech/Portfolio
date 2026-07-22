import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# APP TITLE
# =========================
st.title("💳 Credit Card Fraud Detection System")

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "fraud_detection_model.pkl")

model = joblib.load(model_path)
st.success("Model Loaded Successfully ✅")

# =========================
# LOAD TRAIN DATA (FOR MAPPING)
# =========================
train_path = os.path.join(BASE_DIR, "fraudTrain.csv")
train_df = pd.read_csv(train_path).sample(5000, random_state=42)

# =========================
# SAFE ENCODING FUNCTION (NO CRASH)
# =========================
def safe_encode(input_df, reference_df):
    categorical_cols = reference_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for col in categorical_cols:
        if col in input_df.columns:

            # Create mapping from training data
            unique_vals = reference_df[col].astype(str).unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}

            # Apply mapping safely
            input_df[col] = input_df[col].astype(str).map(mapping)

            # Handle unseen values (VERY IMPORTANT FIX)
            input_df[col] = input_df[col].fillna(-1)

    return input_df

# =========================
# USER INPUT
# =========================
st.header("Enter Transaction Details")

cc_num = st.text_input("Credit Card Number")
merchant = st.text_input("Merchant")
category = st.text_input("Category")
amt = st.number_input("Amount", min_value=0.0)
gender = st.text_input("Gender (M/F)")
city = st.text_input("City")
state = st.text_input("State")
zip_code = st.number_input("Zip Code", min_value=0)
lat = st.number_input("Customer Latitude")
long = st.number_input("Customer Longitude")
city_pop = st.number_input("City Population")
job = st.text_input("Job")
unix_time = st.number_input("Unix Time")
merch_lat = st.number_input("Merchant Latitude")
merch_long = st.number_input("Merchant Longitude")

# =========================
# PREDICTION
# =========================
if st.button("Predict Fraud Risk"):

    # Create dataframe
    input_df = pd.DataFrame([{
        "cc_num": cc_num,
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "gender": gender,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "job": job,
        "unix_time": unix_time,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
    }])

    # =========================
    # FEATURE ENGINEERING
    # =========================
    input_df["distance"] = np.sqrt(
        (input_df["lat"] - input_df["merch_lat"])**2 +
        (input_df["long"] - input_df["merch_long"])**2
    )

    # =========================
    # SAFE ENCODING (FIXED)
    # =========================
    input_df = safe_encode(input_df, train_df)

    # =========================
    # ALIGN COLUMNS
    # =========================
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Result")

    if prediction == 1:
        st.error("🚨 FRAUD TRANSACTION DETECTED")
    else:
        st.success("✅ LEGITIMATE TRANSACTION")

    st.write("Fraud Probability:", round(probability, 4))

    risk = probability * 100

    if risk < 30:
        st.info("Risk Level: LOW 🟢")
    elif risk < 70:
        st.warning("Risk Level: MEDIUM 🟡")
    else:
        st.error("Risk Level: HIGH 🔴")