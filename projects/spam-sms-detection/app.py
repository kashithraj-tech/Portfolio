# ==========================================
# SMART SPAM SMS DETECTION SYSTEM
# CodSoft AI Internship Project
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import re
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Spam SMS Detector",
    page_icon="📩",
    layout="wide"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1{
    color:#1E3A8A;
    text-align:center;
}

.stButton>button{
    background-color:#2563EB;
    color:white;
    border-radius:10px;
    height:50px;
    width:220px;
    font-size:18px;
}

.stButton>button:hover{
    background-color:#1D4ED8;
}

textarea{
    border-radius:10px;
}

</style>
""",unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📩 Spam SMS Detector")

st.sidebar.markdown("---")

st.sidebar.subheader("Project Information")

st.sidebar.write("**Internship:** CodSoft")

st.sidebar.write("**Project:** Spam SMS Detection")

st.sidebar.write("**Developer:** Kashithra J")

st.sidebar.markdown("---")

st.sidebar.subheader("Machine Learning")

st.sidebar.success("✔ TF-IDF Feature Extraction")

st.sidebar.success("✔ Logistic Regression")

st.sidebar.success("✔ Explainable AI")

st.sidebar.success("✔ Spam Probability")

st.sidebar.success("✔ Keyword Detection")

st.sidebar.markdown("---")

st.sidebar.info(
"""
This system classifies SMS messages into:

📩 Legitimate (Ham)

🚨 Spam
"""
)

# ==========================================
# MAIN TITLE
# ==========================================

st.title("📩 Smart Spam SMS Detection System")

st.markdown(
"""
### AI Powered Spam Detection using Machine Learning

This application predicts whether an SMS message is **Spam** or **Legitimate** using **TF-IDF** and **Logistic Regression**.
"""
)

st.markdown("---")

message = st.text_area(
    "📨 Enter SMS Message",
    height=170,
    placeholder="Type or paste an SMS here..."
)

analyze = st.button("🚀 Analyze Message")

# ==========================================
# ANALYZE MESSAGE
# ==========================================

spam_keywords = [
    "win","won","free","offer","cash",
    "urgent","claim","prize","click",
    "congratulations","reward"
]

if analyze:

    if message.strip() == "":
        st.warning("Please enter an SMS message.")
    else:

        # Convert text into TF-IDF features
        transformed = vectorizer.transform([message])

        # Predict
        prediction = model.predict(transformed)[0]

        # Probability (only if Logistic Regression is used)
        probability = model.predict_proba(transformed)[0]

        spam_prob = round(probability[1] * 100, 2)
        ham_prob = round(probability[0] * 100, 2)

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"🚨 SPAM Message ({spam_prob}%)")
        else:
            st.success(f"✅ Legitimate Message ({ham_prob}%)")

        st.subheader("Spam Probability")

        st.progress(int(spam_prob))

        col1, col2 = st.columns(2)

        col1.metric("Spam Probability", f"{spam_prob}%")
        col2.metric("Ham Probability", f"{ham_prob}%")

        st.markdown("---")

        st.subheader("Message Statistics")

        words = len(message.split())
        characters = len(message)
        numbers = len(re.findall(r"\d", message))
        special = len(re.findall(r"[^A-Za-z0-9 ]", message))

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Words", words)
        c2.metric("Characters", characters)
        c3.metric("Numbers", numbers)
        c4.metric("Special Characters", special)

        st.markdown("---")

        st.subheader("Detected Spam Keywords")

        detected = []

        msg = message.lower()

        for word in spam_keywords:
            if word in msg:
                detected.append(word)

        if detected:
            for word in detected:
                st.write("🔴", word)
        else:
            st.success("No suspicious keywords found.")

        # Save Prediction History
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Message": message,
            "Prediction": "Spam" if prediction == 1 else "Ham",
            "Spam Probability": spam_prob
        })

        st.markdown("---")

        st.subheader("Prediction History")

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction History",
            csv,
            "prediction_history.csv",
            "text/csv"
        )