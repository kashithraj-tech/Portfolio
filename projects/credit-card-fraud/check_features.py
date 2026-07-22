import joblib

model = joblib.load("fraud_detection_model.pkl")

print(model.feature_names_in_)