import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# LOAD TRAINED MODELS
# -----------------------------------

# Fraud Detection Model
model = joblib.load("fraud_model.pkl")

# NLP Complaint Classification Model
nlp_model = joblib.load("nlp_model.pkl")

# Vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------------
# APP TITLE
# -----------------------------------

st.title("🏦 AI Banking Assistant")

# ===================================
# FRAUD DETECTION SECTION
# ===================================

st.header("💳 Fraud Detection System")

# User Inputs
amount = st.number_input(
    "Enter Transaction Amount",
    min_value=0.0
)

time = st.number_input(
    "Enter Transaction Time",
    min_value=0.0
)

# Predict Fraud Button
if st.button("Predict Fraud"):

    # Create sample input with 30 features
    sample_data = [[0] * 30]

    # Set Time feature
    sample_data[0][0] = time

    # Set Amount feature
    sample_data[0][-1] = amount

    # Convert to DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Predict using trained model
    prediction = model.predict(sample_df)

    # Show Result
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")

    else:
        st.success("✅ Genuine Transaction")

# ===================================
# NLP COMPLAINT CLASSIFICATION
# ===================================

st.header("🧠 NLP Complaint Classification")

# User Complaint Input
complaint = st.text_area(
    "Enter Customer Complaint"
)

# Predict Complaint Button
if st.button("Classify Complaint"):

    # Convert text into vector
    complaint_vector = vectorizer.transform([complaint])

    # Predict using NLP model
    prediction = nlp_model.predict(complaint_vector)

    # Show Prediction
    st.success(
        f"Predicted Complaint Type: {prediction[0]}"
    )