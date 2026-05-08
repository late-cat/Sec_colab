import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load the saved models and scaler
@st.cache_resource
def load_assets():
    with open('models/previous-models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    ann_model = load_model('models/previous-models/model-CNN.keras')
    
    with open('models/previous-models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return rf_model, ann_model, scaler

rf_model, ann_model, scaler = load_assets()

# Set up the UI
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to leave the bank based on their profile.")
st.divider()

# Model Selection
model_choice = st.selectbox("🤖 Select Prediction Model", ["Random Forest", "Artificial Neural Network (ANN)"])


if model_choice == "Random Forest":
    st.info("**Model Stats:** Accuracy: ~86% | Type: Ensemble Learning | Fast & Interpretable")
else:
    st.info("**Model Stats:** Accuracy: ~85.7% | Type: Deep Learning | Architecture: 3 Hidden Layers")

st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.checkbox("Has Credit Card", value=True)
    is_active = st.checkbox("Is Active Member", value=True)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0)

st.divider()

# Prediction logic
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    # Process categorical inputs
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0
    cr_card = 1 if has_cr_card else 0
    active_member = 1 if is_active else 0

    # Create input array
    user_input = np.array([[
        credit_score, age, tenure, balance, num_products, 
        cr_card, active_member, estimated_salary, 
        geo_germany, geo_spain, gender_male
    ]])

    # Scale the input
    scaled_input = scaler.transform(user_input)

    # Predict
    if model_choice == "Random Forest":
        # predict_proba returns an array like [[prob_stay, prob_churn]]
        prediction_prob = rf_model.predict_proba(scaled_input)[0][1]
    else:
        # ANN predict returns a single probability scalar
        prediction_prob = ann_model.predict(scaled_input, verbose=0)[0][0]

    # Show results
    st.subheader(f"Prediction Result ({model_choice}):")
    if prediction_prob >= 0.5:
        st.error(f"🚩 High Risk of Churn! (Probability: {prediction_prob:.1%})")
    else:
        st.success(f"❇️ Customer is Safe to stay. (Probability: {prediction_prob:.1%})")
