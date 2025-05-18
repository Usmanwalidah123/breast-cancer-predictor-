import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Function to load model and scaler
def load_model():
    try:
        with open('svm_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'svm_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

# Load the trained model and scaler
model, scaler = load_model()

# If model and scaler are loaded successfully
if model and scaler:
    # Streamlit app title
    st.title("Breast Cancer Prediction App")

    # Input fields for user data
    age = st.number_input("Age", min_value=20, max_value=100)
    bmi = st.number_input("BMI", min_value=10, max_value=50)
    glucose = st.number_input("Glucose", min_value=0, max_value=300)
    insulin = st.number_input("Insulin", min_value=0, max_value=300)
    homa = st.number_input("HOMA", min_value=0.0, max_value=10.0)
    leptin = st.number_input("Leptin", min_value=0.0, max_value=100.0)
    adiponectin = st.number_input("Adiponectin", min_value=0.0, max_value=100.0)
    resistin = st.number_input("Resistin", min_value=0.0, max_value=100.0)
    mcp1 = st.number_input("MCP.1", min_value=0.0, max_value=100.0)

    # Predict button
    if st.button("Predict"):
        input_data = np.array([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp1]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        
        if prediction[0] == 1:
            st.success("Prediction: Healthy")
        else:
            st.error("Prediction: Patient")
