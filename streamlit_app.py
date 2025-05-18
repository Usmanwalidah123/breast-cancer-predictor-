import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

# Function to load model and scaler
def load_model():
    model_file = 'svm_model.pkl'
    scaler_file = 'scaler.pkl'
    
    if not os.path.exists(model_file):
        st.error(f"Model file '{model_file}' not found.")
        return None, None
    
    if not os.path.exists(scaler_file):
        st.error(f"Scaler file '{scaler_file}' not found.")
        return None, None
    
    try:
        with open(model_file, 'rb') as mf:
            model = pickle.load(mf)
        
        with open(scaler_file, 'rb') as sf:
            scaler = pickle.load(sf)
        
        return model, scaler

    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

# Load the trained model and scaler
model, scaler = load_model()

# If model and scaler are loaded successfully
if model and scaler:
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
