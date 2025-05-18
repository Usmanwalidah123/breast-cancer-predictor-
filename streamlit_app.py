import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris  # Example dataset for demonstration

# Function to train the model
def train_model():
    # Load your data here, for example using a dataset
    # Replace this with your actual dataset
    data = load_iris(/content/dataR2.csv")  # Placeholder for your breast cancer dataset
    X = data.data  # Features
    y = data.target  # Labels

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the SVM model
    model = SVC(kernel='rbf', C=1.0)
    model.fit(X_train, y_train)

    return model, scaler

# Train the model and scaler
model, scaler = train_model()

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
