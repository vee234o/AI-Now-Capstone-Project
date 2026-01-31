import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")

# Use absolute path detection to ensure it works on deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("Insurance Claim Predictor")
st.write("Enter building details below to predict the probability of an insurance claim.")

# Function to load the already saved models
@st.cache_resource
def load_saved_assets():
    # Build paths that work on both local and deployment servers
    # This looks for the 'models' folder at the same level as the 'scripts' folder
    model_path = os.path.join(BASE_DIR, '..', 'models', 'claim_predictor.pkl')
    scaler_path = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        st.error("Saved models not found.")
        st.info(f"The app looked in: {os.path.abspath(model_path)}")
        st.stop()
        return None, None

# Load the existing models
model, scaler = load_saved_assets()

# User Interface for Inputs
col1, col2 = st.columns(2)

with col1:
    year_obs = st.number_input("Year of Observation", 2010, 2025, 2014)
    insured_period = st.slider("Insured Period (fraction of year)", 0.0, 1.0, 1.0)
    residential = st.selectbox("Is the building residential?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    building_dim = st.number_input("Building Dimension (m2)", 1.0, 30000.0, 500.0)
    date_occupancy = st.number_input("Year of First Occupancy", 1800, 2024, 1960)

with col2:
    painted = st.selectbox("Building Painted", ["N", "V"], help="N: Painted, V: Not Painted")
    fenced = st.selectbox("Building Fenced", ["N", "V"], help="N: Fenced, V: Not Fenced")
    garden = st.selectbox("Garden Status", ["O", "V"], help="V: Has Garden, O: No Garden")
    settlement = st.selectbox("Settlement Area", ["R", "U"], help="R: Rural, U: Urban")
    building_type = st.selectbox("Building Type", [1, 2, 3, 4])
    num_windows = st.number_input("Number of Windows", 0, 50, 3)
    geo_code = st.number_input("Geographical Code", 0, 99999, 1053)

# Mapping textual categories to numeric values exactly as trained
mapping = {"N": 0, "V": 1, "O": 0, "R": 0, "U": 1}

# Prepare input data in the exact order the model expects
input_df = pd.DataFrame([{
    'YearOfObservation': year_obs,
    'Insured_Period': insured_period,
    'Residential': residential,
    'Building_Painted': mapping[painted],
    'Building_Fenced': mapping[fenced],
    'Garden': mapping[garden],
    'Settlement': mapping[settlement],
    'Building Dimension': building_dim,
    'Building_Type': building_type,
    'Date_of_Occupancy': float(date_occupancy),
    'NumberOfWindows': float(num_windows),
    'Geo_Code': geo_code
}])

st.divider()

if st.button("Predict Risk"):
    # Apply the saved scaler to the input data
    scaled_input = scaler.transform(input_df)
    
    # Generate prediction and probability using the loaded model
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error("Result: High Risk Policy")
        st.write(f"The model estimates a {probability:.2%} probability of a claim.")
    else:
        st.success("Result: Low Risk Policy")
        st.write(f"The model estimates a {probability:.2%} probability of a claim.")