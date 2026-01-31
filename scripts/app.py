import streamlit as st
import pandas as pd
import joblib

st.title("Insurance Claim Predictor")

try:
    model = joblib.load('models/claim_predictor.pkl')
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please train the model first.")
    st.stop()

year_obs = st.number_input("Year of Observation", 2010, 2030, 2014)
insured_period = st.number_input("Insured Period", 0.0, 1.0, 1.0)
residential = st.selectbox("Residential", [0, 1])
painted = st.selectbox("Building Painted", ["N", "V"])
fenced = st.selectbox("Building Fenced", ["N", "V"])
garden = st.selectbox("Garden", ["O", "V"])
settlement = st.selectbox("Settlement", ["R", "U"])
building_dim = st.number_input("Building Dimension", 0.0, 20000.0, 500.0)
building_type = st.selectbox("Building Type", [1, 2, 3, 4])
date_occupancy = st.number_input("Date of Occupancy", 1800, 2030, 1960)
num_windows = st.number_input("Number of Windows", 0, 100, 3)
geo_code = st.number_input("Geo Code", 0, 100000, 1000)

painted_val = 0 if painted == "N" else 1
fenced_val = 0 if fenced == "N" else 1
garden_val = 0 if garden == "O" else 1
settlement_val = 0 if settlement == "R" else 1

input_data = pd.DataFrame([{
    'YearOfObservation': year_obs,
    'Insured_Period': insured_period,
    'Residential': residential,
    'Building_Painted': painted_val,
    'Building_Fenced': fenced_val,
    'Garden': garden_val,
    'Settlement': settlement_val,
    'Building Dimension': building_dim,
    'Building_Type': building_type,
    'Date_of_Occupancy': date_occupancy,
    'NumberOfWindows': num_windows,
    'Geo_Code': geo_code
}])

if st.button("Predict"):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    if prediction == 1:
        st.write(f"High Risk. Probability: {probability:.2%}")
    else:
        st.write(f"Low Risk. Probability: {probability:.2%}")