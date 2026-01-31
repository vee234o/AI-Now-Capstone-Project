import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("Insurance Claim Predictor")

@st.cache_resource
def get_model():
    model_path = 'models/claim_predictor.pkl'
    scaler_path = 'models/scaler.pkl'

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        try:
            df = pd.read_csv('data/Train_data.csv')
        except FileNotFoundError:
            st.error("Critical Error: data/Train_data.csv not found.")
            st.stop()

        df['NumberOfWindows'] = df['NumberOfWindows'].astype(str).str.strip().replace('.', np.nan)
        df['NumberOfWindows'] = pd.to_numeric(df['NumberOfWindows'], errors='coerce')

        for col in ['Building Dimension', 'Date_of_Occupancy', 'NumberOfWindows']:
            df[col] = df[col].fillna(df[col].median())

        for col in ['Garden', 'Geo_Code', 'Building_Fenced', 'Building_Painted']:
            df[col] = df[col].fillna(df[col].mode()[0])

        df = df.drop('Customer Id', axis=1)

        le = LabelEncoder()
        cols = ['Garden', 'Building_Fenced', 'Building_Painted', 'Settlement', 'Geo_Code', 'Building_Type']
        for col in cols:
            df[col] = le.fit_transform(df[col])

        X = df.drop('Claim', axis=1)
        y = df['Claim']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        return model, scaler

model, scaler = get_model()

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