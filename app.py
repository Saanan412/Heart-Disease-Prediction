import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignore the version warning so it doesn't clutter the app
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import joblib

# Load the model and scaler
# Note: The warning in the terminal is okay, the code will still run!
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("HDscaler.pkl")
columns = joblib.load("HDcolumns.pkl")

st.title("Heart Disease Prediction")
st.markdown("Enter the following details to predict the likelihood of heart disease:")

# User Inputs
Age = st.number_input("Age", min_value=1, max_value=120, value=30)
Sex = st.selectbox("Sex", options=["M", "F"])
ChestPainType = st.selectbox("Chest Pain Type", options=["TA", "ATA", "NAP", "ASY"])
RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=1, max_value=300, value=120)
Cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=1, max_value=600, value=200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1])
RestingECG = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=1, max_value=250, value=150)
ExerciseAngina = st.selectbox("Exercise Induced Angina", options=["Y", "N"])
Oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0)
ST_Slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

if st.button("Predict"):
    st.write("Button clicked! Starting prediction...")
    try:
        raw_input = {
            'Age': Age,
            'RestingBP': RestingBP,
            'Cholesterol': Cholesterol,
            'FastingBS': FastingBS,
            'MaxHR': MaxHR,
            'Oldpeak': Oldpeak,
            'Sex_' + Sex: 1,
            'ChestPainType_' + ChestPainType: 1,
            'RestingECG_' + RestingECG: 1,
            'ExerciseAngina_' + ExerciseAngina: 1,
            'ST_Slope_' + ST_Slope: 1
        }
        
        input_df = pd.DataFrame([raw_input])
        
        # Add missing columns
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder
        input_df = input_df[columns]  
        
        # Scale
        scaled_input = scaler.transform(input_df)
        
        # Predict (Using predict_proba for better SVC handling)
        probability = model.predict_proba(scaled_input)[0]
        prediction = model.predict(scaled_input)[0]

        # Debugging: Print the raw values to the screen so you know it's working
        st.write(f"**Debug:** Raw Prediction value: {prediction}")
        st.write(f"**Debug:** Probability of Disease: {probability[1]:.2%}")
        st.write(f"**Debug:** Input Columns Used: {input_df.columns.tolist()}")

        # Display Result based on probability
        if probability[1] > 0.5:
            st.error(f"High Risk of Heart Disease. (Probability: {probability[1]:.2%})")
        else:
            st.success(f"Low Risk of Heart Disease. (Probability: {probability[1]:.2%})")
            
    except Exception as e:
        # This will display the ACTUAL error on the screen!
        st.error(f"ERROR: {e}")
        st.write("Please check the terminal for the full traceback.")