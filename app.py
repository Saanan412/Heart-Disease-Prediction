import streamlit as st
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import joblib

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("HDscaler.pkl")
columns = joblib.load("HDcolumns.pkl")
st.title("Heart Disease Prediction")
st.markdown("Enter the following details to predict the likelihood of heart disease:")

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
     for col in columns:
         if col not in input_df.columns:
             input_df[col] = 0

     input_df = input_df[columns]  
     scaled_input = scaler.transform(input_df)
     prediction = model.predict(scaled_input)[0]
     if prediction == 1:
         st.error("High Risk of Heart Disease. Please consult a doctor.")
     else:
         st.success("Low Risk of Heart Disease. Keep maintaining a healthy lifestyle.")