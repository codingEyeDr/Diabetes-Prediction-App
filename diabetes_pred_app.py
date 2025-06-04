import streamlit as st
import numpy as np
import joblib

# Load the trained model using joblib
model = joblib.load("diabetics_pred.pkl")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🧬",
    layout="centered"
)

# App Title
st.title("🧬 Diabetes Prediction Web App")
st.markdown("""
This app uses a trained machine learning model to predict whether a person has **diabetes** based on health data.

👉 Enter the details below and click **Predict** to see the result.
""")

# Input form
st.header("🔍 Enter Patient Details")

pregnancies = st.slider("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Level", 0, 200, 110)
blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 140, 70)
skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 80)
bmi = st.slider("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 10, 100, 33)

# Prediction button
if st.button("🔎 Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🔴 Result: The person **has diabetes**.")
    else:
        st.success("🟢 Result: The person **does NOT have diabetes**.")
