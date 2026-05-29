import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="msubburao/predictivemaintenancemodel", filename="predmaintain_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Engine Health Prediction App")
st.write("Engine Health Prediction App is a tool used to predict the health of the Vehicle's Engine using Prdictive Maintenance Machine Learning Model")
st.write("Kindly enter the Engine details to check it's health and if Vehicle needs any maintenance")

# Collect user input

engine_rpm = st.number_input("Engine RPM", min_value=0.0, value=750.0)
lub_oil_pressure = st.number_input("Lube Oil Pressure", min_value=0.0, value=3.0)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, value=6.0)
coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0, value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", min_value=0.0, value=77.0)
coolant_temp = st.number_input("Coolant Temperature", min_value=0.0, value=78.0)

input_data = pd.DataFrame({
    "engine_rpm": [engine_rpm],
    "lub_oil_pressure": [lub_oil_pressure],
    "fuel_pressure": [fuel_pressure],
    "coolant_pressure": [coolant_pressure],
    "lub_oil_temp": [lub_oil_temp],
    "coolant_temp": [coolant_temp]
})

# # Set the classification threshold
classification_threshold = 0.45

if st.button("Predict Engine Condition"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (
        prediction_proba >= classification_threshold
    ).astype(int)
    st.write(
        f"Maintenance Probability: {prediction_proba:.2%}"
    )
    if prediction == 1:
        st.error("Prediction: Engine Requires Maintenance")
    else:
        st.success("Prediction: Engine Operating Normally")
