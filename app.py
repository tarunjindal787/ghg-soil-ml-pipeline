import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Soil Property Predictor", layout="centered")
st.title("🌱 GHG Soil Property ML Predictor")
st.write("Move the sliders below to predict soil moisture and temperature.")

# Point to your trained model files
MOISTURE_PATH = "outputs/ghg_ml_pipeline_outputs/soil_moisture_model.joblib"
TEMP_PATH = "outputs/ghg_ml_pipeline_outputs/soil_temperature_model.joblib"

if not os.path.exists(MOISTURE_PATH) or not os.path.exists(TEMP_PATH):
    st.error("⚠️ Model binaries missing. Please run your training pipeline script first!")
else:
    # Load your pipeline models safely
    moisture_model = joblib.load(MOISTURE_PATH)
    temp_model = joblib.load(TEMP_PATH)

    st.subheader("📊 Control Inputs")
    month = st.slider("Month", 1, 12, 6)
    day_of_year = st.slider("Day of Year", 1, 365, 180)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=1)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"], index=2)
    is_growing = st.radio("Growing Season?", ["Yes", "No"])
    
    # Convert selection to binary format matching your feature logic
    growing_val = 1 if is_growing == "Yes" else 0

    # Match the exact DataFrame structure your pipeline expects
    input_df = pd.DataFrame([{
        "month": month,
        "day_of_year": day_of_year,
        "quarter": quarter,
        "season": season,
        "is_growing_season": growing_val
    }])

    st.markdown("---")

    if st.button("🚀 Run Prediction", type="primary"):
        # Make predictions using the loaded pipelines
        pred_moisture = moisture_model.predict(input_df)[0]
        pred_temp = temp_model.predict(input_df)[0]

        # Display output metrics side-by-side
        col1, col2 = st.columns(2)
        col1.metric("Predicted Soil Moisture (5cm)", f"{pred_moisture:.2f}%")
        col2.metric("Predicted Soil Temperature (5cm)", f"{pred_temp:.2f} °C")
