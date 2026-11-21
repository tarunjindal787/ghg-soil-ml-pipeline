import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Soil Property Predictor", layout="centered")
st.title("🌱 GHG Soil Property ML Predictor")
st.write("Fill in a field observation below to predict soil moisture and temperature at 5cm depth.")

MOISTURE_PATH = "outputs/ghg_ml_pipeline_outputs/soil_moisture_model.joblib"
TEMP_PATH = "outputs/ghg_ml_pipeline_outputs/soil_temperature_model.joblib"

SITES = ['ARL', 'BRADFORD.A', 'FREEMAN', 'HICKS.B', 'HICKS.G', 'HICKS.P', 'HOYTVILLE.LTR',
         'ISUAG', 'KELLOGG', 'LAN', 'MAR', 'MASON', 'NWREC', 'SERF', 'WATERMAN',
         'WOOSTER.COV', 'WOOSTER.LTR']
POSITIONS = ['off_fert_band', 'on_and_off_fert_band', 'on_broadcast', 'on_fert_band']

if not os.path.exists(MOISTURE_PATH) or not os.path.exists(TEMP_PATH):
    st.error("⚠️ Model binaries missing. Please run your training pipeline script first!")
else:
    moisture_model = joblib.load(MOISTURE_PATH)
    temp_model = joblib.load(TEMP_PATH)

    st.subheader("📍 Site & Sampling")
    col1, col2, col3 = st.columns(3)
    uniqueid = col1.selectbox("Site (uniqueid)", SITES, index=SITES.index("FREEMAN"))
    plotid = col2.number_input("Plot ID", min_value=1, max_value=4200, value=214)
    subsample = col3.selectbox("Subsample", [1, 2], index=0)

    col4, col5 = st.columns(2)
    method = col4.selectbox("Sampling method", ["GC", "PAS"])
    position = col5.selectbox("Chamber position", POSITIONS, index=POSITIONS.index("on_fert_band"))

    st.subheader("📅 Date Features")
    col6, col7, col8 = st.columns(3)
    year = col6.slider("Year", 2011, 2015, 2013)
    month = col7.slider("Month", 1, 12, 7)
    day_of_year = col8.slider("Day of Year", 1, 365, 193)

    col9, col10, col11 = st.columns(3)
    quarter = col9.selectbox("Quarter", [1, 2, 3, 4], index=2)
    season = col10.selectbox("Season", ["winter", "spring", "summer", "fall"], index=2)
    is_growing = col11.radio("Growing season?", ["Yes", "No"])
    growing_val = 1 if is_growing == "Yes" else 0

    st.subheader("💨 GHG Flux Measurements")
    col12, col13 = st.columns(2)
    co2_flux = col12.number_input("CO2-C flux (kg/ha/day)", value=12.39, format="%.3f")
    co2_r2 = col13.slider("CO2 model R2", 0.0, 1.0, 0.99)

    col14, col15 = st.columns(2)
    n2o_flux = col14.number_input("N2O-N flux (g/ha/day)", value=2.69, format="%.3f")
    n2o_r2 = col15.slider("N2O model R2", -0.5, 2.1, 0.90)

    col16, col17 = st.columns(2)
    nh3_flux = col16.number_input("NH3-N flux (g/ha/day)", value=0.0, format="%.3f")
    nh3_r2 = col17.slider("NH3 model R2", 0.0, 1.0, 0.18)

    col18, col19 = st.columns(2)
    ch4_flux = col18.number_input("CH4-C flux (g/ha/day)", value=-0.67, format="%.3f")
    ch4_r2 = col19.slider("CH4 model R2", 0.0, 1.0, 0.88)

    st.subheader("🧪 Soil Chemistry")
    col20, col21 = st.columns(2)
    nitrate = col20.number_input("Soil nitrate N (0-10cm)", min_value=0.0, value=9.06, format="%.3f")
    ammonium = col21.number_input("Soil ammonium N (0-10cm)", min_value=0.0, value=14.02, format="%.3f")

    input_df = pd.DataFrame([{
        "uniqueid": uniqueid,
        "plotid": plotid,
        "year": year,
        "method": method,
        "subsample": subsample,
        "position": position,
        "co2_c_flux_kg_ha_day": co2_flux,
        "co2_r2": co2_r2,
        "n2o_n_flux_g_ha_day": n2o_flux,
        "n2o_r2": n2o_r2,
        "nh3_n_flux_g_ha_day": nh3_flux,
        "nh3_r2": nh3_r2,
        "ch4_c_flux_g_ha_day": ch4_flux,
        "ch4_r2": ch4_r2,
        "soil_nitrate_n_0_10cm": nitrate,
        "soil_ammonium_n_0_10cm": ammonium,
        "month": month,
        "day_of_year": day_of_year,
        "quarter": quarter,
        "is_growing_season": growing_val,
        "season": season,
    }])

    st.markdown("---")

    if st.button("🚀 Run Prediction", type="primary"):
        pred_moisture = moisture_model.predict(input_df)[0]
        pred_temp = temp_model.predict(input_df)[0]

        result_col1, result_col2 = st.columns(2)
        result_col1.metric("Predicted Soil Moisture (5cm)", f"{pred_moisture:.2f}%")
        result_col2.metric("Predicted Soil Temperature (5cm)", f"{pred_temp:.2f} °C")
