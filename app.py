import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor

@st.cache_resource
def load_data():
    data = pd.read_csv("Cars_Data.csv")
    return data

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("best_catboost_model.cbm")
    return model

@st.cache_resource
def load_expected_columns():
    return joblib.load("expected_columns2.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler2.pkl")

data = load_data()
model = load_model()
expected_columns = load_expected_columns()
scaler = load_scaler()

# Custom CSS Injection (Background, Dimming Overlay, Global Styles, and Modified Button & Prediction Card)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"], .stApp {
         background: url('https://www.motorfinanceonline.com/wp-content/uploads/sites/6/2025/01/carwow-shutterstock_2356848413.jpg') no-repeat center center fixed !important;
         background-size: cover !important;
         position: relative;
         z-index: 0;
    }
    [data-testid="stAppViewContainer"]::before {
         content: "";
         position: absolute;
         top: 0 !important;
         left: 0 !important;
         right: 0 !important;
         bottom: 0 !important;
         background: rgba(0, 0, 0, 0.75) !important;
         z-index: -1 !important;
    }
    
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p {
         color: #ffffff !important;
         color-scheme: dark !important;
         font-family: 'Inter', sans-serif !important;
    }
    
    .input-container {
         font-family: 'Inter', sans-serif !important;
         background: #121212 !important;
         padding: 20px !important;
         border-radius: 10px !important;
         margin-bottom: 20px !important;
    }
    .stButton>button {
         font-family: 'Inter', sans-serif !important;
         width: 75% !important;
         margin: 0 auto !important;
         margin-top: 36px !important;
         display: block !important;
         background: linear-gradient(135deg, #FF4B4B, #FF7F7F) !important;
         color: white !important;
         font-weight: bold !important;
         padding: 12px 24px !important;
         border-radius: 8px !important;
         font-size: 16px !important;
         transition: transform 0.3s ease;
         border: none !important;
         cursor: pointer !important;
    }
    .stButton>button:hover {
         transform: scale(1.05);
    }
    .prediction-card {
         font-family: 'Inter', sans-serif !important;
         background: #222 !important;
         border-radius: 12px !important;
         padding: 30px !important;
         text-align: center !important;
         font-size: 28px !important;
         font-weight: 600 !important;
         color: #FF4B4B !important;
         box-shadow: 0 6px 12px rgba(0, 0, 0, 0.6) !important;
         margin-top: 20px !important;
    }
    /* Footer styling */
    .footer {
         font-family: 'Inter', sans-serif !important;
         text-align: center !important;
         color: #777 !important;
         margin-top: 40px !important;
         font-size: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Used Car Price Predictor")

# Input Fields in Main Area
st.markdown("### Enter Car Specifications")
col1, col2 = st.columns(2)
with col1:
    brand_input = st.selectbox("Brand", data["brand"].unique().tolist())
    models_for_brand = data[data["brand"] == brand_input]["model"].unique().tolist()
    model_input = st.selectbox("Model", models_for_brand)
    power_ps = st.number_input("Power (PS)", min_value=50, value=150)
    color_input = st.selectbox("Color", data["color"].unique().tolist())
    power_kw = power_ps*0.735 
with col2:
    fuel_type_input = st.selectbox("Fuel Type", data["fuel_type"].unique().tolist())
    transmission_input = st.selectbox("Transmission", [t for t in data["transmission_type"].unique().tolist() if t != "Unknown"])
    # transmission_input = st.selectbox("Transmission", data["transmission_type"].unique().tolist())
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=250000, value=50000)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=5)
    fuel_consumption = 7

# Raw Input DataFrame
raw_input = pd.DataFrame([{
    "power_ps": power_ps,
    "fuel_consumption_l_100km.1": fuel_consumption,
    "mileage_in_km": mileage,
    "vehicle_age": vehicle_age,
    "brand": brand_input,
    "model": model_input,
    "color": color_input,
    "transmission_type": transmission_input,
    "fuel_type": fuel_type_input
}])

# One-Hot Encoding for Categorical Features
categorical_columns = ["brand", "model", "color", "transmission_type", "fuel_type"]
input_dummies = pd.get_dummies(raw_input, columns=categorical_columns, drop_first=True)
input_df = input_dummies.reindex(columns=expected_columns, fill_value=0)

# Scale numerical features
numerical_columns = ["power_ps", "fuel_consumption_l_100km.1", "mileage_in_km", "vehicle_age"]
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# Sidebar: Read-Only Car Specifications Summary
with st.sidebar:
    st.markdown("## Car Specifications Summary")
    spec_summary = {
        "Brand": brand_input,
        "Model": model_input,
        "Transmission": transmission_input,
        "Fuel Type": fuel_type_input,
        "Color": color_input,
        "Power (PS)": power_ps,
        "Power (KW)": power_kw,
        "Mileage (km)": mileage,
        "Vehicle Age": vehicle_age,
    }
    for key, value in spec_summary.items():
        st.write(f"**{key}:** {value}")
        
# Prediction: Using a styled prediction card
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.markdown(f"<div class='prediction-card'>Predicted price : {prediction:,.00f} LE</div>", unsafe_allow_html=True)
    st.balloons()

# Footer
st.markdown("<div class='footer'>Supervised by Prof. Tarek , Cairo University © 2025</div>", unsafe_allow_html=True)
