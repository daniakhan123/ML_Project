import streamlit as st
import pandas as pd
import numpy as np
import pickle


with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Vehicle Price Prediction")
st.write("Enter your vehicle details to estimate its price.")


mileage = st.number_input("Mileage", 0, 500000, 50000, step=1000)
engine_hp = st.number_input("Engine HP", 50, 1000, 180, step=10)
car_age = st.number_input("Car Age (Years)", 0, 50, 5, step=1)
luxury = st.selectbox("Luxury Brand?", ["No", "Yes"])
engine_size = st.number_input("Engine Size (L)", 0.5, 8.0, 2.0, step=0.1)
mpg = st.number_input("MPG", 10, 80, 30, step=1)
tax = st.number_input("Tax (£)", 0, 1000, 150, step=10)
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# ------------------------------
# Prepare input data
# ------------------------------
luxury_binary = 1 if luxury == "Yes" else 0
transmission_encoded = 1 if transmission == "Automatic" else 0

input_data = pd.DataFrame(columns=feature_names)
input_data.loc[0] = 0  # initialize all features with zero

# Map user inputs to model features
feature_mapping = {
    'mileage': mileage,
    'engine_hp': engine_hp,
    'Car_Age': car_age,
    'Is_Luxury': luxury_binary,
    'engineSize': engine_size,
    'mpg': mpg,
    'tax': tax,
    'transmission': transmission_encoded
}

for feature, value in feature_mapping.items():
    if feature in input_data.columns:
        input_data.loc[0, feature] = value


# Prediction button
if st.button("Predict Price"):
    # Scale numeric features (except binary)
    numeric_features = [col for col in input_data.columns if col not in ['Is_Luxury']]
    if numeric_features:
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])

    # Predict
    log_price_pred = model.predict(input_data)[0]
    price_pred = np.expm1(log_price_pred)

    # Convert to PKR
    exchange_rate = 280  # Adjust if needed
    price_pred_pkr = price_pred * exchange_rate

    # Display large price box
    st.markdown(f"""
        <div style="
            background-color: #D6EAF8; 
            padding: 30px; 
            border-radius: 15px; 
            text-align: center;
            border: 2px solid #3498DB;
            margin: 20px 0;">
            <h1 style="color:#154360; font-size:48px;">💰 ₨ {price_pred_pkr:,.0f}</h1>
            <p style="color:#1B4F72; font-size:20px;">Estimated Vehicle Price</p>
        </div>
    """, unsafe_allow_html=True)
