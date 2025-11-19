import streamlit as st
import pandas as pd
import numpy as np
import pickle


model = pickle.load(open("lgb_model.pkl", "rb"))  
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

try:
    with open("label_encoders.pkl", "rb") as f:
        le_dict = pickle.load(f)
except:
    le_dict = {}

st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗", layout="centered")

st.markdown("""
    <h1 style='text-align:center; color:#2E86C1;'>🚗 Vehicle Price Prediction</h1>
    <p style='text-align:center; color:#566573;'>Enter your vehicle details to estimate its price.</p>
""", unsafe_allow_html=True)

st.write("---")

st.subheader("Vehicle Specifications")

col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=50000, step=1000)
    engine_hp = st.number_input("Engine HP", min_value=50, max_value=1000, value=180, step=10)
    
with col2:
    car_age = st.number_input("Car Age (Years)", min_value=0, max_value=50, value=5, step=1)
    luxury = st.selectbox("Luxury Brand?", ["No", "Yes"])
    

st.subheader("Additional Details")

col3, col4 = st.columns(2)

with col3:

    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    mpg = st.number_input("MPG", min_value=10, max_value=80, value=30, step=1)
    
with col4:
    tax = st.number_input("Tax (£)", min_value=0, max_value=1000, value=150, step=10)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])


luxury_binary = 1 if luxury == "Yes" else 0
transmission_encoded = 1 if transmission == "Automatic" else 0  # Simple encoding


input_data = pd.DataFrame(columns=feature_names)


input_data.loc[0] = 0

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

st.write("---")


if st.button("Predict Price", type="primary"):
    try:

        numeric_features_to_scale = [col for col in input_data.columns 
                                   if col not in ['Is_Luxury'] and 
                                   input_data[col].dtype in ['int64', 'float64']]
        
        if numeric_features_to_scale:
            input_data_scaled = input_data.copy()
            input_data_scaled[numeric_features_to_scale] = scaler.transform(input_data_scaled[numeric_features_to_scale])
        else:
            input_data_scaled = input_data
        
        log_price_pred = model.predict(input_data_scaled)[0]
        
        price_pred = np.expm1(log_price_pred)
        
        st.markdown(f"""
            <div style="
                background:#EBF5FB; 
                padding:20px; 
                border-radius:10px; 
                text-align:center;
                border:2px solid #AED6F1;
                margin:20px 0;">
                <h2 style="color:#154360;">💰 Estimated Price:</h2>
                <h2 style='color:#D35400; font-size:32px;'>${price_pred:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 This is an estimated price based on the provided vehicle specifications.")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check that all required inputs are provided and try again.")

with st.expander("Debug Information"):
    st.write("**Features expected by model:**", feature_names)
    st.write("**Current input values:**")
    debug_df = pd.DataFrame([feature_mapping])
    st.dataframe(debug_df.T.rename(columns={0: 'Value'}))


st.write("---")
st.markdown("""
    <div style='background:#F4F6F6; padding:15px; border-radius:10px;'>
    <h4>💡 Usage Tips:</h4>
    <ul>
    <li>Mileage: Total distance vehicle has been driven</li>
    <li>Engine HP: Horsepower of the engine</li>
    <li>Car Age: Years since manufacture</li>
    <li>Luxury Brand: Brands like Mercedes, BMW, Audi</li>
    <li>For best results, provide accurate vehicle specifications</li>
    </ul>
    </div>
""", unsafe_allow_html=True)
