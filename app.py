import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('vehical_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load average prices for car names
vehical_name_avg_price = pd.read_csv('vechical_name_avg_price.csv')

# Title for the web app
st.title("Vehical price prediction")
st.markdown("""
This app predicts the **Selling Price** of a vehical based on its features
""")

# Input fields
Vehical_Name= st.selectbox(
    "Vehical Name",
    options=vehical_name_avg_price["Vehical_Name"].unique()
)
Veh_Name_Avg_Price = vehical_name_avg_price.loc[
    vehical_name_avg_price['Vehical_Name'] == Vehical_Name,'Selling_Price'
].values[0]

Present_Price = st.number_input("Present Price (in Lakhs)", min_value=0.0, step=0.1, format="%.2f")
Kms_Driven = st.number_input("Kilometers Driven", min_value=500, step=1)
Year = st.number_input("Model Year", min_value=2000, max_value=2017, step=1)
Owners = st.number_input("Number of Previous Owners", min_value=0,max_value=3, step=1)

Fuel_Type = st.selectbox(
    "Fuel Type",
    options=["Petrol", "Diesel", "CNG"],
    index=0
)
Fuel_Type_Encoded = {"Petrol": 0, "Diesel": 1, "CNG": 2}[Fuel_Type]

Seller_Type = st.selectbox(
    "Seller Type",
    options=["Dealer", "Individual"]
)
Seller_Type_Encoded = 1 if Seller_Type == "Individual" else 0

Transmission = st.selectbox(
    "Transmission",
    options=["Manual", "Automatic"]
)
Transmission_Encoded = 1 if Transmission == "Manual" else 0

# Predict button
if st.button("Predict Selling Price"):
    try:
        # Prepare the input data
        features = np.array([[
            Year, Present_Price, Kms_Driven, Owners, Fuel_Type_Encoded,
            Seller_Type_Encoded, Transmission_Encoded,Veh_Name_Avg_Price
        ]])
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make a prediction
        predicted_price = model.predict(scaled_features)
        
        # Display the prediction
        st.success(f"The predicted selling price is ₹ {predicted_price[0]:.2f} Lakhs.")
    except ValueError as e:
        st.error(f"Error: {e}")
