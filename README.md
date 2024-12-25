Name : Sumanth Kumar C
Company: CODETECH
Id:CT08DS223
Domain:Data Science
Duration:NOV 30th, 2024 - DEC 30th, 2024 

TASK 2:
  
# Vehicle Price Prediction

This repository contains the code to predict the price of old vehicles using machine learning techniques. The workflow includes data preprocessing, exploratory data analysis, feature engineering, model training, and deployment using Streamlit.

## Project Overview
The project involves analyzing and predicting vehicle prices based on various features such as fuel type, seller type, transmission type, and other relevant attributes. A linear regression model is used for prediction, and the model is deployed using Streamlit for user interaction.

## Requirements
- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - joblib
  - streamlit

## Dataset
The dataset used is named `Vehical_data.csv` and contains the following columns:
- Car_Name
- Fuel_Type
- Seller_Type
- Transmission
- Selling_Price
- Present_Price
- [Other relevant columns]

Ensure that the dataset is present in the root directory before running the script.

## Steps to Run the Project

### 1. Data Loading
The dataset is loaded using pandas, and basic information such as shape, columns, and null values is explored.

### 2. Data Preprocessing
- Missing values are handled.
- Categorical columns are encoded using mapping and one-hot encoding.
- Numerical columns are standardized using `StandardScaler`.

### 3. Exploratory Data Analysis (EDA)
- Visualize categorical data columns using bar plots.
- Analyze the correlation between numerical columns using a heatmap.
- Examine summary statistics for specific groups such as fuel type and seller type.

### 4. Feature Engineering
- Average selling prices for vehicle names are computed and added as a new feature.
- Irrelevant columns such as the original vehicle names are dropped.

### 5. Model Training
- Split the data into training and testing sets.
- Train a linear regression model on the standardized features.
- Evaluate the model using MAE, MSE, and R² score.

### 6. Deployment
- Save the trained model and scaler using `joblib`.
- Deploy the model using Streamlit for interactive prediction.

## File Structure
```
|-- Vehical_data.csv  # Dataset
|-- vehical_price_model.pkl  # Trained model
|-- scaler.pkl  # Scaler object
|-- app.py  # Streamlit application
|-- requirements.txt  # Python dependencies
|-- README.md  # Project documentation
```

## Sample Code
### Running the Streamlit Application
To deploy the model and interact with it using Streamlit, use the following command:
```bash
streamlit run app.py
```
### app.py
```python
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('vehical_price_model.pkl')
scaler = joblib.load('scaler.pkl')

def main():
    st.title("Vehicle Price Prediction")

    # Input fields for user data
    present_price = st.number_input("Present Price", min_value=0.0, step=0.1)
    km_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
    owner = st.selectbox("Owner", [0, 1, 2])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    # Map categorical data
    fuel_type_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_type_mapping = {"Dealer": 1, "Individual": 0}
    transmission_mapping = {"Manual": 1, "Automatic": 0}

    fuel_type = fuel_type_mapping[fuel_type]
    seller_type = seller_type_mapping[seller_type]
    transmission = transmission_mapping[transmission]

    # Create feature vector
    features = [[present_price, km_driven, owner, fuel_type, seller_type, transmission]]
    scaled_features = scaler.transform(features)

    # Predict price
    if st.button("Predict Price"):
        prediction = model.predict(scaled_features)
        st.success(f"Predicted Selling Price: {prediction[0]:.2f} lakhs")

if __name__ == "__main__":
    main()
```

## Results
The model achieves the following metrics on the test data:
- Mean Absolute Error (MAE): X.XX
- Mean Squared Error (MSE): X.XX
- R² Score: X.XX

## Contributions
Feel free to contribute by improving the model or enhancing the deployment interface.

## License
This project is licensed under the MIT License.
