import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Load dataset to get available options
df = pd.read_csv("ford.csv")

# Page configuration
st.set_page_config(
    page_title="Ford Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Title
st.title("🚗 Ford Car Price Prediction")
st.write("Enter the car details below to predict its price.")

# Input fields

model_name = st.selectbox(
    "Car Model",
    sorted(df["model"].unique())
)

year = st.number_input(
    "Year",
    min_value=int(df["year"].min()),
    max_value=int(df["year"].max()),
    value=int(df["year"].max())
)

transmission = st.selectbox(
    "Transmission",
    sorted(df["transmission"].unique())
)

mileage = st.number_input(
    "Mileage",
    min_value=float(df["mileage"].min()),
    max_value=float(df["mileage"].max()),
    value=float(df["mileage"].median())
)

fuel_type = st.selectbox(
    "Fuel Type",
    sorted(df["fuelType"].unique())
)

tax = st.number_input(
    "Tax",
    min_value=float(df["tax"].min()),
    max_value=float(df["tax"].max()),
    value=float(df["tax"].median())
)

mpg = st.number_input(
    "MPG",
    min_value=float(df["mpg"].min()),
    max_value=float(df["mpg"].max()),
    value=float(df["mpg"].median())
)

engine_size = st.number_input(
    "Engine Size",
    min_value=float(df["engineSize"].min()),
    max_value=float(df["engineSize"].max()),
    value=float(df["engineSize"].median())
)

# Prediction button
if st.button("Predict Price"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(
        f"💰 Estimated Car Price: £{prediction:,.2f}"
    )