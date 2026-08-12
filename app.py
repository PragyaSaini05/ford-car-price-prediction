import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Load dataset
df = pd.read_csv("ford.csv")

# Page configuration
st.set_page_config(
    page_title="Ford Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 25px;
    }

    .result-price {
        font-size: 32px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🚗 Ford Car Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enter the vehicle specifications to estimate its market price.</div>',
    unsafe_allow_html=True
)

st.divider()

# Input section
st.subheader("🚘 Vehicle Details")

model_name = st.selectbox(
    "Car Model",
    sorted(df["model"].unique())
)

year = st.number_input(
    "Manufacturing Year",
    min_value=int(df["year"].min()),
    max_value=2026,
    value=2020,
    step=1
)

transmission = st.selectbox(
    "Transmission",
    sorted(df["transmission"].unique())
)

mileage = st.number_input(
    "Mileage",
    min_value=float(df["mileage"].min()),
    max_value=float(df["mileage"].max()),
    value=float(df["mileage"].median()),
    step=100.0
)

fuel_type = st.selectbox(
    "Fuel Type",
    sorted(df["fuelType"].unique())
)

tax = st.number_input(
    "Tax",
    min_value=float(df["tax"].min()),
    max_value=float(df["tax"].max()),
    value=float(df["tax"].median()),
    step=1.0
)

mpg = st.number_input(
    "MPG",
    min_value=float(df["mpg"].min()),
    max_value=float(df["mpg"].max()),
    value=float(df["mpg"].median()),
    step=0.1
)

engine_size = st.number_input(
    "Engine Size",
    min_value=float(df["engineSize"].min()),
    max_value=float(df["engineSize"].max()),
    value=float(df["engineSize"].median()),
    step=0.1
)

st.divider()

# Prediction
if st.button("🔮 Predict Car Price", use_container_width=True):

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

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div class="result-box">
            <div>Estimated Car Price</div>
            <div class="result-price">£{prediction:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.divider()

st.caption(
    "Built with Python, Scikit-learn and Streamlit | "
    "Ford Car Price Prediction"
)