import streamlit as st
import pandas as pd
import joblib

# Load scaler and model
scaler = joblib.load("scalerhouse.pkl")
model = joblib.load("modelhouse.pkl")

# Custom CSS for Styling
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #ff1a1a;
        }
        .prediction-box {
            background-color: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section with Gradient Background
st.markdown(
    """
    <h1 style="text-align:center; color:white; padding:10px; background:linear-gradient(90deg, #ff7e5f, #feb47b); border-radius:10px;">
        🏡 House Price Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("🔹 This app predicts the estimated price of a house based on various factors.")

# Layout using Columns
col1, col2 = st.columns(2)

with col1:
    square_footage = st.number_input("🏠 Square Footage:", min_value=100, step=10)
    num_bedrooms = st.number_input("🛏️ Number of Bedrooms:", min_value=0, step=1)
    num_bathrooms = st.number_input("🛁 Number of Bathrooms:", min_value=0, step=1)
    garage_size = st.number_input("🚗 Garage Size (Cars):", min_value=0, step=1)

with col2:
    year_built = st.number_input("📅 Year Built:", min_value=1800, max_value=2025, step=1)
    lot_size = st.number_input("🌳 Lot Size (Acres):", min_value=0.0, step=0.1)
    neighborhood_quality = st.number_input("🏡 Neighborhood Rating (1-10):", min_value=1, max_value=10, step=1)

# Predict Button
if st.button("🔍 Predict Price"):
    # Prepare Data
    data = pd.DataFrame({
        "Square_Footage": [square_footage],
        "Num_Bedrooms": [num_bedrooms],
        "Num_Bathrooms": [num_bathrooms],
        "Year_Built": [year_built],
        "Lot_Size": [lot_size],
        "Garage_Size": [garage_size],
        "Neighborhood_Quality": [neighborhood_quality],
    })

    # Scale Data
    data_scaled = scaler.transform(data)

    # Predict Price
    pred = model.predict(data_scaled)

    # Display Result in a Beautiful Box
    st.markdown(
        f"""
        <div class="prediction-box">
            💰 Predicted House Price: <br> <strong>${round(pred[0], 2):,.2f}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
