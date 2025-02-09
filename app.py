import streamlit as st
import pickle
import numpy as np
import gdown
import pandas as pd
import os

# ------------------------- Custom CSS for Full-Page Styling -------------------------
st.markdown(
    """
    <style>
        /* Full-page background */
        body {
            background: linear-gradient(to right, #4facfe, #00f2fe);
            font-family: 'Arial', sans-serif;
            color: white;
        }
        
        /* Centered App Box */
        .stApp {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            max-width: 800px;
            margin: auto;
        }

        /* Title & Header */
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ff4b4b;
        }

        /* Buttons */
        .stButton>button {
            background-color: #ff4b4b !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 8px !important;
            transition: 0.3s ease-in-out;
        }
        
        .stButton>button:hover {
            background-color: #ff2e2e !important;
            transform: scale(1.05);
        }

        /* Input Fields */
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            border-radius: 10px !important;
            padding: 8px !important;
            border: 1px solid #ccc !important;
        }
        
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------- Download Dataset from Google Drive -------------------------
file_id = "1Z6--9jyYoMhEbOXNI5HNDhM0JtgJfQj9"
dataset_path = "airline_data.csv"

if not os.path.exists(dataset_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dataset_path, quiet=False)

# Load dataset
df = pd.read_csv(dataset_path)

# Load trained model
model_path = "fare_prediction_model.pkl"
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found! Please upload 'fare_prediction_model.pkl'.")
    st.stop()

# ------------------------- App Layout -------------------------
st.markdown('<h1 class="title">✈️ Airline Fare Prediction</h1>', unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter flight details below to predict the fare.</h4>", unsafe_allow_html=True)

# Sidebar Section
st.sidebar.title("📊 About the App")
st.sidebar.info(
    """
    - This app predicts airline fares based on flight details.
    - Enter flight parameters and click **Predict Fare**.
    - The dataset is automatically loaded from Google Drive.
    """
)

# ------------------------- Organizing Inputs into Columns -------------------------
col1, col2 = st.columns(2)

with col1:
    lf_ms = st.number_input("✈️ Load Factor (lf_ms)", value=0.00)
    large_ms = st.number_input("📦 Large Market Share (large_ms)", value=0.00)
    fare_lg = st.number_input("💰 Fare Large Market (fare_lg)", value=0.00)
    quarter = st.selectbox("📅 Quarter", [1, 2, 3, 4], index=0)

with col2:
    fare_low = st.number_input("💵 Low Fare (fare_low)", value=0.00)
    year = st.slider("📆 Year", 1993, 2024, 1993)
    feature_7 = st.number_input("🔹 Feature 7", value=0.00)
    feature_8 = st.number_input("🔹 Feature 8", value=0.00)

# Extra Features Section
feature_9 = st.number_input("🔹 Feature 9", value=0.00)
feature_10 = st.number_input("🔹 Feature 10", value=0.00)
feature_11 = st.number_input("🔹 Feature 11", value=0.00)
feature_12 = st.number_input("🔹 Feature 12", value=0.00)

# Collect features into an array
features = np.array([[lf_ms, large_ms, fare_lg, quarter, fare_low, year,
                      feature_7, feature_8, feature_9, feature_10, feature_11, feature_12]])

# ------------------------- Predict Button -------------------------
st.markdown("<br>", unsafe_allow_html=True)  # Add spacing

if st.button("🎯 Predict Fare", key="predict"):
    try:
        prediction = model.predict(features)
        st.success(f"💸 Predicted Fare: **${prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")



