import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('fare_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("✈️ Airline Fare Prediction App")
st.write("Enter flight details to predict the fare")

# Input fields
lf_ms = st.number_input("Enter lf_ms:", value=0.00)
large_ms = st.number_input("Enter large_ms:", value=0.00)
fare_lg = st.number_input("Enter fare_lg:", value=0.00)
quarter = st.number_input("Enter quarter:", value=1, min_value=1, max_value=4)
fare_low = st.number_input("Enter fare_low:", value=0.00)
year = st.number_input("Enter Year:", value=1993, min_value=1993, max_value=2024)
feature_7 = st.number_input("Enter feature_7:", value=0.00)  # Replace with actual feature name
feature_8 = st.number_input("Enter feature_8:", value=0.00)
feature_9 = st.number_input("Enter feature_9:", value=0.00)
feature_10 = st.number_input("Enter feature_10:", value=0.00)
feature_11 = st.number_input("Enter feature_11:", value=0.00)
feature_12 = st.number_input("Enter feature_12:", value=0.00)

# Collect features into an array
features = np.array([[lf_ms, large_ms, fare_lg, quarter, fare_low, year,
                      feature_7, feature_8, feature_9, feature_10, feature_11, feature_12]])

# Debugging: Print number of features
st.write(f"Number of input features: {features.shape[1]}")
st.write("Feature values:", features)

# Predict
if st.button("Predict Fare"):
    prediction = model.predict(features)
    st.write(f"Predicted Fare: ${prediction[0]:.2f}")
