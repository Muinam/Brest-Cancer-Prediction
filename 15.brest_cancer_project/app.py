import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load(r"E:\Langchain_Model\15.brest_cancer_project\best_breast_cancer_model.pkl")


st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Breast Cancer Prediction App")
st.markdown("Enter patient feature values below and click **Predict** to see the result.")

# Define 30 feature names (Wisconsin Breast Cancer Dataset)
feature_names = ['mean_radius', 'mean_texture', 'mean_smoothness', 'mean_compactness', 'mean_symmetry']

# Create inputs (auto-arranged in 3 columns)
cols = st.columns(3)
user_input = []

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(f"{feature}", min_value=0.0, format="%.5f")
        user_input.append(val)

# Predict button
if st.button("🔍 Predict"):
    input_array = np.array([user_input])
    try:
        prediction = model.predict(input_array)
        result = "🩸 Malignant (Cancerous)" if prediction[0] == "M" else "✅ Benign (Non-Cancerous)"
        st.success(result)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Optional footer
st.markdown("---")
st.caption("Developed with ❤️ in Streamlit for interactive ML prediction.")
