import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('model/decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Prediksi Kualitas Red Wine 🍷")
st.write("Masukkan nilai fitur-fitur kimia untuk memprediksi kualitas anggur merah.")

# Input fitur dari pengguna
features = {
    "fixed acidity": st.number_input("Fixed Acidity", 4.0, 16.0, 7.4),
    "volatile acidity": st.number_input("Volatile Acidity", 0.10, 1.5, 0.7),
    "citric acid": st.number_input("Citric Acid", 0.0, 1.0, 0.0),
    "residual sugar": st.number_input("Residual Sugar", 0.9, 15.5, 1.9),
    "chlorides": st.number_input("Chlorides", 0.012, 0.2, 0.076),
    "free sulfur dioxide": st.number_input("Free Sulfur Dioxide", 1, 72, 11),
    "total sulfur dioxide": st.number_input("Total Sulfur Dioxide", 6, 300, 34),
    "density": st.number_input("Density", 0.9900, 1.0050, 0.9978),
    "pH": st.number_input("pH", 2.5, 4.0, 3.51),
    "sulphates": st.number_input("Sulphates", 0.3, 2.0, 0.56),
    "alcohol": st.number_input("Alcohol", 8.0, 15.0, 9.4)
}

# Prediksi
if st.button("Prediksi Kualitas"):
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    st.success(f"Prediksi kualitas wine: **{prediction}**")

    
