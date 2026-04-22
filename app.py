import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests

# Page configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transition: 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_anim = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qh5z2fdq.json")

# Load the model
@st.cache_resource
def load_model():
    with open("model (5).pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Header Section
st.title("📊 Customer Behavior Analysis")
col_a, col_b = st.columns([2, 1])
with col_a:
    st.write("Provide customer demographics and account details to predict the outcome.")
with col_b:
    st_lottie(lottie_anim, height=150, key="analytics")

# Input Interface
st.subheader("Customer Profile Details")
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 35)

    with col2:
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col3:
        has_card = st.radio("Has Credit Card?", ["Yes", "No"])
        is_active = st.radio("Is Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 75000.0)

# Preprocessing Inputs (Mapping categorical to numeric)
geo_map = {"France": 0, "Germany": 1, "Spain": 2}
gender_map = {"Male": 1, "Female": 0}
binary_map = {"Yes": 1, "No": 0}

input_data = np.array([[
    credit_score, geo_map[geography], gender_map[gender], age, 
    tenure, balance, num_products, binary_map[has_card], 
    binary_map[is_active], salary
]])

# Prediction Button
if st.button("Generate Prediction"):
    prediction = model.predict(input_data)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.error("### ⚠️ Result: The customer is likely to churn.")
    else:
        st.balloons()
        st.success("### ✅ Result: The customer is likely to stay!")
