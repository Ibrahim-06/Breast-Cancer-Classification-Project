import streamlit as st
import numpy as np
import joblib

model = joblib.load("../Model/Logistic_model.pkl")
scaler = joblib.load("../Scaler/Scaler.pkl")
feature_cols = joblib.load("../Features/features.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, rgba(255,182,193,0.4), rgba(32,178,170,0.4));
    backdrop-filter: blur(10px);
}

@keyframes floatText {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-8px); }
  100% { transform: translateY(0px); }
}
            
h1 {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5rem;
    animation: floatText 3s ease-in-out infinite;
}

.stButton > button {
    background-color: #20b2aa;
    color: white;
    font-size: 18px;
    padding: 0.6em 2em;
    border-radius: 15px;
    border: none;
    transition: 0.3s ease-in-out;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}
.stButton > button:hover {
    background-color: #ff6f61;
    transform: scale(1.07);
    box-shadow: 0px 6px 15px rgba(0,0,0,0.3);
}

.result-card {
    background: linear-gradient(135deg, rgba(255,182,193,0.4), rgba(32,178,170,0.4));
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔬 Breast Cancer Prediction App</h1>", unsafe_allow_html=True)
st.write("Enter the values for the features to predict if the tumor is **Benign** or **Malignant**.")

cols = st.columns(3)
user_data = []

for i, col in enumerate(feature_cols):
    with cols[i % 3]:
        val = st.number_input(f"{col}", value=0.0, format="%.4f")
        user_data.append(val)

user_array = np.array(user_data).reshape(1, -1)
user_array_scaled = scaler.transform(user_array)

if st.button("🔍 Predict"):
    prediction = model.predict(user_array_scaled)[0]
    if prediction == 1:
        st.markdown('<div class="result-card" style="color:#e74c3c;">⚠️ Prediction: Malignant</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card" style="color:#27ae60;">✅ Prediction: Benign</div>', unsafe_allow_html=True)
