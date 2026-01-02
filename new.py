import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Heart Stroke Prediction",
    page_icon="🫀",
    layout="wide"
)

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.markdown("""
<style>

html, body, [data-testid="stApp"] {
    height: 100%;
    background: radial-gradient(circle at 20% 20%, rgba(255,0,120,0.35), transparent 40%),
                radial-gradient(circle at 80% 30%, rgba(0,255,200,0.35), transparent 40%),
                radial-gradient(circle at 50% 80%, rgba(120,120,255,0.35), transparent 40%),
                linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1d2671);
    background-size: 200% 200%;
    animation: deepMove 18s ease infinite;
}

@keyframes deepMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.glass {
    background: rgba(255,255,255,0.16);
    backdrop-filter: blur(22px);
    border-radius: 22px;
    padding: 30px;
    box-shadow: 0 30px 80px rgba(0,0,0,0.45);
}

.title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    color: #ff4b5c;
    text-shadow: 0 0 35px rgba(255,75,92,0.9);
}

.subtitle {
    text-align: center;
    color: #ffffff;
    font-size: 18px;
    margin-bottom: 30px;
}

.heart-img {
    display: flex;
    justify-content: center;
    animation: pulse 1.4s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.07); }
    100% { transform: scale(1); }
}

.stButton>button {
    width: 100%;
    padding: 16px;
    font-size: 20px;
    border-radius: 14px;
    background: linear-gradient(135deg, #ff512f, #dd2476);
    color: white;
    border: none;
    box-shadow: 0 18px 45px rgba(255,0,100,0.6);
}

.stButton>button:hover {
    transform: scale(1.08);
}

.result-high {
    background: linear-gradient(135deg, #ff0844, #ffb199);
    padding: 25px;
    border-radius: 18px;
    color: white;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
}

.result-low {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    padding: 25px;
    border-radius: 18px;
    color: white;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🫀 Heart Stroke Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Medical Risk Analysis | Developed by Saksham Jain</div>", unsafe_allow_html=True)

left, center, right = st.columns([1.4, 1, 1.4])

with center:
    st.markdown("<div class='heart-img'>", unsafe_allow_html=True)
    st.image("heart.png", width=260)
    st.markdown("</div>", unsafe_allow_html=True)
    st.audio("heartbeat.mp3")

with left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    predict = st.button("🔍 Predict Risk")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    if predict:
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]
        scaled = scaler.transform(df)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1] * 100

        st.metric("🧠 Risk Probability", f"{probability:.2f} %")
        st.progress(int(probability))

        if prediction == 1:
            st.markdown("<div class='result-high'>⚠️ High Risk Detected</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-low'>✅ Low Risk Detected</div>", unsafe_allow_html=True)
