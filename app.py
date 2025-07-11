import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

st.set_page_config(page_title="Insurance Cost Estimator", layout="centered")

st.title("🩺 Medical Insurance Cost Estimator")

# Load model from MLflow registry
try:
    model = mlflow.sklearn.load_model("models:/Best_MedicalCostModel/Production")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# Input form
with st.form("User Input"):
    age = st.slider("Age", 18, 100)
    sex = st.selectbox("Gender", ["male", "female"])
    bmi = st.slider("BMI", 10.0, 50.0)
    children = st.selectbox("Children", list(range(6)))
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    submit = st.form_submit_button("Estimate Cost")

# Feature transformation
if submit:
    input_dict = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0
    }
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Medical Insurance Cost: ₹{prediction:,.2f}")