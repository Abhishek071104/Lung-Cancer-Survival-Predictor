import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Page title
st.set_page_config(page_title="Lung Cancer Survival Predictor", layout='centered')
st.title("🩺 Lung Cancer Survival Prediction")
st.markdown("Enter patient details to predict survival outcome after lung cancer treatment.")

# Input fields
age = st.slider("Age", 10, 100, 50)
gender = st.selectbox("Gender", ['Male', 'Female'])
country = st.selectbox("Country", ['India', 'USA', 'UK', 'Other'])
cancer_stage = st.selectbox("Cancer Stage", ['Stage I', 'Stage II', 'Stage III', 'Stage IV'])
family_history = st.selectbox("Family History of Cancer", ['Yes', 'No'])
smoking_status = st.selectbox("Smoking Status", ['Never Smoked', 'Former Smoker', 'Current Smoker', 'Passive Smoker'])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
cholesterol = st.number_input("Cholesterol Level", min_value=50, max_value=500, value=200)
hypertension = st.selectbox("Hypertension", ['Yes', 'No'])
asthma = st.selectbox("Asthma", ['Yes', 'No'])
cirrhosis = st.selectbox("Cirrhosis", ['Yes', 'No'])
other_cancer = st.selectbox("Other Cancer History", ['Yes', 'No'])
treatment_type = st.selectbox("Treatment Type", ['Surgery', 'Chemotherapy', 'Radiation', 'Combined'])

# Encode categorical input
def encode_input(val, mapping):
    return mapping[val]

gender_map = {'Male': 1, 'Female': 0}
country_map = {'India': 0, 'USA': 1, 'UK': 2, 'Other': 3}
stage_map = {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}
bool_map = {'Yes': 1, 'No': 0}
smoke_map = {'Never Smoked': 0, 'Former Smoker': 1, 'Current Smoker': 2, 'Passive Smoker': 3}
treat_map = {'Surgery': 0, 'Chemotherapy': 1, 'Radiation': 2, 'Combined': 3}

# Final input vector
input_data = np.array([[
    age,
    encode_input(gender, gender_map),
    encode_input(country, country_map),
    encode_input(stage_map[cancer_stage], {0:0,1:1,2:2,3:3}),
    encode_input(family_history, bool_map),
    encode_input(smoking_status, smoke_map),
    bmi,
    cholesterol,
    encode_input(hypertension, bool_map),
    encode_input(asthma, bool_map),
    encode_input(cirrhosis, bool_map),
    encode_input(other_cancer, bool_map),
    encode_input(treatment_type, treat_map)
]])

# Preprocess
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(scaled_input)[0]
    confidence = model.predict_proba(scaled_input)[0][prediction] * 100

    if prediction == 1:
        st.success(f"✅ Patient is **likely to SURVIVE**. Confidence: {confidence:.2f}%")
    else:
        st.error(f"⚠️ Patient is **unlikely to survive**. Confidence: {confidence:.2f}%")
