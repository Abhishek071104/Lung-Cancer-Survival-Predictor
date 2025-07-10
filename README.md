# 🩺 Lung Cancer Survival Prediction

This project aims to predict the **survival outcome of lung cancer patients** based on various clinical and demographic factors using machine learning. It uses a **Random Forest Classifier** trained on structured medical data and provides an intuitive **Streamlit web application** for real-time predictions.

---

## 📌 Project Objective

To build a predictive model that forecasts whether a patient diagnosed with lung cancer is likely to survive, using features like age, smoking status, treatment type, BMI, and comorbidities.

---

## 📊 Dataset Overview

The dataset includes information on:

- Demographics: Age, Gender, Country  
- Diagnosis & Health History: Cancer Stage, Smoking Status, Hypertension, Cirrhosis, Asthma  
- Lifestyle: BMI, Cholesterol, Family History, Other Cancers  
- Treatment: Type of treatment taken  
- Target: `survived` (Yes/No)

---

## 🧠 ML Approach

- **Preprocessing**: Label encoding, standardization, missing value handling  
- **Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)  
- **Model**: Random Forest Classifier  
- **Evaluation**: Achieved ~85% accuracy on test data  
- **Deployment**: Interactive Streamlit UI

---

├── app.py                  # Streamlit UI
├── model.pkl               # Trained Random Forest model
├── scaler.pkl              # Scaler used for input normalization
├── dataset_med.csv         # Dataset (optional for reference)
└── README.md
