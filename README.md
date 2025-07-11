# 🩺 Lung Cancer Survival Prediction

This project predicts whether a patient diagnosed with lung cancer is likely to survive, based on clinical and demographic details. The model is trained using Random Forest and deployed as an interactive Streamlit app.

---

## 📌 Objective

To demonstrate a full ML pipeline: from data preprocessing and model training to web deployment — optimized for lightweight, GitHub-friendly models.

---

## 🚀 Technologies Used

- Python 3
- Scikit-learn (RandomForestClassifier)
- Imbalanced-learn (SMOTE)
- Streamlit (UI)
- Joblib (model serialization)

---

## 📊 Features Used for Prediction

- Age  
- Gender  
- Country  
- Cancer Stage  
- Family History of Cancer  
- Smoking Status  
- BMI  
- Cholesterol Level  
- Hypertension  
- Asthma  
- Cirrhosis  
- Other Cancer History  
- Treatment Type

🗑️ **Dropped**: `id`, `diagnosis_date`, `end_treatment_date`

---

## 🧠 ML Model Details

| Property            | Value                           |
|---------------------|----------------------------------|
| Model               | RandomForestClassifier           |
| n_estimators        | 25–40 (finalized for <25MB)      |
| Data Used           | ~10–12% (stratified sample)      |
| Balancing           | SMOTE                            |
| Accuracy Achieved   | ~74%                             |
| Compression         | Joblib `compress=3 or 4`         |
| Model Size          | ✅ ~20–25 MB                     |
| GitHub Friendly     | ✅ Yes (no LFS required)          |

---

## 🖥️ Streamlit App Features

- Tabbed interface: **Predict** and **Prediction History**
- Real-time survival prediction with confidence %
- Option to download history as CSV
- Mobile-responsive layout
- Clean, professional sidebar with author info

---

## 📦 Files Included

| File             | Purpose                              |
|------------------|---------------------------------------|
| `app.py`         | Streamlit frontend                    |
| `model.pkl`      | Trained and compressed ML model       |
| `scaler.pkl`     | Feature scaler                        |
| `requirements.txt` | All Python dependencies             |
| `README.md`      | Project overview and instructions     |

---

