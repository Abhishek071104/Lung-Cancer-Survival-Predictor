# -*- coding: utf-8 -*-
"""Lung-Cancer-Survival-Predictor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1caL0YyDjK1Tcv2dFk6my0k7CfKhJfb1U
"""

!pip install -q pandas scikit-learn imbalanced-learn joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib, os, warnings
warnings.filterwarnings('ignore')

# Load original dataset
df = pd.read_csv("dataset_med.csv")

# Drop unnecessary columns
df = df.drop(columns=['id', 'diagnosis_date', 'end_treatment_date'])

# Drop missing targets
df = df.dropna(subset=['survived'])

# Remove duplicates
df = df.drop_duplicates()

# Fill missing numeric columns with median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Stratified sample (e.g., 12% of data per class)
df_small = df.groupby('survived', group_keys=False).apply(lambda x: x.sample(frac=0.15, random_state=42)).reset_index(drop=True)

print("Reduced dataset shape:", df_small.shape)

le = LabelEncoder()
for col in df_small.select_dtypes(include='object').columns:
    df_small[col] = le.fit_transform(df_small[col])

X = df_small.drop('survived', axis=1)
y = df_small['survived']

# Optional: SMOTE if imbalance still exists
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Final split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=24, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

joblib.dump(model, 'model.pkl', compress=3)
joblib.dump(scaler, 'scaler.pkl')

print("📦 Model file size (MB):", os.path.getsize('model.pkl') / (1024 * 1024))