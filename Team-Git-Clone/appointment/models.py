# accounts/models.py

from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    disease = models.CharField(max_length=100, blank=True, null=True)  # Field for disease

    def __str__(self):
        return self.user.username
    
# appointment/vital_analysis_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATASET_FILE = 'human_vital_signs_dataset_2024.csv'

def load_dataset():
    df = pd.read_csv(DATASET_FILE)
    df.columns = df.columns.str.strip()
    return df

def train_ml_model():
    df = load_dataset()
    X = df[['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation', 'Systolic Blood Pressure', 'Diastolic Blood Pressure']]
    y = df['Risk Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict_risk_with_ml(model, heart_rate, respiratory_rate, body_temperature, oxygen_saturation, systolic_blood_pressure, diastolic_blood_pressure):
    input_data = [[heart_rate, respiratory_rate, body_temperature, oxygen_saturation, systolic_blood_pressure, diastolic_blood_pressure]]
    prediction = model.predict(input_data)
    return prediction[0]

def generate_issues_report(risk_category):
    issues = {
        "Low Risk": "No immediate concerns detected. Maintain a healthy lifestyle.",
        "High Risk": "Possible health issues detected. Consult a healthcare provider."
    }
    with open('patient_issues_report.txt', 'w') as file:
        file.write(f"Predicted Risk Category: {risk_category}\n")
        file.write(issues[risk_category] + "\n")

