import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.preprocess import FEATURES

app = FastAPI(title="ChurnGuard API", version="1.0")
model = joblib.load("models/model.joblib")


class Customer(BaseModel):
    Age: int
    TenureinMonths: int
    Number_of_Referrals: int
    AvgMonthlyLongDistanceCharges: float
    AvgMonthlyGBDownload: float
    MonthlyCharge: float
    TotalCharges: float
    TotalRefunds: float
    Gender: str
    SeniorCitizen: str
    Married: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    InternetType: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtectionPlan: str
    PremiumTechSupport: str
    StreamingTV: str
    StreamingMovies: str
    UnlimitedData: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])[FEATURES]
    proba = float(model.predict_proba(df)[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "will_churn": proba >= 0.5,
        "risk_level": "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low",
    }