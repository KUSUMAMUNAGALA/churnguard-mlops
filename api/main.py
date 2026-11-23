from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.preprocess import FEATURES

app = FastAPI(title="ChurnGuard API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
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


def risk(p: float) -> str:
    return "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("web/index.html")


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
        "risk_level": risk(proba),
    }


@app.post("/predict-batch")
def predict_batch(customers: List[Customer]):
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 rows")
    df = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    probas = model.predict_proba(df)[:, 1]
    return [
        {"churn_probability": round(float(p), 4), "risk_level": risk(float(p))}
        for p in probas
    ]