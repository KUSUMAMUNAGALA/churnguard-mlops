from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

HIGH_RISK = {
    "Age": 30, "TenureinMonths": 2, "Number_of_Referrals": 0,
    "AvgMonthlyLongDistanceCharges": 10.0, "AvgMonthlyGBDownload": 20,
    "MonthlyCharge": 85.5, "TotalCharges": 171.0, "TotalRefunds": 0.0,
    "Gender": "Female", "SeniorCitizen": "No", "Married": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Yes",
    "InternetType": "Fiber Optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtectionPlan": "No", "PremiumTechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "UnlimitedData": "No",
    "Contract": "Month-to-Month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Bank Withdrawal",
}

LOW_RISK = {
    **HIGH_RISK,
    "TenureinMonths": 60, "TotalCharges": 4000.0, "Contract": "Two Year",
    "PremiumTechSupport": "Yes", "OnlineSecurity": "Yes",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_valid_response():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}


def test_high_risk_scores_higher_than_low_risk():
    high = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK).json()["churn_probability"]
    assert high > low


def test_predict_rejects_bad_input():
    bad = {**HIGH_RISK, "TenureinMonths": "not a number"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422