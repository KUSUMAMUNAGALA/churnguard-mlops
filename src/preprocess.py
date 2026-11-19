import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "ChurnLabel"

NUMERIC_FEATURES = [
    "Age", "TenureinMonths", "Number_of_Referrals",
    "AvgMonthlyLongDistanceCharges", "AvgMonthlyGBDownload",
    "MonthlyCharge", "TotalCharges", "TotalRefunds",
]

CATEGORICAL_FEATURES = [
    "Gender", "SeniorCitizen", "Married", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "InternetType", "OnlineSecurity",
    "OnlineBackup", "DeviceProtectionPlan", "PremiumTechSupport",
    "StreamingTV", "StreamingMovies", "UnlimitedData", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: str) -> pd.DataFrame:
    # utf-8-sig removes a hidden character at the start of the file.
    # keep_default_na=False keeps the word "None" as text (e.g. InternetType = None).
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    df.columns = df.columns.str.strip()

    # Keep only the columns we need. This automatically drops the leakage columns.
    df = df[FEATURES + [TARGET]].copy()

    # Make sure numeric columns are really numbers.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Turn "Yes"/"No" into 1/0.
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    # Numbers: fill missing values with the median, then scale.
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    # Text categories: fill missing with "Unknown", then one 0/1 column per option.
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])