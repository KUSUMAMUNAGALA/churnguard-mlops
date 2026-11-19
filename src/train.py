import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocess import FEATURES, TARGET, build_preprocessor, load_data

DATA_PATH = "data/telco.csv"
MODEL_PATH = "models/model.joblib"


def evaluate(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
    }


def main():
    df = load_data(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Saved now, used later for drift monitoring.
    X_train.to_csv("data/reference.csv", index=False)

    candidates = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=2.8, eval_metric="logloss", random_state=42,
        ),
    }

    mlflow.set_experiment("churnguard")
    best_auc, best_pipe = 0, None

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([
                ("prep", build_preprocessor()),
                ("model", model),
            ])
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)

            mlflow.log_param("model_type", name)
            mlflow.log_params({k: v for k, v in model.get_params().items() if v is not None})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

            print(name, {k: round(v, 3) for k, v in metrics.items()})
            if metrics["roc_auc"] > best_auc:
                best_auc, best_pipe = metrics["roc_auc"], pipe

    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Saved best model (ROC-AUC {best_auc:.3f}) to {MODEL_PATH}")


if __name__ == "__main__":
    main()