import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.preprocess import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, load_data

ALPHA = 0.05  # p-value below this means "drifted"


def check_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        # Kolmogorov-Smirnov test: do both samples come from the same distribution?
        stat, p = ks_2samp(reference[col].dropna(), current[col].dropna())
        rows.append((col, "numeric", "KS test", round(stat, 3), p))
    for col in CATEGORICAL_FEATURES:
        # Chi-square test: do category frequencies differ?
        counts = pd.concat(
            [reference[col].value_counts(), current[col].value_counts()], axis=1
        ).fillna(0)
        stat, p, _, _ = chi2_contingency(counts.values)
        rows.append((col, "categorical", "Chi-square", round(stat, 3), p))

    out = pd.DataFrame(rows, columns=["feature", "type", "test", "statistic", "p_value"])
    out["drifted"] = out["p_value"] < ALPHA
    out["p_value"] = out["p_value"].map(lambda v: f"{v:.4g}")
    return out.sort_values("drifted", ascending=False)


def save_html(result: pd.DataFrame, path: str):
    n_drift = int(result["drifted"].sum())
    rows = ""
    for _, r in result.iterrows():
        color = "#fde2e2" if r["drifted"] else "#e6f4ea"
        rows += (
            f"<tr style='background:{color}'><td>{r['feature']}</td><td>{r['type']}</td>"
            f"<td>{r['test']}</td><td>{r['statistic']}</td><td>{r['p_value']}</td>"
            f"<td>{'DRIFT' if r['drifted'] else 'ok'}</td></tr>"
        )
    html = f"""<html><head><meta charset='utf-8'><title>ChurnGuard Drift Report</title>
<style>body{{font-family:Arial;margin:30px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}th{{background:#222;color:#fff}}</style>
</head><body><h1>ChurnGuard Data Drift Report</h1>
<p><b>{n_drift}</b> of {len(result)} features drifted (p &lt; {ALPHA}).</p>
<table><tr><th>Feature</th><th>Type</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Status</th></tr>
{rows}</table></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    reference = pd.read_csv("data/reference.csv", keep_default_na=False)
    for col in NUMERIC_FEATURES:
        reference[col] = pd.to_numeric(reference[col], errors="coerce")

    # Simulate "production" data: sample customers, then shift some values
    current = load_data("data/telco.csv")[FEATURES].sample(1000, random_state=1)
    current["MonthlyCharge"] = current["MonthlyCharge"] * 1.3  # price increase
    current["TenureinMonths"] = (current["TenureinMonths"] * 0.5).astype(int)  # newer customers

    result = check_drift(reference, current)
    print(result[["feature", "test", "p_value", "drifted"]].head(8).to_string(index=False))
    save_html(result, "reports/drift_report.html")
    print("\nSaved reports/drift_report.html")


if __name__ == "__main__":
    main()