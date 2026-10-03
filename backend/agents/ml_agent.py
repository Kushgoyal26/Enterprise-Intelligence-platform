"""
ML Agent — serves churn predictions using the trained model.

Loads the model trained by ml/train_churn_model.py, builds the same features
for currently ACTIVE customers, predicts churn probability, and uses SHAP to
explain the top reasons behind each customer's risk score in plain language.

Usage:
    python backend/agents/ml_agent.py
"""

import os
import json
import pandas as pd
import psycopg2
import joblib
import shap
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME", "enterprise_intel"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
)

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ml")
MODEL_PATH = os.path.join(ML_DIR, "churn_model.pkl")
FEATURES_PATH = os.path.join(ML_DIR, "feature_columns.json")

# Maps raw feature names to a human-readable phrase, used when explaining SHAP results.
FEATURE_LABELS = {
    "avg_login_recent": "recent login activity",
    "avg_usage_recent": "recent feature usage",
    "login_trend": "change in login activity over time",
    "usage_trend": "change in feature usage over time",
    "latest_mrr": "monthly revenue",
    "months_active": "account tenure (months)",
    "n_tickets": "total support tickets",
    "n_high_priority_tickets": "high/critical priority tickets",
    "ticket_rate": "support ticket frequency",
    "avg_satisfaction": "average support satisfaction score",
    "tenure_days": "account age (days)",
    "employee_count": "company size",
}

_model = None
_feature_cols = None


def load_model():
    global _model, _feature_cols
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found. Run 'python ml/train_churn_model.py' first."
            )
        _model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            _feature_cols = json.load(f)
    return _model, _feature_cols


def build_features_for_active_customers() -> pd.DataFrame:
    """Rebuild the same features used in training, but only for active customers."""
    conn = psycopg2.connect(**DB_CONFIG)
    customers = pd.read_sql("SELECT * FROM customers WHERE is_active = true", conn)
    subs = pd.read_sql("SELECT * FROM subscriptions", conn)
    tickets = pd.read_sql("SELECT * FROM support_tickets", conn)
    conn.close()

    subs = subs.sort_values(["customer_id", "billing_month"])
    tickets["created_date"] = pd.to_datetime(tickets["created_date"])

    rows = []
    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        cust_subs = subs[subs["customer_id"] == cid]
        cust_tickets = tickets[tickets["customer_id"] == cid]
        if len(cust_subs) == 0:
            continue

        recent = cust_subs.tail(3)
        earlier = cust_subs.head(max(len(cust_subs) - 3, 1))

        avg_login_recent = recent["login_count"].mean()
        avg_usage_recent = recent["feature_usage_score"].mean()
        login_trend = avg_login_recent - earlier["login_count"].mean()
        usage_trend = avg_usage_recent - earlier["feature_usage_score"].mean()

        avg_satisfaction = cust_tickets["satisfaction_score"].mean()
        if pd.isna(avg_satisfaction):
            avg_satisfaction = 3.0

        signup = pd.to_datetime(cust["signup_date"])
        tenure_days = (pd.Timestamp.now() - signup).days

        rows.append({
            "customer_id": cid,
            "company_name": cust["company_name"],
            "plan_tier": cust["plan_tier"],
            "region": cust["region"],
            "industry": cust["industry"],
            "employee_count": cust["employee_count"],
            "tenure_days": tenure_days,
            "avg_login_recent": avg_login_recent,
            "avg_usage_recent": avg_usage_recent,
            "login_trend": login_trend,
            "usage_trend": usage_trend,
            "latest_mrr": cust_subs.iloc[-1]["mrr_amount"],
            "months_active": len(cust_subs),
            "n_tickets": len(cust_tickets),
            "n_high_priority_tickets": len(cust_tickets[cust_tickets["priority"].isin(["High", "Critical"])]),
            "ticket_rate": len(cust_tickets) / max(len(cust_subs), 1),
            "avg_satisfaction": avg_satisfaction,
        })

    return pd.DataFrame(rows)


def explain_row(shap_row, feature_cols, top_n=3) -> list[str]:
    """Turn a SHAP row into plain-language reasons, strongest effect first."""
    contributions = list(zip(feature_cols, shap_row))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, value in contributions[:top_n]:
        if abs(value) < 0.01:
            continue
        label = FEATURE_LABELS.get(feat, feat)
        direction = "increases" if value > 0 else "decreases"
        reasons.append(f"{label} {direction} churn risk")
    return reasons


def get_churn_predictions(top_n: int = 10) -> list[dict]:
    """Return the top_n highest-risk active customers with churn probability and reasons."""
    model, feature_cols = load_model()

    raw_df = build_features_for_active_customers()
    if len(raw_df) == 0:
        return []

    info_cols = ["customer_id", "company_name"]
    categorical_cols = ["plan_tier", "region", "industry"]

    df_encoded = pd.get_dummies(raw_df, columns=categorical_cols)
    # align columns with what the model was trained on (missing cols = 0)
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    X = df_encoded[feature_cols]

    probabilities = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    results = []
    for i, (_, row) in enumerate(raw_df.iterrows()):
        reasons = explain_row(shap_values[i], feature_cols)
        results.append({
            "customer_id": int(row["customer_id"]),
            "company_name": row["company_name"],
            "churn_probability": round(float(probabilities[i]), 3),
            "reasons": reasons,
        })

    results.sort(key=lambda x: x["churn_probability"], reverse=True)
    return results[:top_n]


if __name__ == "__main__":
    predictions = get_churn_predictions(top_n=10)
    print(f"\nTop {len(predictions)} churn-risk customers:\n")
    for p in predictions:
        print(f"  {p['company_name']} — {p['churn_probability']*100:.1f}% risk")
        for r in p["reasons"]:
            print(f"      - {r}")
        print()
