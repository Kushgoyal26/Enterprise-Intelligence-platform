"""
Trains an XGBoost churn prediction model on the SaaS customer data.

Feature engineering summary per customer:
  - Recent engagement: average login count & feature usage over their most
    recent 3 months of subscription data, plus the trend (recent vs earlier).
  - Support signal: total tickets, count of High/Critical priority tickets,
    average satisfaction score.
  - Account attributes: plan tier, region, industry, employee count, tenure.

Label: is_active == False  ->  1 (churned), else 0.

Usage:
    python ml/train_churn_model.py

Outputs (saved to ml/):
  - churn_model.pkl       — trained XGBoost model (joblib)
  - feature_columns.json  — ordered list of feature columns the model expects
  - shap_summary.png      — SHAP feature importance plot
  - metrics.json          — AUC, precision, recall on the held-out test set
"""

import os
import json
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
import shap
import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME", "enterprise_intel"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
)

ML_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    customers = pd.read_sql("SELECT * FROM customers", conn)
    subscriptions = pd.read_sql("SELECT * FROM subscriptions", conn)
    tickets = pd.read_sql("SELECT * FROM support_tickets", conn)
    conn.close()
    return customers, subscriptions, tickets


def build_features(customers: pd.DataFrame, subs: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    rows = []

    subs = subs.sort_values(["customer_id", "billing_month"])
    tickets["created_date"] = pd.to_datetime(tickets["created_date"])

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        cust_subs = subs[subs["customer_id"] == cid]
        cust_tickets = tickets[tickets["customer_id"] == cid]

        if len(cust_subs) == 0:
            continue  # no subscription history, skip

        # recent engagement: last up to 3 months on record for this customer
        recent = cust_subs.tail(3)
        earlier = cust_subs.head(max(len(cust_subs) - 3, 1))

        avg_login_recent = recent["login_count"].mean()
        avg_usage_recent = recent["feature_usage_score"].mean()
        avg_login_earlier = earlier["login_count"].mean()
        avg_usage_earlier = earlier["feature_usage_score"].mean()

        login_trend = avg_login_recent - avg_login_earlier
        usage_trend = avg_usage_recent - avg_usage_earlier

        latest_mrr = cust_subs.iloc[-1]["mrr_amount"]
        months_active = len(cust_subs)

        n_tickets = len(cust_tickets)
        n_high_priority = len(cust_tickets[cust_tickets["priority"].isin(["High", "Critical"])])
        avg_satisfaction = cust_tickets["satisfaction_score"].mean()
        if pd.isna(avg_satisfaction):
            avg_satisfaction = 3.0  # neutral default if no rated tickets

        signup = pd.to_datetime(cust["signup_date"])
        end_date = pd.to_datetime(cust["churned_date"]) if pd.notna(cust["churned_date"]) else pd.Timestamp.now()
        tenure_days = (end_date - signup).days

        rows.append({
            "customer_id": cid,
            "plan_tier": cust["plan_tier"],
            "region": cust["region"],
            "industry": cust["industry"],
            "employee_count": cust["employee_count"],
            "tenure_days": tenure_days,
            "avg_login_recent": avg_login_recent,
            "avg_usage_recent": avg_usage_recent,
            "login_trend": login_trend,
            "usage_trend": usage_trend,
            "latest_mrr": latest_mrr,
            "months_active": months_active,
            "n_tickets": n_tickets,
            "n_high_priority_tickets": n_high_priority,
            "ticket_rate": n_tickets / max(months_active, 1),
            "avg_satisfaction": avg_satisfaction,
            "churned": 0 if cust["is_active"] else 1,
        })

    return pd.DataFrame(rows)


def main():
    print("Loading data from PostgreSQL...")
    customers, subs, tickets = load_data()

    print("Building features...")
    df = build_features(customers, subs, tickets)
    print(f"Built features for {len(df)} customers ({df['churned'].sum()} churned)")

    # one-hot encode categoricals
    categorical_cols = ["plan_tier", "region", "industry"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    feature_cols = [c for c in df_encoded.columns if c not in ("customer_id", "churned")]
    X = df_encoded[feature_cols]
    y = df_encoded["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\nTest set performance:")
    print(f"  AUC:       {auc:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])}")

    # SHAP explainability
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(ML_DIR, "shap_summary.png"), dpi=120)
    plt.close()
    print(f"SHAP summary plot saved to ml/shap_summary.png")

    # save artifacts
    joblib.dump(model, os.path.join(ML_DIR, "churn_model.pkl"))
    with open(os.path.join(ML_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(ML_DIR, "metrics.json"), "w") as f:
        json.dump({"auc": auc, "precision": precision, "recall": recall,
                    "n_customers": len(df), "n_churned": int(df["churned"].sum())}, f, indent=2)

    print(f"\nModel saved to ml/churn_model.pkl")
    print("Done.")


if __name__ == "__main__":
    main()
