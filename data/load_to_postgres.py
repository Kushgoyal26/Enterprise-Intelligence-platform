"""
Loads the generated CSVs into PostgreSQL using the schema in schema.sql.
Uses batch inserts (execute_values) so this runs quickly even against a
cloud database like Neon, where each round-trip has network latency.

Usage:
    1. Make sure PostgreSQL is running and .env points to it
    2. Generate data:    python data/generate_data.py
    3. Load into DB:     python data/load_to_postgres.py
"""

import csv
import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME", "enterprise_intel"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
    sslmode=os.getenv("DB_SSLMODE", "prefer"),
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def run_schema(conn):
    with open(SCHEMA_PATH) as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("Schema applied.")


def load_customers(conn):
    path = f"{RAW_DIR}/customers.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [(
            row["customer_id"], row["company_name"], row["industry"], row["region"],
            row["plan_tier"], row["signup_date"], row["employee_count"], row["account_owner"],
            row["is_active"] == "True", row["churned_date"] or None,
        ) for row in reader]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO customers
            (customer_id, company_name, industry, region, plan_tier, signup_date,
             employee_count, account_owner, is_active, churned_date)
            VALUES %s
        """, rows)
    conn.commit()
    print(f"customers loaded ({len(rows)} rows).")


def load_subscriptions(conn):
    path = f"{RAW_DIR}/subscriptions.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [(
            row["customer_id"], row["billing_month"], row["mrr_amount"],
            row["seats"], row["login_count"], row["feature_usage_score"],
        ) for row in reader]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO subscriptions
            (customer_id, billing_month, mrr_amount, seats, login_count, feature_usage_score)
            VALUES %s
        """, rows)
    conn.commit()
    print(f"subscriptions loaded ({len(rows)} rows).")


def load_tickets(conn):
    path = f"{RAW_DIR}/support_tickets.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [(
            row["ticket_id"], row["customer_id"], row["created_date"], row["priority"],
            row["category"], row["resolution_hours"], row["satisfaction_score"] or None,
        ) for row in reader]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO support_tickets
            (ticket_id, customer_id, created_date, priority, category,
             resolution_hours, satisfaction_score)
            VALUES %s
        """, rows)
    conn.commit()
    print(f"support_tickets loaded ({len(rows)} rows).")


def load_payments(conn):
    path = f"{RAW_DIR}/payments.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [(
            row["payment_id"], row["customer_id"], row["payment_date"], row["amount"],
            row["payment_method"], row["status"], row["is_anomaly"] == "True",
        ) for row in reader]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO payments
            (payment_id, customer_id, payment_date, amount, payment_method, status, is_anomaly)
            VALUES %s
        """, rows)
    conn.commit()
    print(f"payments loaded ({len(rows)} rows).")


if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        run_schema(conn)
        load_customers(conn)
        load_subscriptions(conn)
        load_tickets(conn)
        load_payments(conn)
        print("\nAll data loaded successfully.")
    finally:
        conn.close()
