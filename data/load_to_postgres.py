"""
Loads the generated CSVs into PostgreSQL using the schema in schema.sql.

Usage:
    1. Start Postgres:   docker compose up -d
    2. Generate data:    python data/generate_data.py
    3. Load into DB:     python data/load_to_postgres.py
"""

import csv
import os
import psycopg2

DB_CONFIG = dict(
    host="localhost",
    port=5432,
    dbname="enterprise_intel",
    user="ei_user",
    password="ei_pass",
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
    with open(path) as f, conn.cursor() as cur:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute("""
                INSERT INTO customers
                (customer_id, company_name, industry, region, plan_tier, signup_date,
                 employee_count, account_owner, is_active, churned_date)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                row["customer_id"], row["company_name"], row["industry"], row["region"],
                row["plan_tier"], row["signup_date"], row["employee_count"], row["account_owner"],
                row["is_active"] == "True", row["churned_date"] or None,
            ))
    conn.commit()
    print("customers loaded.")


def load_subscriptions(conn):
    path = f"{RAW_DIR}/subscriptions.csv"
    with open(path) as f, conn.cursor() as cur:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute("""
                INSERT INTO subscriptions
                (customer_id, billing_month, mrr_amount, seats, login_count, feature_usage_score)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (
                row["customer_id"], row["billing_month"], row["mrr_amount"],
                row["seats"], row["login_count"], row["feature_usage_score"],
            ))
    conn.commit()
    print("subscriptions loaded.")


def load_tickets(conn):
    path = f"{RAW_DIR}/support_tickets.csv"
    with open(path) as f, conn.cursor() as cur:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute("""
                INSERT INTO support_tickets
                (ticket_id, customer_id, created_date, priority, category,
                 resolution_hours, satisfaction_score)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                row["ticket_id"], row["customer_id"], row["created_date"], row["priority"],
                row["category"], row["resolution_hours"], row["satisfaction_score"] or None,
            ))
    conn.commit()
    print("support_tickets loaded.")


def load_payments(conn):
    path = f"{RAW_DIR}/payments.csv"
    with open(path) as f, conn.cursor() as cur:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute("""
                INSERT INTO payments
                (payment_id, customer_id, payment_date, amount, payment_method, status, is_anomaly)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                row["payment_id"], row["customer_id"], row["payment_date"], row["amount"],
                row["payment_method"], row["status"], row["is_anomaly"] == "True",
            ))
    conn.commit()
    print("payments loaded.")


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
