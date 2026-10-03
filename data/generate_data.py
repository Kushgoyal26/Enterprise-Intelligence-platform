"""
Synthetic data generator for the Enterprise Intelligence Platform (SaaS domain).

Generates:
  - customers.csv        (500 companies)
  - subscriptions.csv    (18 months of MRR per customer)
  - support_tickets.csv  (tickets, more for churned customers)
  - payments.csv         (transactions with ~1% flagged anomalies)

Key design choice: churn is NOT random. Customers who churn show a real,
learnable pattern in the months before they churn:
  - login_count drops
  - feature_usage_score drops
  - support ticket volume/priority rises
  - satisfaction_score drops

This means a churn model trained on this data will actually learn something,
instead of learning noise.
"""

import csv
import random
from datetime import date, timedelta
from faker import Faker

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)

OUT_DIR = "/home/claude/enterprise-intel/data/raw"

N_CUSTOMERS = 500
START_MONTH = date(2025, 4, 1)   # 18 months of history
N_MONTHS = 18
TODAY = date(2026, 9, 1)

REGIONS = ["North", "South", "East", "West", "Central"]
INDUSTRIES = ["Retail", "Healthcare", "Manufacturing", "FinTech", "EdTech", "Logistics", "Media"]
PLANS = {"Starter": (5000, 15000), "Growth": (15000, 50000), "Enterprise": (50000, 200000)}
REPS = [fake.name() for _ in range(12)]


def month_range(start, n):
    months = []
    y, m = start.year, start.month
    for _ in range(n):
        months.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


ALL_MONTHS = month_range(START_MONTH, N_MONTHS)


def gen_customers():
    customers = []
    for cid in range(1, N_CUSTOMERS + 1):
        signup = fake.date_between(start_date=date(2023, 1, 1), end_date=date(2026, 6, 1))
        plan = random.choices(list(PLANS.keys()), weights=[0.5, 0.35, 0.15])[0]

        # ~18% of customers will churn — decide this upfront so we can bake in the pattern
        will_churn = random.random() < 0.18
        churned_date = None
        if will_churn:
            # churn happens sometime after signup, before today
            earliest_churn = max(signup + timedelta(days=90), START_MONTH)
            if earliest_churn < TODAY:
                days_span = (TODAY - earliest_churn).days
                if days_span > 0:
                    churned_date = earliest_churn + timedelta(days=random.randint(0, days_span))

        customers.append({
            "customer_id": cid,
            "company_name": fake.company(),
            "industry": random.choice(INDUSTRIES),
            "region": random.choice(REGIONS),
            "plan_tier": plan,
            "signup_date": signup.isoformat(),
            "employee_count": random.randint(10, 5000),
            "account_owner": random.choice(REPS),
            "is_active": churned_date is None,
            "churned_date": churned_date.isoformat() if churned_date else "",
            "_will_churn": will_churn,
            "_churned_date_obj": churned_date,
            "_plan": plan,
        })
    return customers


def gen_subscriptions(customers):
    rows = []
    for c in customers:
        base_mrr = random.randint(*PLANS[c["_plan"]])
        base_login = random.randint(15, 30)
        base_usage = random.uniform(60, 95)

        for month in ALL_MONTHS:
            signup = date.fromisoformat(c["signup_date"])
            if month < date(signup.year, signup.month, 1):
                continue  # not yet a customer

            churn_dt = c["_churned_date_obj"]
            if churn_dt and month > date(churn_dt.year, churn_dt.month, 1):
                continue  # already churned, no more billing

            # distance (in months) to churn, if applicable — drives the decay pattern
            months_to_churn = None
            if churn_dt:
                months_to_churn = (churn_dt.year - month.year) * 12 + (churn_dt.month - month.month)

            mrr = base_mrr
            login = base_login
            usage = base_usage

            if months_to_churn is not None and 0 <= months_to_churn <= 4:
                # decay zone: last 4 months before churn show declining engagement
                decay_factor = (5 - months_to_churn) / 5  # ramps 0.2 -> 1.0
                login = max(1, int(base_login * (1 - 0.7 * decay_factor)))
                usage = max(5, base_usage * (1 - 0.6 * decay_factor))
                mrr = base_mrr * (1 - 0.1 * decay_factor)  # slight MRR shrink too
            else:
                # normal healthy fluctuation
                login = max(1, int(base_login + random.randint(-5, 5)))
                usage = min(100, max(10, base_usage + random.uniform(-8, 8)))
                mrr = base_mrr * random.uniform(0.95, 1.05)

            rows.append({
                "customer_id": c["customer_id"],
                "billing_month": month.isoformat(),
                "mrr_amount": round(mrr, 2),
                "seats": random.randint(1, max(1, c["employee_count"] // 20)),
                "login_count": login,
                "feature_usage_score": round(usage, 2),
            })
    return rows


def gen_tickets(customers):
    rows = []
    tid = 1
    for c in customers:
        signup = date.fromisoformat(c["signup_date"])
        churn_dt = c["_churned_date_obj"]
        end = churn_dt if churn_dt else TODAY
        if end <= signup:
            continue

        span_days = (end - signup).days
        # churned customers generate more tickets, especially near the end
        base_ticket_count = random.randint(2, 8)
        extra_if_churn = random.randint(3, 10) if churn_dt else 0
        n_tickets = base_ticket_count + extra_if_churn

        for _ in range(n_tickets):
            created = signup + timedelta(days=random.randint(0, span_days))
            near_churn = churn_dt and (churn_dt - created).days <= 60
            priority = random.choices(
                ["Low", "Medium", "High", "Critical"],
                weights=[0.5, 0.3, 0.15, 0.05] if not near_churn else [0.15, 0.3, 0.35, 0.2]
            )[0]
            satisfaction = None
            if random.random() < 0.7:
                satisfaction = random.randint(1, 3) if near_churn else random.randint(3, 5)

            rows.append({
                "ticket_id": tid,
                "customer_id": c["customer_id"],
                "created_date": created.isoformat(),
                "priority": priority,
                "category": random.choice(["Billing", "Bug", "Feature Request", "Onboarding"]),
                "resolution_hours": round(random.uniform(1, 72), 1),
                "satisfaction_score": satisfaction if satisfaction else "",
            })
            tid += 1
    return rows


def gen_payments(customers, subscriptions):
    rows = []
    pid = 1
    for s in subscriptions:
        pay_date = date.fromisoformat(s["billing_month"]) + timedelta(days=random.randint(1, 5))
        is_anomaly = random.random() < 0.01  # 1% deliberate anomalies

        amount = s["mrr_amount"]
        status = "Success"
        method = random.choices(["Card", "Bank Transfer", "UPI"], weights=[0.5, 0.3, 0.2])[0]

        if is_anomaly:
            # anomaly types: huge spike, huge drop, or failed/refunded burst
            anomaly_type = random.choice(["spike", "drop", "failed"])
            if anomaly_type == "spike":
                amount = amount * random.uniform(5, 12)
            elif anomaly_type == "drop":
                amount = amount * random.uniform(0.01, 0.1)
            else:
                status = "Failed"
        elif random.random() < 0.03:
            status = "Refunded"

        rows.append({
            "payment_id": pid,
            "customer_id": s["customer_id"],
            "payment_date": pay_date.isoformat(),
            "amount": round(amount, 2),
            "payment_method": method,
            "status": status,
            "is_anomaly": is_anomaly,
        })
        pid += 1
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    customers = gen_customers()
    subscriptions = gen_subscriptions(customers)
    tickets = gen_tickets(customers)
    payments = gen_payments(customers, subscriptions)

    write_csv(f"{OUT_DIR}/customers.csv", customers,
              ["customer_id", "company_name", "industry", "region", "plan_tier",
               "signup_date", "employee_count", "account_owner", "is_active", "churned_date"])
    write_csv(f"{OUT_DIR}/subscriptions.csv", subscriptions,
              ["customer_id", "billing_month", "mrr_amount", "seats", "login_count", "feature_usage_score"])
    write_csv(f"{OUT_DIR}/support_tickets.csv", tickets,
              ["ticket_id", "customer_id", "created_date", "priority", "category",
               "resolution_hours", "satisfaction_score"])
    write_csv(f"{OUT_DIR}/payments.csv", payments,
              ["payment_id", "customer_id", "payment_date", "amount", "payment_method", "status", "is_anomaly"])

    n_churned = sum(1 for c in customers if not c["is_active"])
    n_anomalies = sum(1 for p in payments if p["is_anomaly"])

    print(f"Customers:        {len(customers)}  ({n_churned} churned, {n_churned/len(customers)*100:.1f}%)")
    print(f"Subscription rows: {len(subscriptions)}")
    print(f"Support tickets:   {len(tickets)}")
    print(f"Payments:          {len(payments)}  ({n_anomalies} flagged anomalies)")
    print(f"\nFiles written to: {OUT_DIR}")
