"""
Text-to-SQL Agent.

Takes a natural-language question, generates a validated read-only SQL query
using Claude, runs it against PostgreSQL, and returns the result.

Safety measures:
  - Only SELECT statements are allowed (checked via sqlglot).
  - Runs with a hard row limit and timeout.
  - If the query errors out, the agent retries with the error message fed
    back to the LLM (up to MAX_RETRIES times).
"""

import os
import json
import time
import psycopg2
import sqlglot
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 2
ROW_LIMIT = 200

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME", "enterprise_intel"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
)

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# --- Schema description given to the LLM. Keep this in sync with schema.sql ---
SCHEMA_DESCRIPTION = """
Table: customers
  customer_id     INT PRIMARY KEY
  company_name    VARCHAR
  industry        VARCHAR       -- Retail, Healthcare, Manufacturing, FinTech, EdTech, Logistics, Media
  region          VARCHAR       -- North, South, East, West, Central
  plan_tier       VARCHAR       -- Starter, Growth, Enterprise
  signup_date     DATE
  employee_count  INT
  account_owner   VARCHAR       -- sales rep name
  is_active       BOOLEAN       -- true = still a customer, false = churned
  churned_date    DATE          -- NULL if still active

Table: subscriptions  (one row per customer per billing month)
  subscription_id      INT PRIMARY KEY
  customer_id           INT REFERENCES customers
  billing_month         DATE    -- first day of the month
  mrr_amount             NUMERIC  -- monthly recurring revenue in INR
  seats                  INT
  login_count            INT     -- product usage signal
  feature_usage_score    NUMERIC -- 0-100 engagement score

Table: support_tickets
  ticket_id           INT PRIMARY KEY
  customer_id          INT REFERENCES customers
  created_date          DATE
  priority               VARCHAR  -- Low, Medium, High, Critical
  category                VARCHAR  -- Billing, Bug, Feature Request, Onboarding
  resolution_hours        NUMERIC
  satisfaction_score       INT     -- 1-5, NULL if unrated

Table: payments
  payment_id       INT PRIMARY KEY
  customer_id        INT REFERENCES customers
  payment_date         DATE
  amount                 NUMERIC
  payment_method          VARCHAR  -- Card, Bank Transfer, UPI
  status                   VARCHAR  -- Success, Failed, Refunded
  is_anomaly                BOOLEAN
"""

SYSTEM_PROMPT = f"""You are a PostgreSQL expert. Given a business question, write ONE valid
PostgreSQL SELECT query that answers it, using only the schema below.

{SCHEMA_DESCRIPTION}

Rules:
- Output ONLY the raw SQL query. No explanation, no markdown fences, no comments.
- Only SELECT statements. Never write/modify data.
- Always add "LIMIT {ROW_LIMIT}" unless the query already aggregates to a small result.
- Use explicit JOINs with ON clauses, never implicit joins.
- Round monetary/numeric results to 2 decimal places with ROUND().
"""


def generate_sql(question: str, error_feedback: str = None) -> str:
    """Ask the LLM (via Groq) to generate a SQL query for the question."""
    user_msg = question
    if error_feedback:
        user_msg = (
            f"{question}\n\nYour previous query failed with this error:\n{error_feedback}\n"
            f"Please fix the query."
        )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    sql = response.choices[0].message.content.strip()
    # strip markdown fences if the model added them anyway
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def validate_sql(sql: str) -> tuple[bool, str]:
    """Return (is_valid, error_message). Only allows single SELECT statements."""
    try:
        parsed = sqlglot.parse(sql, read="postgres")
    except Exception as e:
        return False, f"SQL parse error: {e}"

    if len(parsed) != 1:
        return False, "Only a single SQL statement is allowed."

    stmt = parsed[0]
    if stmt.key.lower() != "select":
        return False, f"Only SELECT statements are allowed, got: {stmt.key}"

    # block dangerous keywords as a second safety net
    banned = ["insert", "update", "delete", "drop", "alter", "truncate", "grant", "create"]
    lowered = sql.lower()
    for word in banned:
        if word in lowered:
            return False, f"Query contains a disallowed keyword: {word}"

    return True, ""


def run_query(sql: str) -> dict:
    """Execute the SQL against Postgres and return columns + rows."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 5000")  # 5 second timeout
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return {"columns": columns, "rows": rows}
    finally:
        conn.close()


def text_to_sql(question: str) -> dict:
    """
    Full pipeline: question -> SQL -> validate -> run -> retry on failure.
    Returns a dict with the question, final SQL, result (or error), and attempt count.
    """
    error_feedback = None
    last_sql = None

    for attempt in range(1, MAX_RETRIES + 2):  # initial try + retries
        sql = generate_sql(question, error_feedback)
        last_sql = sql

        is_valid, val_error = validate_sql(sql)
        if not is_valid:
            error_feedback = val_error
            continue

        try:
            result = run_query(sql)
            return {
                "question": question,
                "sql": sql,
                "columns": result["columns"],
                "rows": result["rows"],
                "attempts": attempt,
                "success": True,
            }
        except Exception as e:
            error_feedback = str(e)
            continue

    return {
        "question": question,
        "sql": last_sql,
        "error": error_feedback,
        "attempts": MAX_RETRIES + 1,
        "success": False,
    }


if __name__ == "__main__":
    # quick manual test
    q = "Which region has the highest total MRR this year?"
    result = text_to_sql(q)
    print(json.dumps(result, indent=2, default=str))
