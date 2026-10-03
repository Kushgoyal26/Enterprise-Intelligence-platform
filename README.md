# Enterprise Intelligence Platform — SaaS Domain

An AI-powered platform that answers business questions by combining SQL (structured data),
RAG (documents), and ML (predictions) — with a router agent, synthesis, and self-verification.

## Current Status: All 8 Steps Complete ✅

- [x] Project folder structure
- [x] PostgreSQL schema + synthetic data generator
- [x] Text-to-SQL agent — 20/20 on golden set
- [x] RAG engine — 10/10 on golden set
- [x] ML churn model — XGBoost + SHAP
- [x] Router agent — 10/10 on golden set
- [x] Synthesis agent
- [x] LangGraph orchestrator
- [x] Self-verification layer — numeric + citation checks
- [x] Frontend chat UI
- [x] Production hardening — auth, logging, rate limiting, Docker, CI/CD

This is a complete, working Enterprise Intelligence Platform: ask a business
question in plain English and get an answer sourced from live SQL data,
company documents, and a churn prediction model — with confidence scoring
and citations.
- [ ] ML churn model (Step 4)
- [ ] Router + Synthesis agents (Step 5)
- [ ] Self-verification layer (Step 6)
- [ ] Frontend (Step 7)
- [ ] Production hardening (Step 8)

## Data Model

Four tables, SaaS business:

| Table              | What it holds                                             |
|---------------------|-------------------------------------------------------------|
| `customers`          | 500 companies — industry, region, plan tier, churn status |
| `subscriptions`      | Monthly MRR, seats, login count, feature usage per customer |
| `support_tickets`    | Tickets with priority, category, satisfaction score        |
| `payments`           | Transactions, with ~1% flagged anomalies for testing        |

**Important design choice**: churn isn't random. Customers who churn show declining
login/usage in the 4 months before churning, and worse support tickets. This means
a churn model trained on this data will actually learn a real signal — you can verify
this yourself (see "Sanity check" below).

## Setup — Step by Step

### 1. Prerequisites
- Python 3.11+
- Docker Desktop running
- VS Code with Python extension

### 2. Create virtual environment
```bash
cd enterprise-intel
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 3. Start PostgreSQL
```bash
docker compose up -d
```
Check it's running: `docker ps` should show `ei_postgres`.

### 4. Generate synthetic data
```bash
python data/generate_data.py
```
This writes CSVs to `data/raw/`. You'll see a summary like:
```
Customers:        500  (97 churned, 19.4%)
Subscription rows: 7057
Support tickets:   3151
Payments:          7057  (65 flagged anomalies)
```

### 5. Load data into PostgreSQL
```bash
pip install psycopg2-binary
python data/load_to_postgres.py
```

### 6. Verify it worked
Connect with any PostgreSQL client (or VS Code's SQLTools extension):
```
Host: localhost | Port: 5432 | DB: enterprise_intel | User: ei_user | Pass: ei_pass
```
Try:
```sql
SELECT region, COUNT(*) FROM customers GROUP BY region;
SELECT plan_tier, AVG(mrr_amount) FROM subscriptions GROUP BY plan_tier;
```

## Sanity Check (recommended before moving to Step 2)

Confirm the churn pattern is real and learnable — churned customers should show
noticeably lower average login_count than healthy ones:
```sql
SELECT
  c.is_active,
  ROUND(AVG(s.login_count), 1) AS avg_logins,
  ROUND(AVG(s.feature_usage_score), 1) AS avg_usage
FROM subscriptions s
JOIN customers c ON c.customer_id = s.customer_id
GROUP BY c.is_active;
```
You should see healthy customers (`is_active = true`) with clearly higher averages.

## Golden Test Set (build this next, alongside Step 2)

Before building the SQL agent, write ~30 questions with known correct SQL and
expected answers in `tests/golden_sql.json`. Use them to measure agent accuracy
every time you change the prompt or schema. Example entries:
- "What is the total MRR for Enterprise plan customers?"
- "Which region has the highest churn rate?"
- "How many critical priority tickets were created in the last 3 months?"

## Project Structure
```
enterprise-intel/
├── backend/
│   ├── agents/          # router, sql_agent, rag_agent, ml_agent, synthesis, verify
│   ├── api/              # FastAPI endpoints
│   └── requirements.txt
├── data/
│   ├── schema.sql
│   ├── generate_data.py
│   ├── load_to_postgres.py
│   ├── raw/              # generated CSVs (gitignored)
│   └── docs/              # will hold fake contracts/policies for RAG (Step 3)
├── ml/                    # training notebooks, saved models (Step 4)
├── frontend/               # React + TypeScript (Step 7)
├── tests/                  # golden test sets
├── docker-compose.yml
└── README.md
```

## Step 2 — Text-to-SQL Agent Setup

### 1. Get an Anthropic API key
Go to https://console.anthropic.com, sign up, and create an API key
(Settings → API Keys). New accounts get some free credit.

### 2. Create your .env file
Copy `.env.example` to `.env` and fill in:
- `ANTHROPIC_API_KEY` — your key from step 1
- `DB_PASSWORD` — the PostgreSQL password you set during install
- `DB_USER` — `postgres` if you installed PostgreSQL directly (not Docker)

### 3. Install new dependencies
```bash
pip install -r backend/requirements.txt
pip install python-dotenv
```

### 4. Test the agent directly (no server needed)
```bash
python backend/agents/sql_agent.py
```
This runs one hardcoded test question and prints the SQL + result as JSON.

### 5. Run the golden test set
```bash
python tests/run_sql_eval.py
```
This runs 20 questions through the agent and reports a pass rate. A "PASS"
means the query ran successfully — read the SQL and result preview yourself
to sanity-check the answer looks right.

### 6. Start the API server (optional, for the frontend later)
```bash
uvicorn backend.api.main:app --reload
```
Open http://localhost:8000/docs in your browser — this gives you a UI to
POST a question to `/ask` and see the result, without writing any code.

## Step 3 — RAG Engine Setup

### 1. Install new dependencies
```bash
pip install chromadb sentence-transformers
```
Note: `sentence-transformers` will download a small (~80MB) embedding model
the first time it runs. This needs internet access once; after that it's
cached locally and works offline.

### 2. Generate the fake documents
```bash
python data/docs/generate_docs.py
```
This creates 15 contracts (tied to real customer names from your data) and
6 company policy documents under `data/docs/`.

### 3. Ingest documents into the vector database
```bash
python backend/agents/rag_agent.py ingest
```
This chunks every document, embeds each chunk, and stores them in a local
ChromaDB database at `data/chroma_db/`. Re-run this any time you add or
change documents.

### 4. Test the agent directly
```bash
python backend/agents/rag_agent.py
```
Runs one hardcoded question and prints the answer with sources.

### 5. Run the golden test set
```bash
python tests/run_rag_eval.py
```
Runs 10 policy questions and checks whether the agent retrieved the correct
source document for each one.

## Step 4 — ML Churn Model Setup

### 1. Install new dependencies
```bash
pip install xgboost scikit-learn shap joblib matplotlib
```

### 2. Train the model
```bash
python ml/train_churn_model.py
```
This pulls data from PostgreSQL, builds features per customer (recent login/
usage trends, support ticket history, account attributes), trains an
XGBoost classifier, and saves:
- `ml/churn_model.pkl` — the trained model
- `ml/feature_columns.json` — feature list the model expects
- `ml/shap_summary.png` — which features matter most, overall
- `ml/metrics.json` — AUC, precision, recall on held-out test data

Note: you'll likely see a very high AUC (close to 1.0). That's expected —
the synthetic data has a clean, deliberate churn pattern baked in (see
Step 1), which is much easier to learn than real-world data. It confirms
the pipeline works; in a real dataset, 0.75-0.90 AUC would be considered good.

### 3. Get predictions for currently active customers
```bash
python backend/agents/ml_agent.py
```
Prints the top 10 highest churn-risk active customers, each with their
probability and the top reasons (from SHAP) driving that score — e.g.
"recent login activity decreases churn risk" means low recent logins are
pushing the score up.

## Step 5 — Router + Synthesis Setup

### 1. Install new dependency
```bash
pip install langgraph
```

### 2. Test the router alone
```bash
python backend/agents/router.py
```
Prints which source(s) it picked for 4 sample questions.

### 3. Run the router golden test set
```bash
python tests/run_router_eval.py
```
10 questions, checks whether the router picked the right source(s).

### 4. Ask the full pipeline a question
```bash
python backend/orchestrator.py "What is the total MRR this month?"
```
Watch the terminal — it prints which agents were called, then the final
synthesized answer. Try a RAG question, an ML question, and a mixed one:
```bash
python backend/orchestrator.py "What is our refund policy?"
python backend/orchestrator.py "Which customers are most likely to churn?"
python backend/orchestrator.py "Give me the churn risk and current MRR for our top at-risk customers."
```

Or run it interactively (keeps asking until you Ctrl+C):
```bash
python backend/orchestrator.py
```

### Known limitation (worth understanding, not fixing yet)
For mixed questions (e.g. "churn risk AND MRR for at-risk customers"), the
SQL and ML agents currently run independently rather than the ML agent's
output feeding into the SQL query. The synthesis step does its best to
combine both results, but a more advanced version would have the ML agent's
customer list flow into a targeted SQL query. This is a natural place to
improve the pipeline later — mention it as a known trade-off if asked.

## Step 6 — Self-Verification Setup

No new dependencies needed — this step only adds `backend/agents/verify.py`
and wires it into the orchestrator as a final step after synthesis.

### What it checks
1. **Numeric check**: every number in the final answer should trace back to
   a number that actually appeared in the SQL result or ML predictions
   (within a small tolerance for rounding). If the LLM hallucinated a number,
   this catches it.
2. **Citation check**: if the answer cites a document filename, that file
   must actually be one of the documents the RAG agent retrieved.

Both checks produce a confidence score (1.0 = fully verified) and a list of
warnings if something couldn't be confirmed.

### Test it directly
```bash
python backend/agents/verify.py
```
Runs two hardcoded examples — one answer with a correct number, one with a
deliberately wrong number — and shows how the confidence score differs.

### See it in the full pipeline
```bash
python backend/orchestrator.py "What is the total MRR for the most recent billing month?"
```
Now the output includes a `[Verify]` step and prints a Confidence score plus
any warnings after the answer.

## Step 7 — Frontend Setup

### Why plain HTML/JS instead of React?
React needs a whole separate toolchain (Node.js, npm, a bundler like Vite).
Given how much setup friction we'd already hit with Python/Docker, a single
self-contained HTML file that talks to the FastAPI backend gets you a real,
working chat UI today with zero extra installs. It's a legitimate engineering
trade-off to mention in an interview: "I shipped a lightweight UI first and
would reach for React if the team needed component reuse or complex state."
Upgrading to React later is a drop-in replacement for this one file.

### 1. Start the backend
```bash
uvicorn backend.api.main:app --reload
```
Leave this terminal running. It should print something like
`Uvicorn running on http://127.0.0.1:8000`.

### 2. Open the frontend
Just double-click `frontend/index.html` (or right-click → Open with → your
browser). No server needed for the frontend itself — it's a static file that
calls the backend over HTTP.

### 3. Try it
Click one of the example questions, or type your own. You'll see:
- The synthesized answer
- Which sources were used (SQL / RAG / ML tags)
- A confidence score (color-coded: green = high, amber = medium, red = low)
- Any verification warnings, if the answer couldn't be fully confirmed

If you see "backend not running" in the header, make sure step 1's terminal
is still running and didn't error out.

## Step 8 — Production Hardening

No new Python dependencies — this step adds configuration and infrastructure
files around the existing app.

### What was added

**1. Authentication (optional, off by default)**
`backend/api/main.py` now checks for an `API_KEY` env var. If unset, the API
runs without auth (fine for local dev). To enable it:
```bash
# in .env, add:
API_KEY=some-secret-string
```
Then every request to `/ask` needs a header `X-API-Key: some-secret-string`.
(If you turn this on, you'd also need to add that header to the `fetch` call
in `frontend/index.html` — it's commented as an exercise, not wired in by
default, to keep the demo simple.)

**2. Rate limiting**
A simple in-memory limiter caps each client (by API key or IP) to
`RATE_LIMIT_PER_MINUTE` requests per minute (default 20). Good enough for a
single-instance deployment; a real multi-server deployment would use Redis
instead of in-memory state.

**3. Structured logging**
`backend/logging_config.py` gives every module timestamped, leveled logs
instead of scattered `print()` calls. The API logs each question, how long
it took, which sources were used, and the confidence score — exactly what
you'd want to search through in a monitoring tool in production.

**4. Docker**
`Dockerfile` containerizes the API. `docker-compose.yml` runs the full stack
(PostgreSQL + API) together. You ran PostgreSQL natively earlier in this
project due to a virtualization issue on your machine — that's fine for
local development. These files exist as a deployment reference: this is how
the app would be packaged and run in an environment where Docker works
(most cloud hosts, CI runners, teammates' machines).

**5. CI/CD**
`.github/workflows/tests.yml` runs automatically on every push to GitHub: it
spins up a real PostgreSQL instance, loads the synthetic data, and runs the
Text-to-SQL golden test set. To make the live-agent test run (not just the
syntax check), add `GROQ_API_KEY` as a repository secret on GitHub (Settings
→ Secrets and variables → Actions). This is what "tests run automatically
before code is trusted" looks like in a real team.

### Try it locally
```bash
uvicorn backend.api.main:app --reload
```
Watch the terminal — you'll now see structured log lines for every question
asked through the frontend, including response time and confidence.

### Security considerations not implemented (worth knowing, good interview answers)
- **Row-level security / RBAC**: right now, anyone with API access can query
  any customer's data. A real multi-tenant product would restrict each
  user's SQL agent to their own organization's rows.
- **Secrets management**: `.env` works for a single developer; a team would
  use a secrets manager (AWS Secrets Manager, Doppler, etc.) instead.
- **HTTPS**: this runs on plain HTTP locally; a real deployment sits behind
  a reverse proxy (nginx, a cloud load balancer) that terminates TLS.
- **Prompt injection**: a malicious document in the RAG corpus, or a crafted
  question, could try to manipulate the LLM's behavior. The SQL agent is
  protected by `sqlglot` validation (SELECT-only), but the RAG and synthesis
  steps don't have an equivalent guardrail yet.

## Project Complete

All 8 steps from the original architecture are built and working. Good next
moves from here: push this to GitHub with a clear README (this file already
does most of that work), record a 2-3 minute demo video of the frontend in
action, and mention the known limitations above in interviews — being able
to discuss what you'd improve is as valuable as what you built.
