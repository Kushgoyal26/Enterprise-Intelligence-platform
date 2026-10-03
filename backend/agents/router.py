"""
Router Agent.

Classifies a natural-language question into which data source(s) are needed
to answer it: sql, rag, ml — or a combination.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

ROUTER_SYSTEM_PROMPT = """You are a router that decides which data source(s) are needed to
answer a business question about a SaaS company. Choose from:

- "sql": structured data — customer counts, MRR/revenue, support ticket stats, payments,
  regions, plan tiers, account owners. Anything involving counting, summing, averaging,
  grouping, or ranking numeric business data.
- "rag": unstructured documents — contract terms (payment terms, termination notice,
  SLA uptime, liability) and company policies (refunds, data retention, support SLAs,
  security, onboarding, billing).
- "ml": churn risk predictions for currently active customers, with reasons why.

Output ONLY a JSON array of the sources needed, nothing else. Examples:
["sql"]
["rag"]
["ml"]
["sql", "ml"]

Use more than one source only when the question genuinely needs both kinds of
information to be fully answered.
"""


def route_question(question: str) -> list[str]:
    """Return a list of sources needed, e.g. ["sql"], ["rag"], ["ml"], or combinations."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        temperature=0,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    # extract the first [...] array in case the model added extra text around it
    import re
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        sources = json.loads(raw)
        valid = [s for s in sources if s in ("sql", "rag", "ml")]
        return valid if valid else ["sql"]  # safe default
    except json.JSONDecodeError:
        print(f"    [Router debug] Could not parse model output: {raw[:200]}")
        return ["sql"]  # safe default if parsing fails


if __name__ == "__main__":
    test_questions = [
        "What is the total MRR this month?",
        "What is our refund policy for early cancellations?",
        "Which customers are most likely to churn?",
        "Give me the top churn-risk customers along with their current MRR.",
    ]
    for q in test_questions:
        sources = route_question(q)
        print(f"{sources}  <-  {q}")
