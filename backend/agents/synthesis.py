"""
Synthesis Agent.

Takes raw results from whichever agents the router dispatched (SQL, RAG,
ML — any combination) and writes one clear, business-friendly answer with
citations. Never invents facts beyond what the tool results contain.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

SYNTHESIS_SYSTEM_PROMPT = """You are a business analyst assistant. You are given raw results
from one or more internal tools (a SQL query result, retrieved document excerpts, and/or a
churn risk prediction) and must write ONE clear, direct answer to the user's question.

Rules:
- Base your answer ONLY on the data provided below. Never invent numbers or facts.
- Write in plain business language, not technical jargon.
- Be concise: a few sentences or a short list, not a long report.
- At the end, add a line starting with "Sources:" listing which tool(s) the answer came from
  (e.g. "Sources: SQL (subscriptions table), ML (churn model)").
- If some requested information wasn't available from any tool, say so plainly rather than
  guessing.
- If CHURN PREDICTIONS data is provided, you MUST briefly mention the top 1-2 reasons behind
  each customer's risk score (not just the probability) — this explainability is the whole
  point of showing churn predictions, so never report a risk percentage alone without at least
  one reason.
"""


def build_context(sql_result: dict = None, rag_result: dict = None, ml_result: list = None) -> str:
    parts = []

    if sql_result:
        if sql_result.get("success"):
            parts.append(
                f"[SQL RESULT]\nQuery: {sql_result['sql']}\n"
                f"Columns: {sql_result['columns']}\nRows: {sql_result['rows']}"
            )
        else:
            parts.append(f"[SQL RESULT]\nFailed to get an answer: {sql_result.get('error')}")

    if rag_result:
        parts.append(
            f"[DOCUMENT SEARCH RESULT]\nAnswer found: {rag_result['answer']}\n"
            f"Sources: {rag_result['sources']}"
        )

    if ml_result:
        lines = [f"  - {p['company_name']}: {p['churn_probability']*100:.1f}% risk "
                  f"({'; '.join(p['reasons'])})" for p in ml_result]
        parts.append("[CHURN PREDICTIONS]\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else "No data available."


def synthesize(question: str, sql_result: dict = None, rag_result: dict = None,
                ml_result: list = None) -> str:
    context = build_context(sql_result, rag_result, ml_result)
    user_msg = f"Question: {question}\n\nAvailable data:\n\n{context}"

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content.strip()
