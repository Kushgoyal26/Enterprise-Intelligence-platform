"""
Runs every question in tests/golden_sql.json through the Text-to-SQL agent
and reports how many succeeded (ran without error) vs failed.

This does NOT check if the answer is semantically correct — it checks that
the agent produced a valid, executable query. Read through the printed
results yourself and sanity-check a few answers against what you'd expect.

Usage:
    python tests/run_sql_eval.py
"""

import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.sql_agent import text_to_sql

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_sql.json")
DELAY_SECONDS = 3  # Groq free tier allows 30 requests/minute, so a small delay is enough


def main():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    passed = 0
    failed = []

    for i, item in enumerate(golden, 1):
        question = item["question"]
        result = text_to_sql(question)

        status = "PASS" if result["success"] else "FAIL"
        if result["success"]:
            passed += 1
            preview = result["rows"][:2]
            print(f"[{i}/{len(golden)}] {status} — {question}")
            print(f"    SQL: {result['sql']}")
            print(f"    Result preview: {preview}")
        else:
            failed.append(item)
            print(f"[{i}/{len(golden)}] {status} — {question}")
            print(f"    Error: {result.get('error')}")
        print()

        if i < len(golden):
            time.sleep(DELAY_SECONDS)

    print("=" * 60)
    print(f"Passed: {passed}/{len(golden)} ({passed/len(golden)*100:.0f}%)")
    if failed:
        print(f"Failed questions:")
        for f_item in failed:
            print(f"  - {f_item['question']}")


if __name__ == "__main__":
    main()
