"""
Runs every question in tests/golden_rag.json through the RAG agent and
checks whether the expected source document was among the retrieved sources.

This checks RETRIEVAL accuracy (did it find the right document), not whether
the generated answer's wording is perfect — read the printed answers yourself
to sanity-check quality.

Usage:
    python tests/run_rag_eval.py
"""

import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.rag_agent import answer_question

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_rag.json")
DELAY_SECONDS = 2


def main():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    passed = 0
    failed = []

    for i, item in enumerate(golden, 1):
        question = item["question"]
        expected = item["expected_source"]
        result = answer_question(question)

        found = expected in result["sources"]
        status = "PASS" if found else "FAIL"
        if found:
            passed += 1
        else:
            failed.append(item)

        print(f"[{i}/{len(golden)}] {status} — {question}")
        print(f"    Expected source: {expected}")
        print(f"    Retrieved sources: {result['sources']}")
        print(f"    Answer: {result['answer'][:200]}")
        print()

        if i < len(golden):
            time.sleep(DELAY_SECONDS)

    print("=" * 60)
    print(f"Passed: {passed}/{len(golden)} ({passed/len(golden)*100:.0f}%)")
    if failed:
        print("Failed questions:")
        for f_item in failed:
            print(f"  - {f_item['question']}")


if __name__ == "__main__":
    main()
