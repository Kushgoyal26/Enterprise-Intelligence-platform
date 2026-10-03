"""
Runs every question in tests/golden_router.json through the router and
checks whether it picked the expected source(s).

Usage:
    python tests/run_router_eval.py
"""

import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.router import route_question

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_router.json")
DELAY_SECONDS = 2


def main():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    passed = 0
    failed = []

    for i, item in enumerate(golden, 1):
        question = item["question"]
        expected = set(item["expected_sources"])
        actual = set(route_question(question))

        match = actual == expected
        status = "PASS" if match else "FAIL"
        if match:
            passed += 1
        else:
            failed.append({**item, "actual": list(actual)})

        print(f"[{i}/{len(golden)}] {status} — {question}")
        print(f"    Expected: {sorted(expected)}   Got: {sorted(actual)}")
        print()

        if i < len(golden):
            time.sleep(DELAY_SECONDS)

    print("=" * 60)
    print(f"Passed: {passed}/{len(golden)} ({passed/len(golden)*100:.0f}%)")
    if failed:
        print("Failed questions:")
        for f_item in failed:
            print(f"  - {f_item['question']}  (expected {f_item['expected_sources']}, got {f_item['actual']})")


if __name__ == "__main__":
    main()
