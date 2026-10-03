"""
Self-Verification Layer.

Checks the synthesized answer against the raw tool results before it reaches
the user:
  1. Numeric check — every number mentioned in the answer should trace back
     to a number that actually appeared in the SQL result or ML predictions.
  2. Citation check — if the answer cites a document source, that source
     must actually be one of the documents the RAG agent retrieved.

This does NOT re-verify the underlying SQL/ML correctness (that's what the
golden test sets are for) — it only catches cases where the synthesis step
hallucinated or misquoted a number/source that wasn't actually in the data.
"""

import re


def _clean_lines_of_list_markers(text: str) -> str:
    """Strip leading '1. ', '2. ' etc. from lines so list numbering isn't
    mistaken for a data value during the numeric check."""
    lines = text.split("\n")
    cleaned = [re.sub(r"^\s*\d+\.\s+", "", line) for line in lines]
    return "\n".join(cleaned)


def _extract_numbers(text: str) -> list[float]:
    """Extract candidate numeric values from text (handles commas, decimals, %)."""
    text = _clean_lines_of_list_markers(text)
    raw_matches = re.findall(r"\d[\d,]*\.?\d*", text)
    numbers = []
    for m in raw_matches:
        cleaned = m.replace(",", "")
        try:
            val = float(cleaned)
            if val != 0:
                numbers.append(val)
        except ValueError:
            continue
    return numbers


def _flatten_sql_values(sql_result: dict) -> list[float]:
    values = []
    if not sql_result or not sql_result.get("success"):
        return values
    for row in sql_result.get("rows", []):
        for cell in row:
            try:
                values.append(float(cell))
            except (ValueError, TypeError):
                continue
    return values


def _flatten_rag_values(rag_result: dict) -> list[float]:
    """Numbers that actually appear in the retrieved document text — these are
    valid ground truth for a RAG answer, same role SQL/ML results play elsewhere."""
    if not rag_result or not rag_result.get("raw_context"):
        return []
    return _extract_numbers(rag_result["raw_context"])


def _flatten_ml_values(ml_result: list) -> list[float]:
    values = []
    if not ml_result:
        return values
    for item in ml_result:
        prob = item.get("churn_probability", None)
        if prob is not None:
            values.append(prob)               # e.g. 0.014
            values.append(round(prob * 100, 1))  # e.g. 1.4 (as shown in "1.4%")
    return values


def _is_supported(number: float, ground_truth: list[float], rel_tol: float = 0.02) -> bool:
    """Check if `number` is close to any value in ground_truth (relative tolerance)."""
    for gt in ground_truth:
        if gt == 0:
            if abs(number) < 0.01:
                return True
            continue
        if abs(number - gt) / abs(gt) <= rel_tol:
            return True
        # also allow exact small-integer matches (counts, years, etc.)
        if abs(number - gt) < 0.5:
            return True
    return False


def verify_numbers(answer: str, sql_result: dict = None, rag_result: dict = None,
                    ml_result: list = None) -> dict:
    ground_truth = (
        _flatten_sql_values(sql_result)
        + _flatten_rag_values(rag_result)
        + _flatten_ml_values(ml_result)
    )
    answer_numbers = _extract_numbers(answer)

    if not answer_numbers:
        return {"checked": 0, "supported": 0, "unsupported": []}

    unsupported = []
    supported_count = 0
    for num in answer_numbers:
        if _is_supported(num, ground_truth):
            supported_count += 1
        else:
            unsupported.append(num)

    return {
        "checked": len(answer_numbers),
        "supported": supported_count,
        "unsupported": unsupported,
    }


def verify_citations(answer: str, rag_result: dict = None) -> dict:
    if not rag_result:
        return {"checked": False, "valid": True, "issues": []}

    retrieved_sources = set(rag_result.get("sources", []))
    # find filenames mentioned in the answer (look for .txt tokens)
    mentioned = set(re.findall(r"[\w\-]+\.txt", answer))

    issues = []
    for m in mentioned:
        if m not in retrieved_sources:
            issues.append(f"Answer cites '{m}' but it was not among the retrieved documents.")

    return {"checked": True, "valid": len(issues) == 0, "issues": issues}


def verify_answer(answer: str, sql_result: dict = None, rag_result: dict = None,
                   ml_result: list = None) -> dict:
    """
    Run all verification checks and return a confidence score + warnings.

    confidence is a simple heuristic:
      - starts at 1.0
      - each unsupported number docks a fixed penalty
      - each citation issue docks a fixed penalty
    """
    number_check = verify_numbers(answer, sql_result, rag_result, ml_result)
    citation_check = verify_citations(answer, rag_result)

    warnings = []
    confidence = 1.0

    if number_check["checked"] > 0 and number_check["unsupported"]:
        ratio_bad = len(number_check["unsupported"]) / number_check["checked"]
        confidence -= min(0.6, ratio_bad)
        warnings.append(
            f"{len(number_check['unsupported'])} of {number_check['checked']} numbers in the "
            f"answer could not be matched back to the raw data: {number_check['unsupported']}"
        )

    if not citation_check["valid"]:
        confidence -= 0.3
        warnings.extend(citation_check["issues"])

    confidence = max(0.0, round(confidence, 2))

    return {
        "confidence": confidence,
        "warnings": warnings,
        "number_check": number_check,
        "citation_check": citation_check,
    }


if __name__ == "__main__":
    # quick manual test
    fake_sql_result = {
        "success": True,
        "columns": ["region", "total_mrr"],
        "rows": [["South", 33888450.81]],
    }
    good_answer = "The South region has the highest MRR at 33888450.81."
    bad_answer = "The South region has the highest MRR at 99999999.99."

    print("Good answer check:", verify_answer(good_answer, sql_result=fake_sql_result))
    print("Bad answer check:", verify_answer(bad_answer, sql_result=fake_sql_result))
