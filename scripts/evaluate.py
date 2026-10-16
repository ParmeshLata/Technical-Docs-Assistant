"""Real-provider HTTP checks; never substitutes a fake model. Run with the API running."""

import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    arguments = parser.parse_args()
    cases = json.loads(
        (Path(__file__).resolve().parents[1] / "examples" / "evaluation.json").read_text()
    )
    results = []
    with httpx.Client(base_url=arguments.url, timeout=200, trust_env=False) as client:
        for case in cases:
            response = client.post("/query", json={"question": case["question"]})
            response.raise_for_status()
            body = response.json()
            matched = body["status"] == case["expected_status"]
            if case["expected_source"]:
                matched = matched and any(
                    case["expected_source"] in source["source"] for source in body["sources"]
                )
            else:
                matched = matched and not body["sources"]
            results.append(
                {
                    "question": case["question"],
                    "contract_passed": matched,
                    "manual_review": case["check"],
                    "response": body,
                }
            )
    # A matching status/source is a contract check, not a factual-accuracy score.
    print(json.dumps({"results": results, "note": "Manually review factual grounding."}, indent=2))
    if not all(result["contract_passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
