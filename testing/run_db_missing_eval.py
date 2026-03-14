#!/usr/bin/env python3
"""Run a lightweight missing-data eval against db_chatbot/chat_app.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "testing" / "db_missing_question.json"

MISSING_HINTS = (
    "없습니다",
    "없음",
    "없다고",
    "찾을 수 없습니다",
    "찾을수 없습니다",
    "원천 데이터에 없음",
    "원천 데이터에 없습니다",
    "데이터베이스에 없습니다",
    "데이터에 없습니다",
    "정보가 없습니다",
    "not disclosed",
    "not found",
    "unavailable",
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid case file: {path}")
    return [item for item in items if isinstance(item, dict)]


def run_chatbot(
    query: str,
    model: str,
    source_mode: str,
    api_data_root: Path,
    build_dir: Path,
) -> tuple[int, str, str]:
    cmd = [
        sys.executable,
        "db_chatbot/chat_app.py",
        "--query",
        query,
        "--model",
        model,
        "--source-mode",
        source_mode,
        "--api-data-root",
        str(api_data_root),
        "--build-dir",
        str(build_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def looks_grounded_missing(response: str) -> bool:
    norm = normalize(response)
    return any(hint in norm for hint in MISSING_HINTS)


def likely_hallucinated_value(response: str) -> bool:
    norm = normalize(response)
    if not norm:
        return False
    if "없" in norm or "not found" in norm or "unavailable" in norm:
        return False
    digits = sum(ch.isdigit() for ch in response)
    return digits >= 4


def extract_query_brand(question: str) -> str:
    q = question.strip()
    markers = ("의 ", "은 ", "는 ", "이 ", "가 ")
    cut = len(q)
    for marker in markers:
        idx = q.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return q[:cut].strip(" ?")


def mentions_other_brand(question: str, response: str) -> bool:
    query_brand = normalize(extract_query_brand(question))
    response_norm = normalize(response)
    if not query_brand or query_brand not in response_norm:
        return False
    if "(" in response and ")" in response:
        return True
    return False


def expected_missing_phrase(case: dict[str, Any]) -> str:
    expected = str(case.get("expected_answer") or "").strip()
    if not expected:
        return ""
    for phrase in (
        "현재 데이터베이스에 없습니다",
        "현재 데이터 베이스에 없습니다",
        "원천 데이터에 없습니다",
        "데이터에 없습니다",
        "정보가 없습니다",
    ):
        if phrase in expected:
            return phrase
    return ""


def evaluate_case(case: dict[str, Any], response: str, returncode: int, stderr: str) -> tuple[bool, str]:
    question = str(case.get("question") or "")
    if returncode != 0:
        return False, f"process_error: {stderr or 'chat_app exited non-zero'}"
    if not response:
        return False, "empty_response"
    if not looks_grounded_missing(response):
        return False, "missing_signal_not_found"
    if likely_hallucinated_value(response):
        return False, "possible_hallucinated_numeric_value"
    if mentions_other_brand(question, response):
        return False, "resolved_to_different_brand_candidate"
    expected_phrase = expected_missing_phrase(case)
    if expected_phrase and expected_phrase not in response:
        return False, f"expected_phrase_missing:{expected_phrase}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run missing-data eval cases against the chatbot.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--source-mode", choices=["api_selected", "build"], default="api_selected")
    parser.add_argument("--api-data-root", type=Path, default=PROJECT_ROOT / "db_chatbot" / "api_data")
    parser.add_argument("--build-dir", type=Path, default=PROJECT_ROOT / "db_chatbot" / "build")
    parser.add_argument("--show-response", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    passed = 0

    for case in cases:
        query = str(case.get("question") or "").strip()
        index = str(case.get("index") or query)
        code, stdout, stderr = run_chatbot(
            query=query,
            model=args.model,
            source_mode=args.source_mode,
            api_data_root=args.api_data_root,
            build_dir=args.build_dir,
        )
        ok, reason = evaluate_case(case, stdout, code, stderr)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {index}")
        print(f"query: {query}")
        print(f"check: {reason}")
        if args.show_response or not ok:
            print(f"response: {stdout or '<empty>'}")
        if stderr and not ok:
            print(f"stderr: {stderr}")
        print()
        passed += int(ok)

    total = len(cases)
    print(f"summary: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
