#!/usr/bin/env python3
"""Run deterministic BrandDataStore smoke evals from JSON case files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_CHATBOT_DIR = PROJECT_ROOT / "db_chatbot"
if str(DB_CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(DB_CHATBOT_DIR))

from data_access import BrandDataStore  # noqa: E402


DEFAULT_CASE_FILES = [
    PROJECT_ROOT / "testing" / "cases" / "overview_cases.json",
    PROJECT_ROOT / "testing" / "cases" / "compare_cases.json",
    PROJECT_ROOT / "testing" / "cases" / "filter_cases.json",
    PROJECT_ROOT / "testing" / "cases" / "trend_cases.json",
    PROJECT_ROOT / "testing" / "cases" / "resolver_cases.json",
]


def load_case_file(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    case_type = str(payload.get("type") or path.stem)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid case file: {path}")
    return case_type, [item for item in items if isinstance(item, dict)]


def extract_values(obj: Any, path: str) -> list[Any]:
    tokens = path.split(".")
    current = [obj]
    for token in tokens:
        next_items: list[Any] = []
        list_mode = token.endswith("[]")
        key = token[:-2] if list_mode else token
        for item in current:
            if isinstance(item, dict):
                value = item.get(key)
            else:
                value = None
            if list_mode:
                if isinstance(value, list):
                    next_items.extend(value)
            elif value is not None:
                next_items.append(value)
        current = next_items
    return current


def single_value(obj: Any, path: str) -> Any:
    values = extract_values(obj, path)
    if len(values) != 1:
        raise AssertionError(f"path '{path}' expected single value, got {len(values)}")
    return values[0]


def check_equals(result: Any, checks: dict[str, Any]) -> None:
    for path, expected in checks.items():
      actual = single_value(result, path)
      if actual != expected:
          raise AssertionError(f"equals failed at '{path}': expected {expected!r}, got {actual!r}")


def check_min(result: Any, checks: dict[str, float]) -> None:
    for path, threshold in checks.items():
        actual = single_value(result, path)
        if actual is None or actual < threshold:
            raise AssertionError(f"min failed at '{path}': expected >= {threshold!r}, got {actual!r}")


def check_max(result: Any, checks: dict[str, float]) -> None:
    for path, threshold in checks.items():
        actual = single_value(result, path)
        if actual is None or actual > threshold:
            raise AssertionError(f"max failed at '{path}': expected <= {threshold!r}, got {actual!r}")


def check_not_null(result: Any, paths: list[str]) -> None:
    for path in paths:
        values = extract_values(result, path)
        if not values:
            raise AssertionError(f"not_null failed at '{path}': no value found")
        if any(value is None for value in values):
            raise AssertionError(f"not_null failed at '{path}': found null value")


def check_len_at_least(result: Any, checks: dict[str, int]) -> None:
    for path, minimum in checks.items():
        values = extract_values(result, path)
        if len(values) == 1 and isinstance(values[0], (list, dict, str)):
            size = len(values[0])
        else:
            size = len(values)
        if size < minimum:
            raise AssertionError(f"len_at_least failed at '{path}': expected >= {minimum}, got {size}")


def check_contains(result: Any, checks: dict[str, Any]) -> None:
    for path, expected in checks.items():
        values = extract_values(result, path)
        if not values:
            raise AssertionError(f"contains failed at '{path}': no values found")
        if isinstance(expected, str):
            matched = any(expected in str(value) for value in values)
        else:
            matched = expected in values
        if not matched:
            raise AssertionError(f"contains failed at '{path}': expected {expected!r}, got {values!r}")


def validate_filter_results(case_input: dict[str, Any], result: dict[str, Any]) -> None:
    conditions = case_input.get("conditions") or []
    for item in result.get("results", []):
        metrics = item.get("metrics") or {}
        for cond in conditions:
            field = cond["field"]
            op = cond["op"]
            target = cond["value"]
            actual = metrics.get(field)
            if actual is None:
                raise AssertionError(f"filter result missing metric '{field}'")
            if op == ">=" and not (actual >= target):
                raise AssertionError(f"filter validation failed: {field}={actual} < {target}")
            if op == ">" and not (actual > target):
                raise AssertionError(f"filter validation failed: {field}={actual} <= {target}")
            if op == "<=" and not (actual <= target):
                raise AssertionError(f"filter validation failed: {field}={actual} > {target}")
            if op == "<" and not (actual < target):
                raise AssertionError(f"filter validation failed: {field}={actual} >= {target}")
            if op == "==" and not (actual == target):
                raise AssertionError(f"filter validation failed: {field}={actual} != {target}")
            if op == "!=" and not (actual != target):
                raise AssertionError(f"filter validation failed: {field}={actual} == {target}")


def validate_trend_metrics(case_input: dict[str, Any], result: dict[str, Any]) -> None:
    metrics = case_input.get("metrics") or []
    if not metrics:
        return
    timeline = result.get("timeline") or []
    if not timeline:
        raise AssertionError("trend validation failed: empty timeline")
    for point in timeline:
        raw = point.get("raw") or {}
        for metric in metrics:
            if metric not in raw:
                raise AssertionError(f"trend validation failed: '{metric}' missing from timeline point")


def run_tool(store: BrandDataStore, tool_name: str, payload: dict[str, Any]) -> Any:
    if tool_name == "brand_overview":
        return store.get_brand_overview(**payload)
    if tool_name == "brand_compare":
        return store.get_brand_compare(**payload)
    if tool_name == "brand_filter_search":
        return store.get_brand_filter_search(**payload)
    if tool_name == "brand_trend":
        return store.get_brand_trend(**payload)
    if tool_name == "resolver_debug":
        return store.resolve_brand_debug(**payload)
    raise ValueError(f"Unsupported tool in case file: {tool_name}")


def evaluate_case(store: BrandDataStore, case: dict[str, Any]) -> tuple[bool, str]:
    tool_name = str(case.get("tool") or "").strip()
    payload = case.get("input") or {}
    assertions = case.get("assertions") or {}
    expected_error = assertions.get("error_contains")

    try:
        result = run_tool(store, tool_name, payload)
        if expected_error:
            return False, f"expected_error_not_raised:{expected_error}"
    except Exception as exc:  # noqa: BLE001
        if expected_error and expected_error in str(exc):
            return True, "ok"
        return False, f"unexpected_error:{exc}"

    try:
        if assertions.get("equals"):
            check_equals(result, assertions["equals"])
        if assertions.get("min"):
            check_min(result, assertions["min"])
        if assertions.get("max"):
            check_max(result, assertions["max"])
        if assertions.get("not_null"):
            check_not_null(result, assertions["not_null"])
        if assertions.get("len_at_least"):
            check_len_at_least(result, assertions["len_at_least"])
        if assertions.get("contains"):
            check_contains(result, assertions["contains"])
        if tool_name == "brand_filter_search":
            validate_filter_results(payload, result)
        if tool_name == "brand_trend":
            validate_trend_metrics(payload, result)
    except AssertionError as exc:
        return False, str(exc)

    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic BrandDataStore eval cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        action="append",
        default=None,
        help="Specific JSON case file to run. Repeatable. Defaults to all testing/*_cases.json files.",
    )
    parser.add_argument(
        "--source-mode",
        choices=["build", "api_selected"],
        default="build",
        help="Data source mode for BrandDataStore.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_ROOT / "db_chatbot" / "build_api_selected",
        help="Directory containing normalized build JSON tables.",
    )
    parser.add_argument(
        "--api-data-root",
        type=Path,
        default=PROJECT_ROOT / "db_chatbot" / "api_data",
        help="Root directory containing selected API JSON files.",
    )
    args = parser.parse_args()

    store = BrandDataStore(
        build_dir=args.build_dir,
        source_mode=args.source_mode,
        api_data_root=args.api_data_root,
    )

    case_files = args.cases or DEFAULT_CASE_FILES
    total = 0
    passed = 0

    for case_file in case_files:
        case_type, items = load_case_file(case_file)
        print(f"## {case_type} ({case_file.name})")
        for case in items:
            total += 1
            index = str(case.get("index") or total)
            ok, reason = evaluate_case(store, case)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {index} {reason}")
            passed += int(ok)
        print()

    print(f"summary: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
