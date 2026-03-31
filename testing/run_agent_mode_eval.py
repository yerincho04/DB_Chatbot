#!/usr/bin/env python3
"""Evaluate agent-mode benchmark output with a lightweight rubric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "testing" / "cases" / "agent_mode_cases.json"
DEFAULT_INPUT = PROJECT_ROOT / "testing" / "artifacts" / "benchmarks" / "agent_mode_benchmark_output.json"
DEFAULT_OUT = PROJECT_ROOT / "testing" / "artifacts" / "reports" / "agent_mode_eval_output.json"

ACTION_ONLY_HINTS = (
    "조회하겠습니다",
    "확인하겠습니다",
    "찾아보겠습니다",
    "사용해 조건에 맞는",
    "잠시만 기다려",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def looks_json_stub(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def looks_action_only(text: str) -> bool:
    norm = normalize(text)
    if len(norm) > 120:
        return False
    return any(hint in norm for hint in ACTION_ONLY_HINTS)


def group_hit(response: str, options: list[str]) -> bool:
    norm = normalize(response)
    return any(normalize(option) in norm for option in options)


def evaluate_response(case_eval: dict[str, Any], response: str, error: str | None) -> dict[str, Any]:
    coverage_groups = case_eval.get("coverage_groups") or []
    missing_groups = case_eval.get("missing_groups") or []

    flags: list[str] = []
    if error:
        flags.append("runtime_error")
    if not str(response).strip():
        flags.append("empty_response")
    if looks_json_stub(response):
        flags.append("json_stub")
    if looks_action_only(response):
        flags.append("action_only")

    coverage_hits = sum(1 for group in coverage_groups if group_hit(response, group))
    coverage_total = len(coverage_groups)
    coverage_score = 0.0 if coverage_total == 0 else coverage_hits / coverage_total

    missing_hits = sum(1 for group in missing_groups if group_hit(response, group))
    missing_total = len(missing_groups)
    missing_score = 1.0 if missing_total == 0 else missing_hits / missing_total

    answered = 0 if error or "empty_response" in flags or "json_stub" in flags or "action_only" in flags else 1
    final_score = (0.7 * coverage_score) + (0.2 * answered) + (0.1 * missing_score)
    if error:
        final_score = 0.0

    return {
        "answered": answered,
        "coverage_hits": coverage_hits,
        "coverage_total": coverage_total,
        "coverage_score": round(coverage_score, 4),
        "missing_hits": missing_hits,
        "missing_total": missing_total,
        "missing_score": round(missing_score, 4),
        "final_score": round(final_score, 4),
        "flags": flags,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate benchmark output for agent modes.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cases_payload = load_json(args.cases)
    benchmark_rows = load_json(args.input)

    case_map = {
        str(item.get("index")): item
        for item in cases_payload.get("items", [])
        if isinstance(item, dict) and item.get("index")
    }

    detailed_rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        index = str(row.get("index") or "")
        case = case_map.get(index)
        if not case:
            continue
        metrics = evaluate_response(
            case_eval=case.get("eval") or {},
            response=str(row.get("response") or ""),
            error=row.get("error"),
        )
        detailed_rows.append(
            {
                "index": index,
                "difficulty": row.get("difficulty") or str(case.get("difficulty") or "unknown"),
                "mode": row.get("mode"),
                "latency_sec": row.get("latency_sec"),
                "response_chars": row.get("response_chars"),
                "error": row.get("error"),
                **metrics,
            }
        )

    summary_by_mode: dict[str, dict[str, Any]] = {}
    summary_by_difficulty: dict[str, dict[str, dict[str, Any]]] = {}
    for row in detailed_rows:
        mode = str(row["mode"])
        difficulty = str(row.get("difficulty") or "unknown")
        bucket = summary_by_mode.setdefault(
            mode,
            {
                "mode": mode,
                "count": 0,
                "answered_rate": 0.0,
                "avg_coverage_score": 0.0,
                "avg_missing_score": 0.0,
                "avg_final_score": 0.0,
                "avg_latency_sec": 0.0,
                "avg_response_chars": 0.0,
                "flag_counts": {},
            },
        )
        bucket["count"] += 1
        bucket["answered_rate"] += row["answered"]
        bucket["avg_coverage_score"] += row["coverage_score"]
        bucket["avg_missing_score"] += row["missing_score"]
        bucket["avg_final_score"] += row["final_score"]
        bucket["avg_latency_sec"] += float(row.get("latency_sec") or 0.0)
        bucket["avg_response_chars"] += float(row.get("response_chars") or 0.0)
        for flag in row.get("flags", []):
            flag_counts = bucket["flag_counts"]
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

        diff_bucket = summary_by_difficulty.setdefault(difficulty, {}).setdefault(
            mode,
            {
                "mode": mode,
                "difficulty": difficulty,
                "count": 0,
                "answered_rate": 0.0,
                "avg_coverage_score": 0.0,
                "avg_missing_score": 0.0,
                "avg_final_score": 0.0,
                "avg_latency_sec": 0.0,
                "avg_response_chars": 0.0,
                "flag_counts": {},
            },
        )
        diff_bucket["count"] += 1
        diff_bucket["answered_rate"] += row["answered"]
        diff_bucket["avg_coverage_score"] += row["coverage_score"]
        diff_bucket["avg_missing_score"] += row["missing_score"]
        diff_bucket["avg_final_score"] += row["final_score"]
        diff_bucket["avg_latency_sec"] += float(row.get("latency_sec") or 0.0)
        diff_bucket["avg_response_chars"] += float(row.get("response_chars") or 0.0)
        for flag in row.get("flags", []):
            flag_counts = diff_bucket["flag_counts"]
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    for bucket in summary_by_mode.values():
        count = max(1, int(bucket["count"]))
        bucket["answered_rate"] = round(bucket["answered_rate"] / count, 4)
        bucket["avg_coverage_score"] = round(bucket["avg_coverage_score"] / count, 4)
        bucket["avg_missing_score"] = round(bucket["avg_missing_score"] / count, 4)
        bucket["avg_final_score"] = round(bucket["avg_final_score"] / count, 4)
        bucket["avg_latency_sec"] = round(bucket["avg_latency_sec"] / count, 4)
        bucket["avg_response_chars"] = round(bucket["avg_response_chars"] / count, 1)

    summary_by_difficulty_rows: list[dict[str, Any]] = []
    for difficulty, mode_map in sorted(summary_by_difficulty.items()):
        for bucket in mode_map.values():
            count = max(1, int(bucket["count"]))
            bucket["answered_rate"] = round(bucket["answered_rate"] / count, 4)
            bucket["avg_coverage_score"] = round(bucket["avg_coverage_score"] / count, 4)
            bucket["avg_missing_score"] = round(bucket["avg_missing_score"] / count, 4)
            bucket["avg_final_score"] = round(bucket["avg_final_score"] / count, 4)
            bucket["avg_latency_sec"] = round(bucket["avg_latency_sec"] / count, 4)
            bucket["avg_response_chars"] = round(bucket["avg_response_chars"] / count, 1)
            summary_by_difficulty_rows.append(bucket)

    output = {
        "input_file": str(args.input),
        "cases_file": str(args.cases),
        "summary_by_mode": sorted(summary_by_mode.values(), key=lambda x: x["mode"]),
        "summary_by_difficulty": sorted(summary_by_difficulty_rows, key=lambda x: (x["difficulty"], x["mode"])),
        "detailed_rows": detailed_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {args.out}")
    print()
    print("summary:")
    for row in output["summary_by_mode"]:
        print(
            f"- {row['mode']}: final={row['avg_final_score']:.3f}, "
            f"coverage={row['avg_coverage_score']:.3f}, answered={row['answered_rate']:.3f}, "
            f"latency={row['avg_latency_sec']:.3f}s, chars={row['avg_response_chars']:.1f}, "
            f"flags={row['flag_counts']}"
        )
    print()
    print("summary by difficulty:")
    for row in output["summary_by_difficulty"]:
        print(
            f"- {row['difficulty']} / {row['mode']}: final={row['avg_final_score']:.3f}, "
            f"coverage={row['avg_coverage_score']:.3f}, answered={row['answered_rate']:.3f}, "
            f"latency={row['avg_latency_sec']:.3f}s, chars={row['avg_response_chars']:.1f}, "
            f"flags={row['flag_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
