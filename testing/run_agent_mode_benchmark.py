#!/usr/bin/env python3
"""Run the same prompts across openai_api, simple, and advanced modes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "testing" / "cases" / "agent_mode_cases.json"
DEFAULT_OUT = PROJECT_ROOT / "testing" / "artifacts" / "benchmarks" / "agent_mode_benchmark_output.json"
DEFAULT_MODES = ["openai_api", "simple", "advanced"]
MODE_ENTRYPOINTS = {
    "openai_api": PROJECT_ROOT / "db_chatbot" / "chat_app_openai_api.py",
    "simple": PROJECT_ROOT / "db_chatbot" / "chat_app.py",
    "advanced": PROJECT_ROOT / "db_chatbot" / "chat_app_advanced.py",
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid case file: {path}")
    return [item for item in items if isinstance(item, dict)]


def run_case(
    question: str,
    mode: str,
    model: str,
    build_dir: Path,
    source_mode: str,
    api_data_root: Path,
    max_tool_rounds: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    cmd = [sys.executable, str(MODE_ENTRYPOINTS[mode]), "--query", question, "--model", model]
    if mode != "openai_api":
        cmd.extend(
            [
                "--build-dir",
                str(build_dir),
                "--source-mode",
                source_mode,
                "--api-data-root",
                str(api_data_root),
            ]
        )
    if mode == "advanced":
        cmd.extend(["--max-tool-rounds", str(max_tool_rounds)])
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        response = (proc.stdout or "").strip()
        error = None if proc.returncode == 0 else (proc.stderr or "non-zero exit")
    except Exception as exc:  # noqa: BLE001
        response = ""
        error = str(exc)
    elapsed = time.perf_counter() - started
    return {
        "mode": mode,
        "latency_sec": round(elapsed, 3),
        "response": response,
        "response_chars": len(response),
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare agent modes on the same prompt set.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--modes",
        action="append",
        choices=DEFAULT_MODES,
        default=None,
        help="Modes to run. Repeatable. Defaults to all three.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_ROOT / "db_chatbot" / "build_api_selected",
    )
    parser.add_argument(
        "--source-mode",
        choices=["build", "api_selected"],
        default="api_selected",
    )
    parser.add_argument(
        "--api-data-root",
        type=Path,
        default=PROJECT_ROOT / "db_chatbot" / "api_data",
    )
    parser.add_argument("--max-tool-rounds", type=int, default=5)
    args = parser.parse_args()

    modes = args.modes or DEFAULT_MODES
    cases = load_cases(args.cases)
    results: list[dict[str, Any]] = []

    for case in cases:
        question = str(case.get("question") or "").strip()
        index = str(case.get("index") or question)
        difficulty = str(case.get("difficulty") or "unknown")
        print(f"## {index}")
        print(f"difficulty: {difficulty}")
        print(f"question: {question}")
        for mode in modes:
            item = run_case(
                question=question,
                mode=mode,
                model=args.model,
                build_dir=args.build_dir,
                source_mode=args.source_mode,
                api_data_root=args.api_data_root,
                max_tool_rounds=args.max_tool_rounds,
            )
            results.append(
                {
                    "index": index,
                    "difficulty": difficulty,
                    "question": question,
                    **item,
                }
            )
            status = "ERROR" if item["error"] else "OK"
            print(
                f"- {mode}: {status}, latency={item['latency_sec']}s, chars={item['response_chars']}"
            )
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {args.out}")
    print()
    print("summary:")
    for mode in modes:
        mode_rows = [row for row in results if row["mode"] == mode]
        avg_latency = sum(row["latency_sec"] for row in mode_rows) / max(1, len(mode_rows))
        error_count = sum(1 for row in mode_rows if row["error"])
        avg_chars = sum(row["response_chars"] for row in mode_rows) / max(1, len(mode_rows))
        print(
            f"- {mode}: avg_latency={avg_latency:.3f}s, avg_chars={avg_chars:.1f}, errors={error_count}/{len(mode_rows)}"
        )
    print()
    print("summary by difficulty:")
    difficulties = sorted({str(row.get("difficulty") or "unknown") for row in results})
    for difficulty in difficulties:
        print(f"- {difficulty}:")
        for mode in modes:
            rows = [row for row in results if row["mode"] == mode and row.get("difficulty") == difficulty]
            if not rows:
                continue
            avg_latency = sum(row["latency_sec"] for row in rows) / len(rows)
            avg_chars = sum(row["response_chars"] for row in rows) / len(rows)
            error_count = sum(1 for row in rows if row["error"])
            print(
                f"  {mode}: avg_latency={avg_latency:.3f}s, avg_chars={avg_chars:.1f}, errors={error_count}/{len(rows)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
