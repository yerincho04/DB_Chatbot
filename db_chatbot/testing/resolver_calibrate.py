#!/usr/bin/env python3
"""Batch calibration utility for brand resolver behavior."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_access import BrandDataStore, _normalize_brand_key


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key.strip()] = value.strip()


def load_queries_from_file(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            lines.append(text)
    return lines


def default_queries(store: BrandDataStore) -> list[str]:
    # Data-driven defaults: exact names + simple transformed variants.
    out: list[str] = []
    for row in store.brand_master:
        name = row["brand_name"]
        out.append(name)
        no_space = "".join(name.split())
        if no_space and no_space != name:
            out.append(no_space)
        lower = name.lower()
        if lower != name:
            out.append(lower)
        norm = _normalize_brand_key(name)
        if norm and norm not in {name, no_space, lower}:
            out.append(norm)
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-check resolver results.")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Optional text file with one query per line.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Inline query (repeatable).",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print JSON list output.")
    args = parser.parse_args()

    load_env_file(BASE_DIR / ".env")
    store = BrandDataStore(build_dir=BASE_DIR / "build")
    queries: list[str] = []
    queries.extend(args.query)
    if args.queries_file:
        queries.extend(load_queries_from_file(args.queries_file))
    if not queries:
        queries = default_queries(store)

    results = []
    for q in queries:
        res = store.resolve_brand_debug(q, top_k=args.top_k)
        results.append(res)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for res in results:
        q = res["query_text"]
        status = res["status"]
        if status == "resolved" and res.get("match"):
            m = res["match"]
            print(f"[RESOLVED] {q} -> {m['brand_name']} (conf={m['confidence']:.4f}, reason={res['reason']})")
        elif status == "ambiguous":
            cands = ", ".join(f"{c['brand_name']}:{c['confidence']:.2f}" for c in res.get("candidates", []))
            print(f"[AMBIGUOUS] {q} -> {cands}")
        else:
            cands = ", ".join(f"{c['brand_name']}:{c['confidence']:.2f}" for c in res.get("candidates", []))
            suffix = f" candidates={cands}" if cands else ""
            print(f"[NOT_FOUND] {q} reason={res.get('reason')}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
