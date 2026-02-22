#!/usr/bin/env python3
"""Resolver diagnostics CLI.

Example:
  python db_chatbot/testing/resolver_debug.py --query "비비큐"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_access import BrandDataStore


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug brand resolver result.")
    parser.add_argument("--query", required=True, help="Raw brand mention to resolve.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidates to include.")
    args = parser.parse_args()

    load_env_file(BASE_DIR / ".env")
    store = BrandDataStore(build_dir=BASE_DIR / "build")
    result = store.resolve_brand_debug(args.query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
