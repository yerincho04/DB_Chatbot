#!/usr/bin/env python3
"""Append staged selected JSON rows into existing selected JSON files with dedupe.

Use after multi-year staging collection is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TABLE_CONFIG = {
    "brand_list_info": {
        "target": Path("db_chatbot/api_data/brand_list_info/output/brand_list_info_selected.json"),
        "staged_glob": "*/brand_list_info/brand_list_info_selected.json",
        "key_fields": ["brandMnno", "jngBizCrtraYr"],
    },
    "brand_frcs_stats": {
        "target": Path("db_chatbot/api_data/brand_frcs_stats/output/brand_frcs_stats_selected.json"),
        "staged_glob": "*/brand_frcs_stats/brand_frcs_stats_selected.json",
        "key_fields": ["brandNm", "yr", "corpNm"],
    },
    "brand_fntn_stats": {
        "target": Path("db_chatbot/api_data/brand_fntn_stats/output/brand_fntn_stats_selected.json"),
        "staged_glob": "*/brand_fntn_stats/brand_fntn_stats_selected.json",
        "key_fields": ["brandNm", "yr", "corpNm"],
    },
    "brand_brand_stats": {
        "target": Path("db_chatbot/api_data/brand_brand_stats/output/brand_brand_stats_selected.json"),
        "staged_glob": "*/brand_brand_stats/brand_brand_stats_selected.json",
        "key_fields": ["brandNm", "yr", "corpNm"],
    },
    "brand_interior_cost": {
        "target": Path("db_chatbot/api_data/brand_interior_cost/output/brand_interior_cost_selected.json"),
        "staged_glob": "*/brand_interior_cost/brand_interior_cost_selected.json",
        "key_fields": [
            "brandMnno",
            "jngBizCrtraYr",
            "storCrtraAr",
            "unitArIntrrAmtScopeVal",
            "intrrAmtScopeVal",
        ],
    },
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)]


def signature(row: dict[str, Any], key_fields: list[str]) -> str:
    vals = [str(row.get(k) or "").strip() for k in key_fields]
    base = "||".join(vals)
    if any(vals):
        return hashlib.sha1(base.encode("utf-8")).hexdigest()
    # fallback full-row signature
    return hashlib.sha1(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def append_table(staging_root: Path, table: str, dry_run: bool) -> tuple[int, int, int]:
    cfg = TABLE_CONFIG[table]
    target: Path = cfg["target"]
    key_fields: list[str] = cfg["key_fields"]

    current = load_rows(target)
    existing_sigs = {signature(r, key_fields) for r in current}

    staged_rows: list[dict[str, Any]] = []
    for p in sorted(staging_root.glob(cfg["staged_glob"])):
        staged_rows.extend(load_rows(p))

    add_rows: list[dict[str, Any]] = []
    for row in staged_rows:
        sig = signature(row, key_fields)
        if sig in existing_sigs:
            continue
        existing_sigs.add(sig)
        add_rows.append(row)

    merged = current + add_rows
    if not dry_run:
        target.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(current), len(add_rows), len(merged)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append staged multi-year rows into existing selected JSON files.")
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("db_chatbot/api_data_multiyear/staging"),
        help="Root folder created by collect_existing_brands_multiyear.py",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.staging_root.exists():
        raise RuntimeError(f"Staging root not found: {args.staging_root}")

    for table in TABLE_CONFIG:
        before, added, after = append_table(args.staging_root, table, args.dry_run)
        mode = "[DRY-RUN]" if args.dry_run else "[APPLIED]"
        print(f"{mode} {table}: before={before}, added={added}, after={after}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
