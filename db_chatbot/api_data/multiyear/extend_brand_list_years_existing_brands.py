#!/usr/bin/env python3
"""Extend brand_list_info with new years for existing brands only.

- Uses current brand set from existing brand_list_info_selected.json.
- Fetches brand_list_info per target year into staging.
- Filters staged rows to current brand set only (no new brands).
- Optionally appends deduped rows into existing selected JSON.

No other API tables are touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


EXISTING_SELECTED = Path("db_chatbot/api_data/brand_list_info/output/brand_list_info_selected.json")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Extend brand_list_info by year for existing brands only.")
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--num-rows", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument(
        "--staging-root",
        type=Path,
        default=Path("db_chatbot/api_data_multiyear/staging_existing_brands_only"),
    )
    p.add_argument("--apply", action="store_true", help="Actually append into existing selected JSON.")
    args = p.parse_args()

    existing_rows = load_rows(EXISTING_SELECTED)
    if not existing_rows:
        raise RuntimeError(f"Existing selected JSON is empty or missing: {EXISTING_SELECTED}")

    existing_mnno = {str(r.get("brandMnno") or "").strip() for r in existing_rows if str(r.get("brandMnno") or "").strip()}
    existing_name = {str(r.get("brandNm") or "").strip() for r in existing_rows if str(r.get("brandNm") or "").strip()}

    py = ".venv/bin/python"
    staged_filtered_all: list[dict[str, Any]] = []

    for year in args.years:
        out_dir = args.staging_root / str(year) / "brand_list_info"
        run(
            [
                py,
                "db_chatbot/api_data/brand_list_info/fetch_brand_list_info.py",
                "--year",
                str(year),
                "--num-rows",
                str(args.num_rows),
                "--max-pages",
                str(args.max_pages),
                "--output-dir",
                str(out_dir),
            ]
        )

        staged_path = out_dir / "brand_list_info_selected.json"
        staged_rows = load_rows(staged_path)
        matched = [
            r
            for r in staged_rows
            if (str(r.get("brandMnno") or "").strip() in existing_mnno)
            or (str(r.get("brandNm") or "").strip() in existing_name)
        ]

        filtered_path = out_dir / "brand_list_info_selected_existing_brands_only.json"
        filtered_path.write_text(json.dumps(matched, ensure_ascii=False, indent=2), encoding="utf-8")

        staged_filtered_all.extend(matched)
        print(f"year={year}: staged_rows={len(staged_rows)}, matched_existing={len(matched)}")

    # Dedupe and optional append
    def sig(r: dict[str, Any]) -> tuple[str, str]:
        return (str(r.get("brandMnno") or "").strip(), str(r.get("jngBizCrtraYr") or "").strip())

    existing_sig = {sig(r) for r in existing_rows}
    add_rows: list[dict[str, Any]] = []
    for r in staged_filtered_all:
        s = sig(r)
        if s in existing_sig:
            continue
        existing_sig.add(s)
        add_rows.append(r)

    print(f"existing={len(existing_rows)}, add={len(add_rows)}, after={len(existing_rows) + len(add_rows)}")

    if args.apply:
        merged = existing_rows + add_rows
        EXISTING_SELECTED.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Applied append to: {EXISTING_SELECTED}")
    else:
        print("Dry-run only. Add --apply to append.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
