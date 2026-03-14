#!/usr/bin/env python3
"""Extend non-brand-list tables with new years for existing brands only.

Flow:
1) Collect year-wise staged data for other tables using a fixed brand list JSON.
2) Append staged rows into existing selected JSONs with dedupe.

This script does not modify brand_list_info_selected.json.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Extend other tables with new years for existing brands.")
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument(
        "--brand-list-json",
        type=Path,
        default=Path("db_chatbot/api_data/brand_list_info/output/brand_list_info_selected.json"),
        help="Fixed brand list used for all years.",
    )
    p.add_argument(
        "--staging-root",
        type=Path,
        default=Path("db_chatbot/api_data_multiyear/staging_existing_brands_only_other_tables"),
    )
    p.add_argument("--num-rows", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument("--single-page", action="store_true")
    p.add_argument("--dry-run-append", action="store_true")
    p.add_argument("--apply", action="store_true", help="Apply append to existing selected JSONs.")
    args = p.parse_args()

    if not args.brand_list_json.exists():
        raise RuntimeError(f"brand list json not found: {args.brand_list_json}")

    py = ".venv/bin/python"

    for year in args.years:
        ydir = args.staging_root / str(year)

        # 1) brand_frcs_stats
        cmd = [
            py,
            "db_chatbot/api_data/brand_frcs_stats/fetch_brand_frcs_stats.py",
            "--year",
            str(year),
            "--num-rows",
            str(args.num_rows),
            "--max-pages",
            str(args.max_pages),
            "--brand-list-json",
            str(args.brand_list_json),
            "--output-dir",
            str(ydir / "brand_frcs_stats"),
        ]
        if args.single_page:
            cmd.append("--single-page")
        run(cmd)

        # 2) brand_fntn_stats
        cmd = [
            py,
            "db_chatbot/api_data/brand_fntn_stats/fetch_brand_fntn_stats.py",
            "--year",
            str(year),
            "--num-rows",
            str(args.num_rows),
            "--max-pages",
            str(args.max_pages),
            "--brand-list-json",
            str(args.brand_list_json),
            "--output-dir",
            str(ydir / "brand_fntn_stats"),
        ]
        if args.single_page:
            cmd.append("--single-page")
        run(cmd)

        # 3) brand_brand_stats
        cmd = [
            py,
            "db_chatbot/api_data/brand_brand_stats/fetch_brand_brand_stats.py",
            "--year",
            str(year),
            "--num-rows",
            str(args.num_rows),
            "--max-pages",
            str(args.max_pages),
            "--brand-list-json",
            str(args.brand_list_json),
            "--output-dir",
            str(ydir / "brand_brand_stats"),
        ]
        if args.single_page:
            cmd.append("--single-page")
        run(cmd)

        # 4) brand_interior_cost (bulk mode)
        run(
            [
                py,
                "db_chatbot/api_data/brand_interior_cost/fetch_brand_interior_cost.py",
                "--year",
                str(year),
                "--brand-list-json",
                str(args.brand_list_json),
                "--output-dir",
                str(ydir / "brand_interior_cost"),
            ]
        )

    # Append staged into existing selected JSONs
    append_cmd = [
        py,
        "db_chatbot/api_data/multiyear/append_staging_into_existing_selected.py",
        "--staging-root",
        str(args.staging_root),
    ]

    if args.dry_run_append and not args.apply:
        run(append_cmd + ["--dry-run"])
        print("Dry-run append complete. Re-run with --apply to write changes.")
        return 0

    if args.apply:
        run(append_cmd)
        print("Append applied.")
    else:
        run(append_cmd + ["--dry-run"])
        print("No apply flag set. Dry-run shown only.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
