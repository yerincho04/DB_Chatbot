#!/usr/bin/env python3
"""Collect multi-year data for existing brand set into a staging directory.

This script DOES NOT modify current api_data output files.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect multi-year data into staging (no overwrite).")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Years to collect, e.g. --years 2021 2022 2023 2024",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("db_chatbot/api_data_multiyear/staging"),
        help="Staging root folder.",
    )
    parser.add_argument("--num-rows", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--single-page", action="store_true")
    args = parser.parse_args()

    py = ".venv/bin/python"

    for year in args.years:
        ydir = args.staging_root / str(year)
        bl_out = ydir / "brand_list_info"

        # 1) Brand list info for that year (base brand set source for this year).
        cmd = [
            py,
            "db_chatbot/api_data/brand_list_info/fetch_brand_list_info.py",
            "--year",
            str(year),
            "--num-rows",
            str(args.num_rows),
            "--max-pages",
            str(args.max_pages),
            "--output-dir",
            str(bl_out),
        ]
        if args.single_page:
            cmd.append("--single-page")
        run(cmd)

        brand_list_json = bl_out / "brand_list_info_selected.json"

        # 2) brand_frcs_stats filtered by that year's brand list.
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
            str(brand_list_json),
            "--output-dir",
            str(ydir / "brand_frcs_stats"),
        ]
        if args.single_page:
            cmd.append("--single-page")
        run(cmd)

        # 3) brand_fntn_stats filtered by that year's brand list.
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
            str(brand_list_json),
            "--output-dir",
            str(ydir / "brand_fntn_stats"),
        ]
        if args.single_page:
            cmd.append("--single-page")
        run(cmd)

        # 4) brand_brand_stats filtered by that year's brand list.
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
            str(brand_list_json),
            "--output-dir",
            str(ydir / "brand_brand_stats"),
        ]
        if args.single_page:
            cmd.append("--single-page")
        run(cmd)

        # 5) brand_interior_cost bulk mode using that year's brand list.
        run(
            [
                py,
                "db_chatbot/api_data/brand_interior_cost/fetch_brand_interior_cost.py",
                "--year",
                str(year),
                "--brand-list-json",
                str(brand_list_json),
                "--output-dir",
                str(ydir / "brand_interior_cost"),
            ]
        )

    print("Staging collection complete:", args.staging_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
