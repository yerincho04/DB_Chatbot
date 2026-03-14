#!/usr/bin/env python3
"""Rebuild API Excel tables from existing selected JSON files, adding brand_id.

No API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook

from brand_id_utils import build_brand_id_maps, find_brand_id


BASE = Path("db_chatbot/api_data")
BRAND_LIST_JSON = BASE / "brand_list_info/output/brand_list_info_selected.json"

TABLES: list[dict[str, str]] = [
    {
        "name": "brand_list_info",
        "selected_json": "brand_list_info/output/brand_list_info_selected.json",
        "xlsx": "brand_list_info/output/brand_list_info.xlsx",
        "sheet": "brand_list_info",
    },
    {
        "name": "brand_frcs_stats",
        "selected_json": "brand_frcs_stats/output/brand_frcs_stats_selected.json",
        "xlsx": "brand_frcs_stats/output/brand_frcs_stats.xlsx",
        "sheet": "brand_frcs_stats",
    },
    {
        "name": "brand_interior_cost",
        "selected_json": "brand_interior_cost/output/brand_interior_cost_selected.json",
        "xlsx": "brand_interior_cost/output/brand_interior_cost.xlsx",
        "sheet": "brand_interior_cost",
    },
    {
        "name": "brand_fntn_stats",
        "selected_json": "brand_fntn_stats/output/brand_fntn_stats_selected.json",
        "xlsx": "brand_fntn_stats/output/brand_fntn_stats.xlsx",
        "sheet": "brand_fntn_stats",
    },
    {
        "name": "brand_brand_stats",
        "selected_json": "brand_brand_stats/output/brand_brand_stats_selected.json",
        "xlsx": "brand_brand_stats/output/brand_brand_stats.xlsx",
        "sheet": "brand_brand_stats",
    },
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def collect_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)
    return columns


def write_excel(
    rows: list[dict[str, Any]],
    xlsx_path: Path,
    sheet_name: str,
    brand_mnno_to_id: dict[str, int],
    brand_nm_to_id: dict[str, int],
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name[:31]

    cols = collect_columns(rows)
    ws.append(["brand_id", *cols])
    for row in rows:
        brand_id = find_brand_id(row, brand_mnno_to_id, brand_nm_to_id)
        ws.append([brand_id, *[row.get(c) for c in cols]])

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)


def main() -> int:
    brand_rows = load_rows(BRAND_LIST_JSON)
    brand_mnno_to_id, brand_nm_to_id = build_brand_id_maps(BRAND_LIST_JSON, fallback_rows=brand_rows)

    print(f"Brand map size by mnno: {len(brand_mnno_to_id)}")
    print(f"Brand map size by name: {len(brand_nm_to_id)}")

    for t in TABLES:
        selected_path = BASE / t["selected_json"]
        xlsx_path = BASE / t["xlsx"]
        rows = load_rows(selected_path)
        write_excel(rows, xlsx_path, t["sheet"], brand_mnno_to_id, brand_nm_to_id)
        print(f"[{t['name']}] rows={len(rows)} -> {xlsx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
