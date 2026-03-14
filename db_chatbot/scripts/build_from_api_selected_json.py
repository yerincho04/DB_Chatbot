#!/usr/bin/env python3
"""Build db_chatbot/build tables directly from api_data selected JSON files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import sys

sys.path.append(str((Path(__file__).resolve().parents[1] / "api_data")))
from brand_id_utils import build_brand_id_maps, find_brand_id


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return [r for r in data if isinstance(r, dict)]


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def parse_range_mid(value: Any) -> int | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return None
    vals = [float(n) for n in nums]
    return int(round(sum(vals) / len(vals)))


def non_empty_count(row: dict[str, Any]) -> int:
    return sum(1 for v in row.values() if v is not None and str(v).strip() != "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build normalized JSON tables from api_data selected JSON files.")
    parser.add_argument("--api-root", type=Path, default=Path("db_chatbot/api_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("db_chatbot/build_api_selected"))
    args = parser.parse_args()

    api = args.api_root
    brand_list_path = api / "brand_list_info/output/brand_list_info_selected.json"
    brand_list = load_json(brand_list_path)
    frcs = load_json(api / "brand_frcs_stats/output/brand_frcs_stats_selected.json")
    fntn = load_json(api / "brand_fntn_stats/output/brand_fntn_stats_selected.json")
    interior = load_json(api / "brand_interior_cost/output/brand_interior_cost_selected.json")
    mnno_to_id, name_to_id = build_brand_id_maps(brand_list_path, fallback_rows=brand_list)

    # brand_master
    master_by_id: dict[int, dict[str, Any]] = {}
    for r in brand_list:
        brand_id = to_int(r.get("brand_id")) or find_brand_id(r, mnno_to_id, name_to_id)
        if brand_id is None:
            continue
        rec = {
            "brand_id": brand_id,
            "brand_name": r.get("brandNm"),
            "company_name": r.get("corpNm"),
            "category_main": r.get("indutyLclasNm"),
            "category_sub": r.get("indutyMlsfcNm"),
            "franchise_start_date": r.get("jngBizStrtDate"),
        }
        prev = master_by_id.get(brand_id)
        if prev is None or non_empty_count(rec) >= non_empty_count(prev):
            master_by_id[brand_id] = rec
    brand_master = sorted(master_by_id.values(), key=lambda x: x["brand_id"])

    # brand_year_stats from frcs stats
    rows_by_brand: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in frcs:
        brand_id = to_int(r.get("brand_id")) or find_brand_id(r, mnno_to_id, name_to_id)
        year = to_int(r.get("yr"))
        if brand_id is None or year is None:
            continue
        store_count = to_int(r.get("frcsCnt")) or 0
        new_stores = to_int(r.get("newFrcsRgsCnt")) or 0
        closed_stores = to_int(r.get("ctrtCncltnCnt")) or 0
        rec = {
            "brand_id": brand_id,
            "year": year,
            "store_count": store_count,
            "new_stores": new_stores,
            "closed_stores": closed_stores,
            "avg_sales_krw": to_int(r.get("avrgSlsAmt")),
            "net_store_change": new_stores - closed_stores,
            "store_growth_rate": None,
            "closure_rate": (closed_stores / store_count) if store_count > 0 else 0.0,
            "churn_rate": ((new_stores + closed_stores) / store_count) if store_count > 0 else 0.0,
        }
        prev = rows_by_brand[brand_id].get(year)
        if prev is None or non_empty_count(rec) >= non_empty_count(prev):
            rows_by_brand[brand_id][year] = rec

    brand_year_stats: list[dict[str, Any]] = []
    for brand_id, year_map in rows_by_brand.items():
        years = sorted(year_map.keys())
        for i, year in enumerate(years):
            row = year_map[year]
            net = row["net_store_change"] or 0
            if i > 0:
                prev_store_count = float(year_map[years[i - 1]]["store_count"] or 0)
            else:
                prev_store_count = float((row["store_count"] or 0) - net)
            row["store_growth_rate"] = (net / prev_store_count) if prev_store_count > 0 else 0.0
            brand_year_stats.append(row)
    brand_year_stats.sort(key=lambda x: (x["brand_id"], x["year"]))

    # brand_store_types from interior
    store_types_by_brand: dict[int, dict[str, Any]] = {}
    for r in interior:
        brand_id = to_int(r.get("brand_id")) or find_brand_id(r, mnno_to_id, name_to_id)
        if brand_id is None:
            continue
        area = r.get("storCrtraAr")
        rec = {
            "brand_id": brand_id,
            "store_type": "Standard",
            "standard_area_pyeong": float(area) if area is not None else None,
        }
        prev = store_types_by_brand.get(brand_id)
        if prev is None or non_empty_count(rec) >= non_empty_count(prev):
            store_types_by_brand[brand_id] = rec
    brand_store_types = sorted(store_types_by_brand.values(), key=lambda x: (x["brand_id"], x["store_type"]))

    # brand_store_type_costs from fntn + interior
    interior_mid_by_brand_year: dict[tuple[int, int], int] = {}
    for r in interior:
        brand_id = to_int(r.get("brand_id")) or find_brand_id(r, mnno_to_id, name_to_id)
        year = to_int(r.get("jngBizCrtraYr"))
        if brand_id is None or year is None:
            continue
        mid = parse_range_mid(r.get("intrrAmtScopeVal"))
        if mid is not None:
            interior_mid_by_brand_year[(brand_id, year)] = mid

    cost_rows: list[dict[str, Any]] = []
    for r in fntn:
        brand_id = to_int(r.get("brand_id")) or find_brand_id(r, mnno_to_id, name_to_id)
        year = to_int(r.get("yr"))
        if brand_id is None or year is None:
            continue
        pairs = [
            ("initial_fee", to_int(r.get("jngBzmnJngAmt"))),
            ("education", to_int(r.get("jngBzmnEduAmt"))),
            ("other", to_int(r.get("jngBzmnEtcAmt"))),
            ("guarantee", to_int(r.get("jngBzmnAssrncAmt"))),
            ("total_initial_cost", to_int(r.get("smtnAmt"))),
        ]
        mid = interior_mid_by_brand_year.get((brand_id, year))
        if mid is not None:
            pairs.append(("interior", mid))
        for cat, amount in pairs:
            if amount is None:
                continue
            cost_rows.append(
                {
                    "brand_id": brand_id,
                    "year": year,
                    "store_type": "Standard",
                    "cost_category": cat,
                    "cost_amount_krw": amount,
                }
            )

    dedup_cost: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for row in cost_rows:
        key = (row["brand_id"], row["year"], row["store_type"], row["cost_category"])
        dedup_cost[key] = row
    brand_store_type_costs = sorted(
        dedup_cost.values(),
        key=lambda x: (x["brand_id"], x["year"], x["store_type"], x["cost_category"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "brand_master.json").write_text(
        json.dumps(brand_master, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "brand_year_stats.json").write_text(
        json.dumps(brand_year_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "brand_store_types.json").write_text(
        json.dumps(brand_store_types, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "brand_store_type_costs.json").write_text(
        json.dumps(brand_store_type_costs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "source": "api_data selected json",
        "row_counts": {
            "brand_master": len(brand_master),
            "brand_year_stats": len(brand_year_stats),
            "brand_store_types": len(brand_store_types),
            "brand_store_type_costs": len(brand_store_type_costs),
        },
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Wrote build:", args.output_dir)
    print("Rows:", report["row_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
