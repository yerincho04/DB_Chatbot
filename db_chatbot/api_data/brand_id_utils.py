#!/usr/bin/env python3
"""Utilities for stable brand_id mapping across API tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _rows_from_path(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def build_brand_id_maps(
    brand_list_json: Path,
    fallback_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (brand_mnno_to_id, brand_nm_to_id) with 1-based incremental IDs."""
    rows = _rows_from_path(brand_list_json)
    if not rows and fallback_rows:
        rows = [row for row in fallback_rows if isinstance(row, dict)]

    brand_mnno_to_id: dict[str, int] = {}
    brand_nm_to_id: dict[str, int] = {}

    next_id = 1
    for row in rows:
        mnno = str(row.get("brandMnno") or "").strip()
        name = str(row.get("brandNm") or "").strip()

        existing_id: int | None = None
        if mnno and mnno in brand_mnno_to_id:
            existing_id = brand_mnno_to_id[mnno]
        elif name and name in brand_nm_to_id:
            existing_id = brand_nm_to_id[name]

        if existing_id is None:
            existing_id = next_id
            next_id += 1

        if mnno and mnno not in brand_mnno_to_id:
            brand_mnno_to_id[mnno] = existing_id
        if name and name not in brand_nm_to_id:
            brand_nm_to_id[name] = existing_id

    return brand_mnno_to_id, brand_nm_to_id


def find_brand_id(
    row: dict[str, Any],
    brand_mnno_to_id: dict[str, int],
    brand_nm_to_id: dict[str, int],
) -> int | None:
    mnno = str(row.get("brandMnno") or "").strip()
    if mnno and mnno in brand_mnno_to_id:
        return brand_mnno_to_id[mnno]

    name = str(row.get("brandNm") or "").strip()
    if name and name in brand_nm_to_id:
        return brand_nm_to_id[name]

    return None
