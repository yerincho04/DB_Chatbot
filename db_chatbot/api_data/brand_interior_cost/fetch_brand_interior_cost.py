#!/usr/bin/env python3
"""Fetch FTC API: brand interior cost info, save raw JSON, and export selected fields."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
from urllib.error import HTTPError
from urllib.parse import urlencode, unquote
from urllib.request import Request, urlopen

sys.path.append(str(Path(__file__).resolve().parents[1]))
from brand_id_utils import build_brand_id_maps, find_brand_id


API_URL_CANDIDATES = [
    "https://apis.data.go.kr/1130000/FftcBrandFrcsIntInfo2_Service/getbrandFrcsBzmnIntrrctinfo",
    "https://apis.data.go.kr/1130000/FftcBrandFrcsIntInfo2_Service/getBrandFrcsBzmnIntrrctinfo",
    "https://apis.data.go.kr/1130000/FftcBrandFrcsIntInfo2_Service/getBrandFrcsBzmnIntrrctInfo",
]

SELECT_FIELDS = [
    "crrncyUnitCdNm",
    "jngBizCrtraYr",
    "brandMnno",
    "jnghdqrtrsMnno",
    "brandNm",
    "indutyLclasNm",
    "indutyMlsfcNm",
    "unitArIntrrAmtScopeVal",
    "storCrtraAr",
    "intrrAmtScopeVal",
]


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


def resolve_service_key(explicit_key: str | None) -> str:
    if explicit_key:
        return explicit_key.strip()
    key = os.getenv("FTC_SERVICE_KEY") or os.getenv("serviceKey")
    if not key:
        raise RuntimeError("Missing FTC_SERVICE_KEY. Set it in env or pass --service-key.")
    return key.strip()


def key_candidates(raw_key: str) -> list[str]:
    out: list[str] = []
    for candidate in (raw_key.strip(), unquote(raw_key.strip())):
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def fetch_json(api_url: str, params: dict[str, str]) -> dict[str, Any]:
    query = urlencode(params)
    req = Request(
        f"{api_url}?{query}",
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json,*/*"},
    )
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        msg = f"HTTP {exc.code} {exc.reason}"
        if body:
            msg += f" | body={body[:800]}"
        raise RuntimeError(msg) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response: {raw[:300]}") from exc


def fetch_xml(api_url: str, params: dict[str, str]) -> str:
    query = urlencode(params)
    req = Request(
        f"{api_url}?{query}",
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/xml,text/xml,*/*"},
    )
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        msg = f"HTTP {exc.code} {exc.reason}"
        if body:
            msg += f" | body={body[:800]}"
        raise RuntimeError(msg) from exc


def extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []

    if isinstance(payload.get("response"), dict):
        body = payload["response"].get("body")
        if isinstance(body, dict):
            items = body.get("items")
            if isinstance(items, dict):
                candidates.append(items.get("item"))
            candidates.append(items)

    if isinstance(payload.get("body"), dict):
        items = payload["body"].get("items")
        if isinstance(items, dict):
            candidates.append(items.get("item"))
        candidates.append(items)

    items = payload.get("items")
    if isinstance(items, dict):
        candidates.append(items.get("item"))
    candidates.append(items)

    for c in candidates:
        if isinstance(c, list):
            return [row for row in c if isinstance(row, dict)]
        if isinstance(c, dict):
            return [c]
    return []


def extract_items_from_xml(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        row: dict[str, Any] = {}
        for child in list(item):
            row[child.tag] = (child.text or "").strip()
        if row:
            rows.append(row)
    return rows


def select_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: row.get(k) for k in SELECT_FIELDS} for row in rows]


def write_excel(
    rows: list[dict[str, Any]],
    xlsx_path: Path,
    brand_mnno_to_id: dict[str, int],
    brand_nm_to_id: dict[str, int],
) -> None:
    try:
        from openpyxl import Workbook
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Missing dependency: openpyxl. Install with `pip install openpyxl`.") from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "brand_interior_cost"
    ws.append(["brand_id", *SELECT_FIELDS])
    for row in rows:
        ws.append([find_brand_id(row, brand_mnno_to_id, brand_nm_to_id), *[row.get(col) for col in SELECT_FIELDS]])
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)


def fetch_for_brand(
    service_key: str,
    year: int,
    page_no: int,
    num_rows: int,
    result_type: str,
    brand_mnno: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, str | None]:
    payload: dict[str, Any] | None = None
    xml_text: str | None = None
    used_api_url: str | None = None
    used_type_param: str | None = None
    last_error: Exception | None = None

    for api_url in API_URL_CANDIDATES:
        stop_all = False
        for key_value in key_candidates(service_key):
            for type_name, type_param in (
                ("resultType", {"resultType": result_type}),
                ("_type", {"_type": result_type}),
                ("type", {"type": result_type}),
                ("none", {}),
            ):
                params = {
                    "serviceKey": key_value,
                    "pageNo": str(page_no),
                    "numOfRows": str(num_rows),
                    "jngBizCrtraYr": str(year),
                    "brandMnno": brand_mnno,
                }
                params.update(type_param)
                try:
                    payload = fetch_json(api_url, params)
                    used_api_url = api_url
                    used_type_param = type_name
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if "http 401" in str(exc).lower():
                        stop_all = True
                        break
            if payload is None and not stop_all:
                xml_params = {
                    "serviceKey": key_value,
                    "pageNo": str(page_no),
                    "numOfRows": str(num_rows),
                    "jngBizCrtraYr": str(year),
                    "brandMnno": brand_mnno,
                    "resultType": "xml",
                }
                try:
                    xml_text = fetch_xml(api_url, xml_params)
                    used_api_url = api_url
                    used_type_param = "resultType(xml)"
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if "http 401" in str(exc).lower():
                        stop_all = True
            if payload is not None or xml_text is not None or stop_all:
                break
        if payload is not None or xml_text is not None or stop_all:
            break

    if payload is None and xml_text is None:
        return [], {"error": str(last_error) if last_error else "request failed"}, None, None

    items = extract_items(payload) if payload is not None else extract_items_from_xml(xml_text or "")
    raw_obj: dict[str, Any] | None
    if payload is not None:
        raw_obj = payload
    else:
        raw_obj = {"raw_xml": xml_text or ""}
    meta = {"used_api_url": used_api_url, "used_type_param": used_type_param}
    return items, meta, raw_obj, None


def load_brand_mnno_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        v = str(row.get("brandMnno") or "").strip()
        if v and v not in out:
            out.append(v)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Brand Interior Cost API and export selected columns.")
    parser.add_argument("--service-key", default=None)
    parser.add_argument("--year", type=int, default=2024, help="jngBizCrtraYr")
    parser.add_argument("--brand-mnno", default=None, help="Single brandMnno. If omitted, bulk mode is used.")
    parser.add_argument(
        "--brand-list-json",
        type=Path,
        default=Path("db_chatbot/api_data/brand_list_info/output/brand_list_info_selected.json"),
        help="Used in bulk mode: JSON list containing brandMnno values.",
    )
    parser.add_argument("--page-no", type=int, default=1)
    parser.add_argument("--num-rows", type=int, default=100)
    parser.add_argument("--result-type", default="json")
    parser.add_argument("--output-dir", type=Path, default=Path("db_chatbot/api_data/brand_interior_cost/output"))
    args = parser.parse_args()

    load_env_file(Path(".env"))
    load_env_file(Path("db_chatbot/.env"))
    load_env_file(Path("db_chatbot/api_data/.env"))

    service_key = resolve_service_key(args.service_key)
    if args.brand_mnno:
        brand_ids = [str(args.brand_mnno).strip()]
    else:
        if not args.brand_list_json.exists():
            raise RuntimeError(
                f"Bulk mode needs brand list JSON. Not found: {args.brand_list_json}"
            )
        brand_ids = load_brand_mnno_list(args.brand_list_json)
        if not brand_ids:
            raise RuntimeError(f"No brandMnno found in {args.brand_list_json}")

    all_items: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    raw_by_brand: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    meta_by_brand: list[dict[str, Any]] = []

    for idx, brand_id in enumerate(brand_ids, start=1):
        items, meta, raw_obj, error = fetch_for_brand(
            service_key=service_key,
            year=args.year,
            page_no=args.page_no,
            num_rows=args.num_rows,
            result_type=args.result_type,
            brand_mnno=brand_id,
        )
        if error:
            errors.append({"brandMnno": brand_id, "error": error})
            print(f"[{idx}/{len(brand_ids)}] {brand_id}: error={error}")
            continue

        selected = select_columns(items)
        all_items.extend(items)
        all_selected.extend(selected)
        meta_by_brand.append({"brandMnno": brand_id, **meta, "row_count": len(items)})
        raw_by_brand.append({"brandMnno": brand_id, "raw": raw_obj})
        print(f"[{idx}/{len(brand_ids)}] {brand_id}: rows={len(items)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "brand_interior_cost_raw.json"
    selected_path = args.output_dir / "brand_interior_cost_selected.json"
    xlsx_path = args.output_dir / "brand_interior_cost.xlsx"
    meta_path = args.output_dir / "brand_interior_cost_meta.json"
    errors_path = args.output_dir / "brand_interior_cost_errors.json"

    raw_path.write_text(json.dumps(raw_by_brand, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_path.write_text(json.dumps(all_selected, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta_by_brand, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    brand_mnno_to_id, brand_nm_to_id = build_brand_id_maps(
        brand_list_json=args.brand_list_json,
        fallback_rows=all_selected,
    )
    write_excel(all_selected, xlsx_path, brand_mnno_to_id, brand_nm_to_id)

    print(f"Brands requested: {len(brand_ids)}")
    print(f"Rows fetched: {len(all_items)}")
    print(f"Selected rows: {len(all_selected)}")
    print(f"Brand errors: {len(errors)}")
    print(f"Raw JSON: {raw_path}")
    print(f"Selected JSON: {selected_path}")
    print(f"Meta JSON: {meta_path}")
    print(f"Errors JSON: {errors_path}")
    print(f"Excel: {xlsx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
