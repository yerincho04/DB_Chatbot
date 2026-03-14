#!/usr/bin/env python3
"""Fetch one FTC API (brand list info), save raw JSON, and export selected fields to Excel."""

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
    "https://apis.data.go.kr/1130000/FftcBrandRlssInfo2_Service/getBrandinfo",
    "https://apis.data.go.kr/1130000/FftcBrandRlssInfo2_Service/getBrandInfo",
    "https://apis.data.go.kr/1130000/FftcBrandRlsInfo2_Service/getBrandinfo",
    "https://apis.data.go.kr/1130000/FftcBrandRlsInfo2_Service/getBrandInfo",
]
SELECT_FIELDS = [
    "jngBizCrtraYr",
    "brandMnno",
    "jnghdqrtrsMnno",
    "brno",
    "crno",
    "jnghdqrtrsRprsvNm",
    "brandNm",
    "indutyLclasNm",
    "indutyMlsfcNm",
    "majrGdsNm",
    "jngBizStrtDate",
    "corpNm",
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
    url = f"{api_url}?{query}"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,*/*",
        },
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
    url = f"{api_url}?{query}"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/xml,text/xml,*/*",
        },
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
    # Supports both common Data.go.kr wrappers and direct body.
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
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({k: row.get(k) for k in SELECT_FIELDS})
    return out


def type_params_from_name(name: str, result_type: str) -> dict[str, str]:
    if name == "resultType":
        return {"resultType": result_type}
    if name == "_type":
        return {"_type": result_type}
    if name == "type":
        return {"type": result_type}
    if name == "resultType(xml)":
        return {"resultType": "xml"}
    return {}


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
    ws.title = "brand_list_info"
    ws.append(["brand_id", *SELECT_FIELDS])
    for row in rows:
        ws.append([find_brand_id(row, brand_mnno_to_id, brand_nm_to_id), *[row.get(col) for col in SELECT_FIELDS]])

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Brand List Info API and export selected columns.")
    parser.add_argument("--service-key", default=None)
    parser.add_argument("--year", type=int, default=2024, help="jngBizCrtraYr")
    parser.add_argument("--page-no", type=int, default=1)
    parser.add_argument("--num-rows", type=int, default=100)
    parser.add_argument("--result-type", default="json")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Maximum number of pages to collect when --all-pages is enabled.",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        default=True,
        help="Collect all pages using the working request pattern.",
    )
    parser.add_argument(
        "--single-page",
        action="store_true",
        help="Only collect one page (disables --all-pages behavior).",
    )
    parser.add_argument(
        "--year-fallback-window",
        type=int,
        default=2,
        help="Retry older years on 500 Unexpected errors (e.g. 2 => year, year-1, year-2).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("db_chatbot/api_data/brand_list_info/output"))
    args = parser.parse_args()

    load_env_file(Path(".env"))
    load_env_file(Path("db_chatbot/.env"))
    load_env_file(Path("db_chatbot/api_data/.env"))

    service_key = resolve_service_key(args.service_key)
    payload: dict[str, Any] | None = None
    xml_text: str | None = None
    used_year: int | None = None
    last_error: Exception | None = None
    used_api_url: str | None = None

    used_num_rows: int | None = None
    used_type_param: str | None = None
    used_key_value: str | None = None

    row_candidates: list[int] = []
    for n in (args.num_rows, 100, 50, 20, 10, 1):
        if n > 0 and n not in row_candidates:
            row_candidates.append(n)

    for year in [args.year - i for i in range(max(0, args.year_fallback_window) + 1)]:
        stop_all = False
        for api_url in API_URL_CANDIDATES:
            for key_value in key_candidates(service_key):
                for num_rows in row_candidates:
                    for type_name, type_param in (
                        ("resultType", {"resultType": args.result_type}),
                        ("_type", {"_type": args.result_type}),
                        ("type", {"type": args.result_type}),
                        ("none", {}),
                    ):
                        params = {
                            "serviceKey": key_value,
                            "pageNo": str(args.page_no),
                            "numOfRows": str(num_rows),
                            "jngBizCrtraYr": str(year),
                        }
                        params.update(type_param)
                        try:
                            payload = fetch_json(api_url, params)
                            used_api_url = api_url
                            used_year = year
                            used_num_rows = num_rows
                            used_type_param = type_name
                            used_key_value = key_value
                            break
                        except Exception as exc:  # noqa: BLE001
                            last_error = exc
                            msg = str(exc).lower()
                            if "http 401" in msg:
                                stop_all = True
                                break
                            continue
                    if payload is None and not stop_all:
                        xml_params = {
                            "serviceKey": key_value,
                            "pageNo": str(args.page_no),
                            "numOfRows": str(num_rows),
                            "jngBizCrtraYr": str(year),
                            "resultType": "xml",
                        }
                        try:
                            xml_text = fetch_xml(api_url, xml_params)
                            used_api_url = api_url
                            used_year = year
                            used_num_rows = num_rows
                            used_type_param = "resultType(xml)"
                            used_key_value = key_value
                        except Exception as exc:  # noqa: BLE001
                            last_error = exc
                            msg = str(exc).lower()
                            if "http 401" in msg:
                                stop_all = True
                    if payload is not None or stop_all:
                        break
                    if xml_text is not None:
                        break
                if payload is not None or xml_text is not None or stop_all:
                    break
            if payload is not None or xml_text is not None or stop_all:
                break
        if payload is not None or xml_text is not None or stop_all:
            break
        if last_error is not None:
            msg = str(last_error).lower()
            if not ("http 500" in msg and "unexpected errors" in msg):
                break

    if payload is None and xml_text is None:
        raise RuntimeError(str(last_error) if last_error else "request failed")

    items = extract_items(payload) if payload is not None else extract_items_from_xml(xml_text or "")
    all_items = list(items)
    raw_pages: list[Any] = [payload if payload is not None else {"raw_xml": xml_text or ""}]

    collect_all = args.all_pages and not args.single_page
    if collect_all and used_api_url and used_key_value and used_num_rows and used_year is not None:
        for page in range(args.page_no + 1, args.max_pages + 1):
            params = {
                "serviceKey": used_key_value,
                "pageNo": str(page),
                "numOfRows": str(used_num_rows),
                "jngBizCrtraYr": str(used_year),
            }
            params.update(type_params_from_name(used_type_param or "none", args.result_type))
            try:
                if used_type_param == "resultType(xml)":
                    page_xml = fetch_xml(used_api_url, params)
                    page_rows = extract_items_from_xml(page_xml)
                    raw_pages.append({"raw_xml": page_xml})
                else:
                    page_payload = fetch_json(used_api_url, params)
                    page_rows = extract_items(page_payload)
                    raw_pages.append(page_payload)
            except Exception:
                break

            if not page_rows:
                break
            all_items.extend(page_rows)
            if len(page_rows) < used_num_rows:
                break

    items = all_items
    selected_rows = select_columns(items)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "brand_list_info_raw.json"
    selected_path = args.output_dir / "brand_list_info_selected.json"
    xlsx_path = args.output_dir / "brand_list_info.xlsx"

    raw_path.write_text(json.dumps(raw_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_path.write_text(json.dumps(selected_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    brand_mnno_to_id, brand_nm_to_id = build_brand_id_maps(
        brand_list_json=selected_path,
        fallback_rows=selected_rows,
    )
    write_excel(selected_rows, xlsx_path, brand_mnno_to_id, brand_nm_to_id)

    print(f"Rows fetched: {len(items)}")
    print(f"Selected rows: {len(selected_rows)}")
    if used_year is not None:
        print(f"Used year: {used_year}")
    if used_num_rows is not None:
        print(f"Used numOfRows: {used_num_rows}")
    print(f"Pages collected: {len(raw_pages)}")
    if used_type_param is not None:
        print(f"Used type param: {used_type_param}")
    if used_api_url is not None:
        print(f"Used API URL: {used_api_url}")
    print(f"Raw JSON: {raw_path}")
    print(f"Selected JSON: {selected_path}")
    print(f"Excel: {xlsx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
