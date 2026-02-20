#!/usr/bin/env python3
"""Load and validate brand data from XLSX using the project data contract.

Outputs normalized JSON tables and a validation report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
    raise SystemExit(
        "Missing dependency 'PyYAML' for this interpreter.\n"
        "Install with:\n"
        "  /opt/homebrew/bin/python3.14 -m pip install pyyaml\n"
        "Or run this script with the interpreter that already has PyYAML."
    ) from exc


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"a": NS_MAIN}


def col_to_index(col: str) -> int:
    value = 0
    for ch in col:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value


def parse_cell_ref(cell_ref: str) -> tuple[int, int]:
    match = re.match(r"^([A-Z]+)(\d+)$", cell_ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    col = col_to_index(match.group(1))
    row = int(match.group(2))
    return row, col


def excel_serial_to_iso(serial: float) -> str:
    # Excel date serial with 1900 date system (including leap-year bug offset).
    base = date(1899, 12, 30)
    dt = base + timedelta(days=float(serial))
    return dt.isoformat()


def parse_sheet_xml(xlsx_path: Path, sheet_name: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels}

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("a:si", NS):
                parts = [t.text or "" for t in si.iter(f"{{{NS_MAIN}}}t")]
                shared_strings.append("".join(parts))

        sheet_target = None
        for sheet in workbook.findall("a:sheets/a:sheet", NS):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib.get(f"{{{NS_REL}}}id")
                sheet_target = rel_map.get(rel_id)
                break
        if not sheet_target:
            raise ValueError(f"Sheet not found in workbook: {sheet_name}")

        sheet_path = f"xl/{sheet_target}" if not sheet_target.startswith("xl/") else sheet_target
        sheet_xml = ET.fromstring(zf.read(sheet_path))

        grid: dict[int, dict[int, Any]] = defaultdict(dict)
        for c in sheet_xml.findall(".//a:sheetData/a:row/a:c", NS):
            ref = c.attrib.get("r")
            if not ref:
                continue
            row_idx, col_idx = parse_cell_ref(ref)
            cell_type = c.attrib.get("t")

            value: Any = None
            v_node = c.find("a:v", NS)
            if v_node is not None:
                raw = v_node.text or ""
                if cell_type == "s":
                    value = shared_strings[int(raw)] if raw.isdigit() else raw
                elif cell_type == "b":
                    value = "TRUE" if raw == "1" else "FALSE"
                else:
                    value = raw
            else:
                inline_node = c.find("a:is", NS)
                if inline_node is not None:
                    value = "".join(t.text or "" for t in inline_node.iter(f"{{{NS_MAIN}}}t"))
            grid[row_idx][col_idx] = value

        if 1 not in grid:
            return []

        header_row = grid[1]
        max_col = max(header_row.keys()) if header_row else 0
        col_headers: dict[int, str] = {}
        for col in range(1, max_col + 1):
            header_val = header_row.get(col)
            header = (str(header_val).strip() if header_val is not None else "")
            if header:
                col_headers[col] = header

        rows: list[dict[str, Any]] = []
        for row_idx in sorted(k for k in grid.keys() if k > 1):
            row_data = grid[row_idx]
            row_obj = {h: row_data.get(c) for c, h in col_headers.items()}
            if any(v is not None and str(v).strip() != "" for v in row_obj.values()):
                rows.append(row_obj)
        return rows


def try_parse_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    txt = str(value).strip().replace(",", "")
    return float(txt)


def cast_value(raw: Any, col_def: dict[str, Any], missing_tokens: set[str]) -> Any:
    if raw is None:
        return None

    txt = str(raw).strip()
    if txt in missing_tokens:
        return None
    if txt == "":
        return None

    dtype = col_def["type"]
    if dtype == "string":
        return txt
    if dtype == "int":
        number = try_parse_number(txt)
        if not math.isclose(number, round(number), rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Expected integer, got {txt}")
        return int(round(number))
    if dtype == "float":
        return float(try_parse_number(txt))
    if dtype == "date":
        # Expected Excel serial in source file.
        try:
            return excel_serial_to_iso(try_parse_number(txt))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid date value: {txt}") from exc
    raise ValueError(f"Unsupported type: {dtype}")


@dataclass
class ValidationResult:
    tables: dict[str, list[dict[str, Any]]]
    errors: list[str]
    warnings: list[str]


def resolve_source_path(contract_path: Path, source_ref: str) -> Path:
    src = Path(source_ref)
    candidates = [
        src,
        contract_path.parent / source_ref,
        contract_path.parent.parent / source_ref,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve source file '{source_ref}'. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def normalize_table_rows(
    table_name: str,
    rows: list[dict[str, Any]],
    table_def: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    missing_tokens = set(contract["normalization"]["missing_tokens"])
    columns = table_def["columns"]

    normalized_rows = []
    for idx, row in enumerate(rows, start=2):
        out: dict[str, Any] = {}
        row_has_error = False
        for col_def in columns:
            src = col_def["source"]
            canonical = col_def["canonical"]
            nullable = bool(col_def["nullable"])
            raw_val = row.get(src)
            try:
                casted = cast_value(raw_val, col_def, missing_tokens)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{table_name}: row {idx}, field '{canonical}' cast error: {exc}")
                row_has_error = True
                continue

            if casted is None and not nullable:
                errors.append(f"{table_name}: row {idx}, field '{canonical}' is required but missing")
                row_has_error = True
            out[canonical] = casted

        if not row_has_error:
            normalized_rows.append(out)
    return normalized_rows, errors


def apply_validations(contract: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    hard = contract["validation"]["hard_fail"]
    soft = contract["validation"]["soft_warn"]

    year_min = int(hard["year_range"]["min"])
    year_max = int(hard["year_range"]["max"])
    non_negative_fields = set(hard["non_negative_fields"])
    rate_bounds = hard["rate_bounds"]
    required_keys = hard["required_keys"]

    deduped_tables: dict[str, list[dict[str, Any]]] = {}
    for table_name, rows in tables.items():
        table_def = contract["tables"][table_name]
        pk = table_def["primary_key"]
        dedup_map: dict[tuple[Any, ...], dict[str, Any]] = {}

        for i, row in enumerate(rows, start=2):
            row_has_error = False

            for req in required_keys.get(table_name, []):
                if row.get(req) is None:
                    errors.append(f"{table_name}: row {i}, missing required key '{req}'")
                    row_has_error = True

            year = row.get("year")
            if year is not None and not (year_min <= year <= year_max):
                errors.append(f"{table_name}: row {i}, year out of range: {year}")
                row_has_error = True

            for field in non_negative_fields:
                if field in row and row[field] is not None and row[field] < 0:
                    errors.append(f"{table_name}: row {i}, field '{field}' must be non-negative")
                    row_has_error = True

            for rate_field, bounds in rate_bounds.items():
                if rate_field in row and row[rate_field] is not None:
                    val = row[rate_field]
                    if "min" in bounds and val < bounds["min"]:
                        errors.append(f"{table_name}: row {i}, field '{rate_field}' below minimum {bounds['min']}")
                        row_has_error = True
                    if "max" in bounds and val > bounds["max"]:
                        errors.append(f"{table_name}: row {i}, field '{rate_field}' above maximum {bounds['max']}")
                        row_has_error = True

            if row_has_error:
                continue

            key = tuple(row.get(k) for k in pk)
            if any(k is None for k in key):
                errors.append(f"{table_name}: row {i}, primary key contains null: {pk}")
                continue
            if key in dedup_map and hard["duplicate_policy"]["warn"]:
                warnings.append(f"{table_name}: duplicate primary key {key}, keeping last occurrence")
            dedup_map[key] = row

        deduped_tables[table_name] = list(dedup_map.values())

    upper = float(soft["suspicious_rate_upper_bound"]["value"])
    for table_name, rows in deduped_tables.items():
        for row in rows:
            for rate_field in soft["suspicious_rate_upper_bound"]["applies_to"]:
                if rate_field in row and row[rate_field] is not None and row[rate_field] > upper:
                    warnings.append(
                        f"{table_name}: suspiciously high rate {rate_field}={row[rate_field]} (>{upper})"
                    )

    # Soft consistency check for year stats table.
    for row in deduped_tables.get("brand_year_stats", []):
        lhs = row["new_stores"] - row["closed_stores"]
        rhs = row["net_store_change"]
        if lhs != rhs:
            warnings.append(
                "brand_year_stats: net_store_change mismatch for "
                f"brand_id={row['brand_id']}, year={row['year']} ({lhs} != {rhs})"
            )

    # Soft consistency for total initial cost.
    grouped: dict[tuple[int, int, str], dict[str, int]] = defaultdict(dict)
    for row in deduped_tables.get("brand_store_type_costs", []):
        gk = (row["brand_id"], row["year"], row["store_type"])
        grouped[gk][row["cost_category"]] = row["cost_amount_krw"]
    for gk, costs in grouped.items():
        if "total_initial_cost" in costs:
            components_sum = sum(v for k, v in costs.items() if k != "total_initial_cost")
            total = costs["total_initial_cost"]
            if components_sum != 0 and total != components_sum:
                warnings.append(
                    "brand_store_type_costs: total_initial_cost mismatch for "
                    f"brand_id={gk[0]}, year={gk[1]}, store_type={gk[2]} "
                    f"(total={total}, components_sum={components_sum})"
                )

    return ValidationResult(tables=deduped_tables, errors=errors, warnings=warnings)


def load_contract(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and validate brand data using contract YAML.")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("db_chatbot/data_contract_v1.yaml"),
        help="Path to contract YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("db_chatbot/build"),
        help="Directory for normalized outputs and validation report.",
    )
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = load_contract(contract_path)
    source_path = resolve_source_path(contract_path, contract["source"]["file"])

    raw_tables: dict[str, list[dict[str, Any]]] = {}
    normalized_tables: dict[str, list[dict[str, Any]]] = {}
    normalize_errors: list[str] = []

    for table_name, table_def in contract["tables"].items():
        source_table = table_def["source_table"]
        rows = parse_sheet_xml(source_path, source_table)
        raw_tables[table_name] = rows
        normalized_rows, errs = normalize_table_rows(table_name, rows, table_def, contract)
        normalized_tables[table_name] = normalized_rows
        normalize_errors.extend(errs)

    result = apply_validations(contract, normalized_tables)
    result.errors = normalize_errors + result.errors

    ensure_dir(args.output_dir)
    for table_name, rows in result.tables.items():
        out = args.output_dir / f"{table_name}.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "contract": str(contract_path),
        "source_file": str(source_path),
        "row_counts": {k: len(v) for k, v in result.tables.items()},
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "errors": result.errors,
        "warnings": result.warnings,
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Wrote normalized tables and validation report to:", args.output_dir)
    print("Rows:", report["row_counts"])
    print("Errors:", report["error_count"])
    print("Warnings:", report["warning_count"])
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
