#!/usr/bin/env python3
"""Data access layer for normalized brand JSON tables."""

from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path
from typing import Any


def _normalize_brand_key(text: str) -> str:
    lowered = str(text).strip().lower()
    no_space = "".join(lowered.split())
    # Keep alnum + Hangul only for stable matching.
    return re.sub(r"[^0-9a-z가-힣]", "", no_space)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _format_int(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,}"


def _format_krw(value: int | None) -> str:
    if value is None:
        return "not disclosed in source data"
    return f"{value:,} KRW"


def _format_pct(decimal_ratio: float | None) -> str:
    if decimal_ratio is None:
        return "N/A"
    return f"{decimal_ratio * 100:.2f}%"


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class BrandResolutionError(ValueError):
    def __init__(
        self,
        query_text: str,
        status: str,
        candidates: list[dict[str, Any]] | None = None,
        reason: str | None = None,
    ) -> None:
        self.query_text = query_text
        self.status = status
        self.candidates = candidates or []
        self.reason = reason or ""
        if status == "ambiguous":
            cand_text = ", ".join(
                f"{c['brand_name']}({c['confidence']:.2f})" for c in self.candidates
            )
            msg = f"브랜드명 '{query_text}'이(가) 모호합니다. 다음 후보 중 선택해 주세요: {cand_text}"
        else:
            msg = f"브랜드 '{query_text}'을(를) 찾을 수 없습니다. 입력을 다시 확인해 주세요."
        super().__init__(msg)

    def to_payload(self) -> dict[str, Any]:
        return {
            "error_type": "brand_resolution",
            "resolution_status": self.status,
            "query_text": self.query_text,
            "candidates": self.candidates,
            "reason": self.reason,
            "error": str(self),
        }


class BrandDataStore:
    def __init__(self, build_dir: Path | str = Path("db_chatbot/build")) -> None:
        self.build_dir = Path(build_dir)
        self.brand_master = _load_json(self.build_dir / "brand_master.json")
        self.brand_year_stats = _load_json(self.build_dir / "brand_year_stats.json")
        self.brand_store_types = _load_json(self.build_dir / "brand_store_types.json")
        self.brand_store_type_costs = _load_json(self.build_dir / "brand_store_type_costs.json")

        self.master_by_id = {row["brand_id"]: row for row in self.brand_master}
        self.brand_ids_by_lower_name: dict[str, list[int]] = defaultdict(list)
        self.brand_ids_by_normalized_name: dict[str, list[int]] = defaultdict(list)
        self.brand_search_signals: dict[int, list[str]] = defaultdict(list)
        for row in self.brand_master:
            brand_id = row["brand_id"]
            brand_name = row["brand_name"]
            company_name = row.get("company_name") or ""

            self.brand_ids_by_lower_name[brand_name.lower()].append(brand_id)
            self.brand_ids_by_normalized_name[_normalize_brand_key(brand_name)].append(brand_id)

            # Search signals are generated from data only (no hardcoded alias map).
            signals = [brand_name]
            if company_name:
                signals.append(company_name)
            normalized_signals = []
            for signal in signals:
                norm = _normalize_brand_key(signal)
                if norm:
                    normalized_signals.append(norm)
            self.brand_search_signals[brand_id] = sorted(set(normalized_signals))

        self.year_stats_by_brand: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.brand_year_stats:
            self.year_stats_by_brand[row["brand_id"]].append(row)
        for brand_id, rows in self.year_stats_by_brand.items():
            self.year_stats_by_brand[brand_id] = sorted(rows, key=lambda r: r["year"])

        self.store_types_by_brand: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.brand_store_types:
            self.store_types_by_brand[row["brand_id"]].append(row)

        self.costs_by_brand_year_type: dict[tuple[int, int, str], dict[str, int]] = defaultdict(dict)
        for row in self.brand_store_type_costs:
            key = (row["brand_id"], row["year"], row["store_type"])
            self.costs_by_brand_year_type[key][row["cost_category"]] = row["cost_amount_krw"]

        # Resolver tuning knobs (override by env vars when needed).
        self.resolver_top_k = int(os.getenv("RESOLVER_TOP_K", "3"))
        self.resolver_high_confidence = float(os.getenv("RESOLVER_HIGH_CONF", "0.90"))
        self.resolver_high_margin = float(os.getenv("RESOLVER_HIGH_MARGIN", "0.08"))
        self.resolver_ambiguous_min = float(os.getenv("RESOLVER_AMBIG_MIN", "0.75"))
        self.resolver_llm_min = float(os.getenv("RESOLVER_LLM_MIN", "0.70"))

    def _llm_resolve_brand(self, brand_query: str) -> dict[str, Any] | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None

        model = os.getenv("RESOLVER_MODEL", "gpt-4.1-mini")
        candidates = [row["brand_name"] for row in self.brand_master]

        prompt = (
            "You are a brand name resolver.\n"
            "Given a user mention and candidate brand names, choose the single best match.\n"
            "Return strict JSON only with keys: brand_name, confidence, reason.\n"
            "If no reliable match, set brand_name to null.\n"
            f"user_mention: {brand_query}\n"
            f"candidates: {json.dumps(candidates, ensure_ascii=False)}"
        )
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(text)
            brand_name = parsed.get("brand_name")
            confidence = float(parsed.get("confidence", 0.0))
            reason = str(parsed.get("reason", "llm_fallback"))
            if not brand_name:
                return None
            target = next((r for r in self.brand_master if r["brand_name"] == brand_name), None)
            if not target:
                return None
            return {
                "brand_id": target["brand_id"],
                "brand_name": target["brand_name"],
                "confidence": max(0.0, min(confidence, 1.0)),
                "reason": reason,
            }
        except Exception:
            return None

    def _resolve_brand_candidate(self, brand_query: str, top_k: int = 3) -> dict[str, Any]:
        q = brand_query.strip()
        if not q:
            raise ValueError("브랜드명이 비어 있습니다.")

        q_norm = _normalize_brand_key(q)
        if not q_norm:
            return {
                "status": "not_found",
                "query_text": brand_query,
                "match": None,
                "candidates": [],
                "reason": "empty_after_normalization",
            }

        exact_ids = self.brand_ids_by_lower_name.get(q.lower(), [])
        if len(exact_ids) == 1:
            bid = exact_ids[0]
            return {
                "status": "resolved",
                "query_text": brand_query,
                "match": {"brand_id": bid, "brand_name": self.master_by_id[bid]["brand_name"], "confidence": 1.0},
                "candidates": [
                    {
                        "brand_id": bid,
                        "brand_name": self.master_by_id[bid]["brand_name"],
                        "confidence": 1.0,
                        "stage": "exact",
                    }
                ],
                "reason": "exact_match",
            }

        normalized_ids = self.brand_ids_by_normalized_name.get(q_norm, [])
        if len(normalized_ids) == 1:
            bid = normalized_ids[0]
            return {
                "status": "resolved",
                "query_text": brand_query,
                "match": {"brand_id": bid, "brand_name": self.master_by_id[bid]["brand_name"], "confidence": 0.98},
                "candidates": [
                    {
                        "brand_id": bid,
                        "brand_name": self.master_by_id[bid]["brand_name"],
                        "confidence": 0.98,
                        "stage": "normalized_exact",
                    }
                ],
                "reason": "normalized_exact_match",
            }

        scored = []
        for brand_id, signals in self.brand_search_signals.items():
            best_score = 0.0
            for signal in signals:
                if q_norm == signal:
                    score = 0.98
                elif q_norm in signal or signal in q_norm:
                    # Strong substring match from data-driven signals.
                    score = 0.93
                else:
                    score = _similarity(q_norm, signal)
                if score > best_score:
                    best_score = score
            scored.append((brand_id, best_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        effective_top_k = max(top_k or self.resolver_top_k, 2)
        top = scored[:effective_top_k]
        if not top:
            return {
                "status": "not_found",
                "query_text": brand_query,
                "match": None,
                "candidates": [],
                "reason": "no_candidates",
            }

        top1_id, top1_score = top[0]
        top2_score = top[1][1] if len(top) > 1 else 0.0
        raw_candidates = [
            {
                "brand_id": bid,
                "brand_name": self.master_by_id[bid]["brand_name"],
                "confidence": round(score, 4),
                "stage": "fuzzy_rank",
            }
            for bid, score in top[:effective_top_k]
        ]
        # Hide non-informative candidates when all scores are effectively zero.
        candidates = [c for c in raw_candidates if c["confidence"] > 0.01]

        if top1_score >= self.resolver_high_confidence and (top1_score - top2_score) >= self.resolver_high_margin:
            return {
                "status": "resolved",
                "query_text": brand_query,
                "match": {
                    "brand_id": top1_id,
                    "brand_name": self.master_by_id[top1_id]["brand_name"],
                    "confidence": round(top1_score, 4),
                },
                "candidates": candidates or raw_candidates[:1],
                "reason": "fuzzy_confident",
            }
        if top1_score >= self.resolver_ambiguous_min:
            return {
                "status": "ambiguous",
                "query_text": brand_query,
                "match": None,
                "candidates": candidates or raw_candidates,
                "reason": "fuzzy_ambiguous",
            }

        # Optional semantic fallback (no hardcoded alias): use LLM only when API key exists.
        llm_pick = self._llm_resolve_brand(brand_query)
        if llm_pick and llm_pick["confidence"] >= self.resolver_llm_min:
            return {
                "status": "resolved",
                "query_text": brand_query,
                "match": {
                    "brand_id": llm_pick["brand_id"],
                    "brand_name": llm_pick["brand_name"],
                    "confidence": round(llm_pick["confidence"], 4),
                },
                "candidates": [
                    {
                        "brand_id": llm_pick["brand_id"],
                        "brand_name": llm_pick["brand_name"],
                        "confidence": round(llm_pick["confidence"], 4),
                        "stage": "llm_fallback",
                    }
                ],
                "reason": f"llm_fallback:{llm_pick['reason']}",
            }
        return {
            "status": "not_found",
            "query_text": brand_query,
            "match": None,
            "candidates": candidates,
            "reason": "low_confidence",
        }

    def resolve_brand_debug(self, brand_query: str, top_k: int = 3) -> dict[str, Any]:
        """Return raw resolver decision payload for diagnostics/tuning."""
        return self._resolve_brand_candidate(brand_query=brand_query, top_k=top_k)

    def resolve_brand(self, brand_query: str) -> dict[str, Any]:
        res = self._resolve_brand_candidate(brand_query, top_k=3)
        if res["status"] == "resolved" and res["match"]:
            return self.master_by_id[res["match"]["brand_id"]]
        if res["status"] == "ambiguous":
            raise BrandResolutionError(
                query_text=brand_query,
                status="ambiguous",
                candidates=res.get("candidates", []),
                reason=res.get("reason"),
            )
        raise BrandResolutionError(
            query_text=brand_query,
            status="not_found",
            candidates=res.get("candidates", []),
            reason=res.get("reason"),
        )

    def _pick_year_row(self, rows: list[dict[str, Any]], year: int | None) -> dict[str, Any]:
        if not rows:
            raise ValueError("해당 브랜드의 연도별 통계 데이터가 없습니다.")

        if year is None:
            return max(rows, key=lambda r: r["year"])

        exact = [r for r in rows if r["year"] == year]
        if exact:
            return exact[0]

        prior = [r for r in rows if r["year"] < year]
        if prior:
            return max(prior, key=lambda r: r["year"])

        raise ValueError(f"요청한 연도({year}) 또는 그 이전 연도의 통계 데이터가 없습니다.")

    def _pick_common_year(self, brand_id_a: int, brand_id_b: int, year: int | None) -> int:
        years_a = {r["year"] for r in self.year_stats_by_brand.get(brand_id_a, [])}
        years_b = {r["year"] for r in self.year_stats_by_brand.get(brand_id_b, [])}
        common = sorted(years_a & years_b)
        if not common:
            raise ValueError("두 브랜드에 공통으로 존재하는 연도가 없습니다.")

        if year is None:
            return max(common)

        if year in common:
            return year

        prior = [y for y in common if y < year]
        if prior:
            return max(prior)

        raise ValueError(f"요청한 연도({year}) 또는 그 이전 공통 연도의 통계 데이터가 없습니다.")

    def _pick_exact_year_row(self, rows: list[dict[str, Any]], year: int) -> dict[str, Any]:
        for row in rows:
            if row["year"] == year:
                return row
        raise ValueError(f"{year}년의 정확한 통계 행이 없습니다.")

    def _select_cost_summary(
        self, brand_id: int, preferred_year: int, requested_store_type: str | None
    ) -> dict[str, Any]:
        candidate_keys = [k for k in self.costs_by_brand_year_type if k[0] == brand_id]
        if not candidate_keys:
            return {
                "year_used": None,
                "store_type_used": None,
                "cost_basis": "no_cost_data",
                "total_initial_cost_krw": None,
                "cost_breakdown_krw": {},
            }

        years = sorted({k[1] for k in candidate_keys})
        year_candidates = [y for y in years if y <= preferred_year] or years
        year_used = max(year_candidates)

        year_keys = [k for k in candidate_keys if k[1] == year_used]
        types = sorted({k[2] for k in year_keys})

        def totals_for_type(store_type: str) -> tuple[int | None, dict[str, int], str]:
            categories = self.costs_by_brand_year_type[(brand_id, year_used, store_type)]
            if "total_initial_cost" in categories:
                return categories["total_initial_cost"], categories, "reported_total_initial_cost"
            computed = sum(categories.values())
            return computed, categories, "computed_from_components"

        selected_type: str | None = None
        if requested_store_type:
            low = requested_store_type.lower()
            for st in types:
                if st.lower() == low:
                    selected_type = st
                    break
            if selected_type is None:
                return {
                    "year_used": year_used,
                    "store_type_used": None,
                    "available_store_types": types,
                    "cost_basis": "requested_store_type_not_found",
                    "total_initial_cost_krw": None,
                    "cost_breakdown_krw": {},
                }
        else:
            for st in types:
                if st.lower() == "standard":
                    selected_type = st
                    break
            if selected_type is None:
                best_type = None
                best_total = None
                best_basis = ""
                for st in types:
                    total, _, basis = totals_for_type(st)
                    if total is None:
                        continue
                    if best_total is None or total < best_total:
                        best_total = total
                        best_type = st
                        best_basis = basis
                if best_type is None:
                    return {
                        "year_used": year_used,
                        "store_type_used": None,
                        "cost_basis": "no_usable_cost_data",
                        "total_initial_cost_krw": None,
                        "cost_breakdown_krw": {},
                    }
                selected_type = best_type

        total, categories, basis = totals_for_type(selected_type)
        return {
            "year_used": year_used,
            "store_type_used": selected_type,
            "available_store_types": types,
            "cost_basis": basis,
            "total_initial_cost_krw": total,
            "cost_breakdown_krw": categories,
        }

    def get_brand_overview(
        self, brand_name: str, year: int | None = None, store_type: str | None = None
    ) -> dict[str, Any]:
        brand = self.resolve_brand(brand_name)
        brand_id = brand["brand_id"]

        year_row = self._pick_year_row(self.year_stats_by_brand.get(brand_id, []), year)
        year_used = year_row["year"]
        cost_summary = self._select_cost_summary(brand_id, preferred_year=year_used, requested_store_type=store_type)

        store_types = [
            {
                "store_type": row["store_type"],
                "standard_area_pyeong": row["standard_area_pyeong"],
            }
            for row in sorted(self.store_types_by_brand.get(brand_id, []), key=lambda r: r["store_type"])
        ]

        avg_sales = year_row.get("avg_sales_krw")
        result = {
            "brand": {
                "brand_id": brand_id,
                "brand_name": brand["brand_name"],
                "company_name": brand["company_name"],
                "franchise_start_date": brand["franchise_start_date"],
                "category_main": brand["category_main"],
                "category_sub": brand["category_sub"],
            },
            "year_used": year_used,
            "stats": {
                "store_count": year_row["store_count"],
                "new_stores": year_row["new_stores"],
                "closed_stores": year_row["closed_stores"],
                "net_store_change": year_row["net_store_change"],
                "store_growth_rate": year_row["store_growth_rate"],
                "closure_rate": year_row["closure_rate"],
                "churn_rate": year_row["churn_rate"],
                "avg_sales_krw": avg_sales,
            },
            "store_types": store_types,
            "startup_cost": cost_summary,
            "formatted": {
                "store_count": _format_int(year_row["store_count"]),
                "new_stores": _format_int(year_row["new_stores"]),
                "closed_stores": _format_int(year_row["closed_stores"]),
                "net_store_change": _format_int(year_row["net_store_change"]),
                "store_growth_rate": _format_pct(year_row["store_growth_rate"]),
                "closure_rate": _format_pct(year_row["closure_rate"]),
                "churn_rate": _format_pct(year_row["churn_rate"]),
                "avg_sales_krw": _format_krw(avg_sales),
                "startup_total_initial_cost_krw": _format_krw(cost_summary["total_initial_cost_krw"]),
            },
        }
        return result

    def get_brand_compare(
        self,
        brand_a_name: str,
        brand_b_name: str,
        year: int | None = None,
        store_type: str | None = None,
    ) -> dict[str, Any]:
        brand_a = self.resolve_brand(brand_a_name)
        brand_b = self.resolve_brand(brand_b_name)
        if brand_a["brand_id"] == brand_b["brand_id"]:
            raise ValueError("비교를 위해 서로 다른 두 브랜드를 입력해 주세요.")

        year_used = self._pick_common_year(brand_a["brand_id"], brand_b["brand_id"], year)
        row_a = self._pick_exact_year_row(self.year_stats_by_brand[brand_a["brand_id"]], year_used)
        row_b = self._pick_exact_year_row(self.year_stats_by_brand[brand_b["brand_id"]], year_used)

        cost_a = self._select_cost_summary(
            brand_id=brand_a["brand_id"], preferred_year=year_used, requested_store_type=store_type
        )
        cost_b = self._select_cost_summary(
            brand_id=brand_b["brand_id"], preferred_year=year_used, requested_store_type=store_type
        )

        def pack(brand: dict[str, Any], row: dict[str, Any], cost: dict[str, Any]) -> dict[str, Any]:
            avg_sales = row.get("avg_sales_krw")
            return {
                "brand": {
                    "brand_id": brand["brand_id"],
                    "brand_name": brand["brand_name"],
                    "company_name": brand["company_name"],
                },
                "year_used": year_used,
                "stats": {
                    "store_count": row["store_count"],
                    "new_stores": row["new_stores"],
                    "closed_stores": row["closed_stores"],
                    "net_store_change": row["net_store_change"],
                    "store_growth_rate": row["store_growth_rate"],
                    "closure_rate": row["closure_rate"],
                    "churn_rate": row["churn_rate"],
                    "avg_sales_krw": avg_sales,
                },
                "startup_cost": {
                    "year_used": cost["year_used"],
                    "store_type_used": cost["store_type_used"],
                    "cost_basis": cost["cost_basis"],
                    "total_initial_cost_krw": cost["total_initial_cost_krw"],
                },
                "formatted": {
                    "store_count": _format_int(row["store_count"]),
                    "new_stores": _format_int(row["new_stores"]),
                    "closed_stores": _format_int(row["closed_stores"]),
                    "net_store_change": _format_int(row["net_store_change"]),
                    "store_growth_rate": _format_pct(row["store_growth_rate"]),
                    "closure_rate": _format_pct(row["closure_rate"]),
                    "churn_rate": _format_pct(row["churn_rate"]),
                    "avg_sales_krw": _format_krw(avg_sales),
                    "startup_total_initial_cost_krw": _format_krw(cost["total_initial_cost_krw"]),
                },
            }

        cost_diff = None
        if (
            cost_a.get("total_initial_cost_krw") is not None
            and cost_b.get("total_initial_cost_krw") is not None
        ):
            cost_diff = cost_a["total_initial_cost_krw"] - cost_b["total_initial_cost_krw"]

        return {
            "comparison_year_used": year_used,
            "store_type_requested": store_type,
            "brand_a": pack(brand_a, row_a, cost_a),
            "brand_b": pack(brand_b, row_b, cost_b),
            "diff": {
                "store_count": row_a["store_count"] - row_b["store_count"],
                "new_stores": row_a["new_stores"] - row_b["new_stores"],
                "closed_stores": row_a["closed_stores"] - row_b["closed_stores"],
                "net_store_change": row_a["net_store_change"] - row_b["net_store_change"],
                "store_growth_rate": row_a["store_growth_rate"] - row_b["store_growth_rate"],
                "closure_rate": row_a["closure_rate"] - row_b["closure_rate"],
                "churn_rate": row_a["churn_rate"] - row_b["churn_rate"],
                "avg_sales_krw": (
                    None
                    if row_a.get("avg_sales_krw") is None or row_b.get("avg_sales_krw") is None
                    else row_a["avg_sales_krw"] - row_b["avg_sales_krw"]
                ),
                "startup_total_initial_cost_krw": cost_diff,
            },
            "diff_formatted": {
                "store_count": _format_int(row_a["store_count"] - row_b["store_count"]),
                "new_stores": _format_int(row_a["new_stores"] - row_b["new_stores"]),
                "closed_stores": _format_int(row_a["closed_stores"] - row_b["closed_stores"]),
                "net_store_change": _format_int(row_a["net_store_change"] - row_b["net_store_change"]),
                "store_growth_rate": _format_pct(row_a["store_growth_rate"] - row_b["store_growth_rate"]),
                "closure_rate": _format_pct(row_a["closure_rate"] - row_b["closure_rate"]),
                "churn_rate": _format_pct(row_a["churn_rate"] - row_b["churn_rate"]),
                "avg_sales_krw": _format_krw(
                    None
                    if row_a.get("avg_sales_krw") is None or row_b.get("avg_sales_krw") is None
                    else row_a["avg_sales_krw"] - row_b["avg_sales_krw"]
                ),
                "startup_total_initial_cost_krw": _format_krw(cost_diff),
            },
        }

    def _evaluate_condition(self, left: Any, op: str, right: Any) -> bool:
        if left is None:
            return False
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        raise ValueError(f"지원하지 않는 연산자입니다: '{op}'. 사용 가능: <, <=, >, >=, ==, !=")

    def _apply_sort_specs(
        self,
        items: list[dict[str, Any]],
        sort_specs: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        out = list(items)
        for spec in reversed(sort_specs):
            field = spec["field"]
            order = spec["order"]
            if order not in {"asc", "desc"}:
                raise ValueError(f"지원하지 않는 정렬 순서입니다: '{order}'. 'asc' 또는 'desc'를 사용하세요.")

            def key_fn(item: dict[str, Any]) -> tuple[bool, float]:
                val = item["metrics"].get(field)
                if val is None:
                    return (True, float("inf"))
                num = float(val)
                return (False, num if order == "asc" else -num)

            out.sort(key=key_fn)
        return out

    def get_brand_filter_search(
        self,
        conditions: list[dict[str, Any]],
        year: int | None = None,
        store_type: str | None = None,
        sort_by: str | None = None,
        sort_order: str = "asc",
        limit: int = 10,
    ) -> dict[str, Any]:
        supported_fields = {
            "store_count",
            "new_stores",
            "closed_stores",
            "net_store_change",
            "store_growth_rate",
            "closure_rate",
            "churn_rate",
            "avg_sales_krw",
            "startup_total_initial_cost_krw",
        }

        if limit <= 0:
            raise ValueError("limit 값은 1 이상이어야 합니다.")

        normalized_conditions = conditions or []
        for cond in normalized_conditions:
            field = cond.get("field")
            op = cond.get("op")
            if field not in supported_fields:
                raise ValueError(f"지원하지 않는 필터 필드입니다: '{field}'.")
            if op not in {"<", "<=", ">", ">=", "==", "!="}:
                raise ValueError(f"지원하지 않는 연산자입니다: '{op}'.")

        candidates: list[dict[str, Any]] = []
        for brand in self.brand_master:
            brand_id = brand["brand_id"]
            stats_rows = self.year_stats_by_brand.get(brand_id, [])
            if not stats_rows:
                continue

            year_row = self._pick_year_row(stats_rows, year)
            cost = self._select_cost_summary(
                brand_id=brand_id,
                preferred_year=year_row["year"],
                requested_store_type=store_type,
            )

            metrics = {
                "store_count": year_row["store_count"],
                "new_stores": year_row["new_stores"],
                "closed_stores": year_row["closed_stores"],
                "net_store_change": year_row["net_store_change"],
                "store_growth_rate": year_row["store_growth_rate"],
                "closure_rate": year_row["closure_rate"],
                "churn_rate": year_row["churn_rate"],
                "avg_sales_krw": year_row.get("avg_sales_krw"),
                "startup_total_initial_cost_krw": cost.get("total_initial_cost_krw"),
            }

            passed = True
            for cond in normalized_conditions:
                field = cond["field"]
                op = cond["op"]
                val = cond["value"]
                if not self._evaluate_condition(metrics.get(field), op, val):
                    passed = False
                    break
            if not passed:
                continue

            candidates.append(
                {
                    "brand": {
                        "brand_id": brand_id,
                        "brand_name": brand["brand_name"],
                        "company_name": brand["company_name"],
                    },
                    "year_used": year_row["year"],
                    "store_type_used_for_cost": cost.get("store_type_used"),
                    "metrics": metrics,
                    "formatted": {
                        "store_count": _format_int(metrics["store_count"]),
                        "new_stores": _format_int(metrics["new_stores"]),
                        "closed_stores": _format_int(metrics["closed_stores"]),
                        "net_store_change": _format_int(metrics["net_store_change"]),
                        "store_growth_rate": _format_pct(metrics["store_growth_rate"]),
                        "closure_rate": _format_pct(metrics["closure_rate"]),
                        "churn_rate": _format_pct(metrics["churn_rate"]),
                        "avg_sales_krw": _format_krw(metrics["avg_sales_krw"]),
                        "startup_total_initial_cost_krw": _format_krw(metrics["startup_total_initial_cost_krw"]),
                    },
                }
            )

        if sort_by:
            if sort_by not in supported_fields:
                raise ValueError(f"지원하지 않는 정렬 필드입니다: '{sort_by}'.")
            sort_specs = [{"field": sort_by, "order": sort_order}]
        else:
            sort_specs = [
                {"field": "churn_rate", "order": "asc"},
                {"field": "store_growth_rate", "order": "desc"},
                {"field": "store_count", "order": "desc"},
            ]

        sorted_candidates = self._apply_sort_specs(candidates, sort_specs)
        selected = sorted_candidates[:limit]

        return {
            "filters_applied": normalized_conditions,
            "year_requested": year,
            "store_type_requested": store_type,
            "sort_applied": sort_specs,
            "limit": limit,
            "total_matches": len(sorted_candidates),
            "results": selected,
        }

    def get_brand_trend(
        self,
        brand_name: str,
        start_year: int | None = None,
        end_year: int | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        supported_metrics = [
            "store_count",
            "new_stores",
            "closed_stores",
            "net_store_change",
            "store_growth_rate",
            "closure_rate",
            "churn_rate",
            "avg_sales_krw",
        ]
        metric_set = set(metrics) if metrics else set(supported_metrics)
        unsupported = [m for m in metric_set if m not in supported_metrics]
        if unsupported:
            raise ValueError(f"지원하지 않는 추이 지표입니다: {unsupported}")

        brand = self.resolve_brand(brand_name)
        rows = self.year_stats_by_brand.get(brand["brand_id"], [])
        if not rows:
            raise ValueError("해당 브랜드의 연도별 통계 데이터가 없습니다.")

        lo = start_year if start_year is not None else min(r["year"] for r in rows)
        hi = end_year if end_year is not None else max(r["year"] for r in rows)
        if lo > hi:
            raise ValueError("start_year는 end_year보다 작거나 같아야 합니다.")

        timeline_rows = [r for r in rows if lo <= r["year"] <= hi]
        if not timeline_rows:
            raise ValueError("요청한 연도 구간에 데이터가 없습니다.")

        timeline = []
        for row in timeline_rows:
            point_raw = {"year": row["year"]}
            point_fmt = {"year": row["year"]}
            for m in metric_set:
                val = row.get(m)
                point_raw[m] = val
                if m in {"store_growth_rate", "closure_rate", "churn_rate"}:
                    point_fmt[m] = _format_pct(val)
                elif m == "avg_sales_krw":
                    point_fmt[m] = _format_krw(val)
                else:
                    point_fmt[m] = _format_int(val)
            timeline.append({"raw": point_raw, "formatted": point_fmt})

        first = timeline_rows[0]
        last = timeline_rows[-1]
        summary: dict[str, Any] = {}
        for m in metric_set:
            start_val = first.get(m)
            end_val = last.get(m)
            delta = None if start_val is None or end_val is None else end_val - start_val
            if delta is None:
                trend = "insufficient_data"
            elif delta > 0:
                trend = "up"
            elif delta < 0:
                trend = "down"
            else:
                trend = "flat"
            if m in {"store_growth_rate", "closure_rate", "churn_rate"}:
                fmt_start = _format_pct(start_val)
                fmt_end = _format_pct(end_val)
                fmt_delta = _format_pct(delta) if delta is not None else "N/A"
            elif m == "avg_sales_krw":
                fmt_start = _format_krw(start_val)
                fmt_end = _format_krw(end_val)
                fmt_delta = _format_krw(delta) if delta is not None else "N/A"
            else:
                fmt_start = _format_int(start_val)
                fmt_end = _format_int(end_val)
                fmt_delta = _format_int(delta) if delta is not None else "N/A"

            summary[m] = {
                "start_year": first["year"],
                "end_year": last["year"],
                "start_value": start_val,
                "end_value": end_val,
                "delta": delta,
                "trend": trend,
                "formatted": {
                    "start_value": fmt_start,
                    "end_value": fmt_end,
                    "delta": fmt_delta,
                },
            }

        return {
            "brand": {
                "brand_id": brand["brand_id"],
                "brand_name": brand["brand_name"],
                "company_name": brand["company_name"],
            },
            "range": {
                "start_year_requested": start_year,
                "end_year_requested": end_year,
                "start_year_used": timeline_rows[0]["year"],
                "end_year_used": timeline_rows[-1]["year"],
            },
            "metrics_used": sorted(metric_set),
            "timeline": timeline,
            "summary": summary,
        }
