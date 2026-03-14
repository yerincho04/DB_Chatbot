#!/usr/bin/env python3
"""LangChain tools for brand-data chatbot."""

from __future__ import annotations

from pydantic import BaseModel, Field

from data_access import BrandDataStore


class BrandOverviewInput(BaseModel):
    brand_name: str = Field(description="Brand name to look up, e.g., BBQ or Kyochon.")
    year: int | None = Field(default=None, description="Optional target year, e.g., 2024.")
    store_type: str | None = Field(
        default=None,
        description="Optional store type for startup cost lookup, e.g., Standard.",
    )


class BrandCompareInput(BaseModel):
    brand_a: str = Field(description="First brand name to compare.")
    brand_b: str = Field(description="Second brand name to compare.")
    year: int | None = Field(default=None, description="Optional comparison year.")
    store_type: str | None = Field(
        default=None,
        description="Optional store type for startup cost comparison.",
    )


class FilterCondition(BaseModel):
    field: str = Field(
        description=(
            "Filter field. One of: store_count, new_stores, closed_stores, "
            "net_store_change, store_growth_rate, closure_rate, churn_rate, "
            "avg_sales_krw, startup_total_initial_cost_krw, contract_end_count, "
            "contract_cancel_count, name_change_count, new_store_registrations, "
            "avg_sales_per_area_krw, avg_sales_total_krw, startup_deposit_krw, "
            "startup_training_krw, startup_other_krw, startup_guarantee_krw, "
            "startup_sum_krw, executives_count, employees_count, interior_store_area, "
            "interior_cost_mid_krw, interior_cost_per_area_mid_krw"
        )
    )
    op: str = Field(description="Operator: <, <=, >, >=, ==, !=")
    value: float = Field(description="Numeric threshold value.")


class BrandFilterSearchInput(BaseModel):
    conditions: list[FilterCondition] = Field(
        default_factory=list,
        description="List of filter conditions to apply with AND semantics.",
    )
    year: int | None = Field(default=None, description="Optional reference year.")
    store_type: str | None = Field(
        default=None,
        description="Optional store type for startup cost metric.",
    )
    sort_by: str | None = Field(
        default=None,
        description="Optional sort field. Defaults to churn_rate, growth, store_count order.",
    )
    sort_order: str = Field(default="asc", description="Sort order: asc or desc.")
    limit: int = Field(default=10, description="Maximum number of results.")


class BrandTrendInput(BaseModel):
    brand_name: str = Field(description="Brand name for trend analysis.")
    start_year: int | None = Field(default=None, description="Optional start year.")
    end_year: int | None = Field(default=None, description="Optional end year.")
    metrics: list[str] | None = Field(
        default=None,
        description=(
            "Optional metric list. Supported: store_count, new_stores, closed_stores, "
            "net_store_change, store_growth_rate, closure_rate, churn_rate, avg_sales_krw, "
            "contract_end_count, contract_cancel_count, name_change_count, "
            "new_store_registrations, avg_sales_per_area_krw, avg_sales_total_krw, "
            "startup_deposit_krw, startup_training_krw, startup_other_krw, "
            "startup_guarantee_krw, startup_sum_krw, executives_count, employees_count, "
            "interior_store_area, interior_cost_mid_krw, interior_cost_per_area_mid_krw"
        ),
    )


class BrandFallbackLookupInput(BaseModel):
    query: str = Field(description="Original user query text.")
    top_k: int = Field(default=5, description="Max candidate brands to include.")
    include_overview: bool = Field(
        default=True,
        description="Whether to include compact brand overviews for candidates.",
    )


def create_brand_overview_tool(store: BrandDataStore):
    from langchain_core.tools import StructuredTool

    def brand_overview(brand_name: str, year: int | None = None, store_type: str | None = None):
        """Return a grounded overview for one franchise brand from local dataset."""
        return store.get_brand_overview(brand_name=brand_name, year=year, store_type=store_type)

    return StructuredTool.from_function(
        func=brand_overview,
        name="brand_overview",
        description=(
            "Use this to answer single-brand overview questions about store count, "
            "growth/churn, average sales, store types, startup cost, organization, "
            "franchise operations, and interior/funding context."
        ),
        args_schema=BrandOverviewInput,
    )


def create_brand_compare_tool(store: BrandDataStore):
    from langchain_core.tools import StructuredTool

    def brand_compare(
        brand_a: str,
        brand_b: str,
        year: int | None = None,
        store_type: str | None = None,
    ):
        """Return a grounded side-by-side comparison for two franchise brands."""
        return store.get_brand_compare(
            brand_a_name=brand_a,
            brand_b_name=brand_b,
            year=year,
            store_type=store_type,
        )

    return StructuredTool.from_function(
        func=brand_compare,
        name="brand_compare",
        description=(
            "Use this to compare two brands side-by-side for store counts, growth/churn, "
            "average sales, startup cost, operations/funding, and organization metrics."
        ),
        args_schema=BrandCompareInput,
    )


def create_brand_filter_search_tool(store: BrandDataStore):
    from langchain_core.tools import StructuredTool

    def brand_filter_search(
        conditions: list[dict] | list[FilterCondition],
        year: int | None = None,
        store_type: str | None = None,
        sort_by: str | None = None,
        sort_order: str = "asc",
        limit: int = 10,
    ):
        """Return brands matching numeric filter conditions."""
        normalized_conditions = []
        for cond in conditions:
            if hasattr(cond, "model_dump"):
                normalized_conditions.append(cond.model_dump())
            else:
                normalized_conditions.append(cond)
        return store.get_brand_filter_search(
            conditions=normalized_conditions,
            year=year,
            store_type=store_type,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
        )

    return StructuredTool.from_function(
        func=brand_filter_search,
        name="brand_filter_search",
        description=(
            "Use this to find brands that satisfy numeric conditions like churn rate, "
            "store count, average sales, startup cost, contract churn, and staffing/funding."
        ),
        args_schema=BrandFilterSearchInput,
    )


def create_brand_trend_tool(store: BrandDataStore):
    from langchain_core.tools import StructuredTool

    def brand_trend(
        brand_name: str,
        start_year: int | None = None,
        end_year: int | None = None,
        metrics: list[str] | None = None,
    ):
        """Return year-by-year trend for a single brand."""
        return store.get_brand_trend(
            brand_name=brand_name,
            start_year=start_year,
            end_year=end_year,
            metrics=metrics,
        )

    return StructuredTool.from_function(
        func=brand_trend,
        name="brand_trend",
        description=(
            "Use this to answer single-brand trend questions over years, including "
            "store counts, growth/churn/closure rates, average sales, contract stats, "
            "funding, staffing, and interior-cost indicators."
        ),
        args_schema=BrandTrendInput,
    )


def create_brand_fallback_lookup_tool(store: BrandDataStore):
    from langchain_core.tools import StructuredTool

    def brand_fallback_lookup(query: str, top_k: int = 5, include_overview: bool = True):
        """Fallback brand lookup tool for unmatched or ambiguous user queries."""
        top_k = max(1, min(int(top_k), 10))
        decision = store.resolve_brand_debug(query, top_k=top_k)
        candidates = (decision.get("candidates") or [])[:top_k]

        candidate_overviews: list[dict] = []
        if include_overview:
            seen_names: set[str] = set()
            for c in candidates:
                name = str(c.get("brand_name") or "").strip()
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                try:
                    ov = store.get_brand_overview(name)
                    candidate_overviews.append(
                        {
                            "brand_name": name,
                            "year_used": ov.get("year_used"),
                            "note": ov.get("note"),
                            "stats": ov.get("stats"),
                            "startup_cost": ov.get("startup_cost"),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    candidate_overviews.append({"brand_name": name, "error": str(exc)})

        return {
            "query": query,
            "resolution_status": decision.get("status"),
            "reason": decision.get("reason"),
            "match": decision.get("match"),
            "candidates": candidates,
            "candidate_overviews": candidate_overviews,
        }

    return StructuredTool.from_function(
        func=brand_fallback_lookup,
        name="brand_fallback_lookup",
        description=(
            "Fallback tool. Use when user query does not fit overview/compare/filter/trend, "
            "or when brand resolution is ambiguous/not_found. Returns lookup candidates and "
            "compact brand data context for final answer."
        ),
        args_schema=BrandFallbackLookupInput,
    )
