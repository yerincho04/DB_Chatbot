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
            "avg_sales_krw, startup_total_initial_cost_krw"
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
            "net_store_change, store_growth_rate, closure_rate, churn_rate, avg_sales_krw"
        ),
    )


def create_brand_overview_tool(store: BrandDataStore):
    from langchain_core.tools import StructuredTool

    def brand_overview(brand_name: str, year: int | None = None, store_type: str | None = None):
        """Return a grounded overview for one chicken brand from local dataset."""
        return store.get_brand_overview(brand_name=brand_name, year=year, store_type=store_type)

    return StructuredTool.from_function(
        func=brand_overview,
        name="brand_overview",
        description=(
            "Use this to answer single-brand overview questions about store count, "
            "growth/churn, average sales, store types, and startup cost."
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
        """Return a grounded side-by-side comparison for two chicken brands."""
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
            "average sales, and startup cost."
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
            "store count, average sales, and startup cost."
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
            "store counts, growth/churn/closure rates, and average sales."
        ),
        args_schema=BrandTrendInput,
    )
