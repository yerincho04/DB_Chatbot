# Chicken Brand Chatbot Scope (V1)

## Goal
Define exactly what the chatbot can answer from `data/brand_data.xlsx` before building LangChain logic.

## Data Available (from Excel)
- `brand_master`: brand identity (`brand_name`, `corp_name`, `start_date`, category)
- `brand_year_stats`: yearly performance (`store_count`, `new_store_count`, `closed_store_count`, `avg_sales_amt`, `net_store_change`, `store_growth_rate`, `closure_rate`, `churn_rate`)
- `brand_store_types`: store type list per brand (`store_type`, `standard_area_pyeong`)
- `brand_store_type_costs`: startup cost breakdown by brand/year/type (`cost_cateogry`, `amount`)

## V1 Supported Questions
1. Brand Overview
- Example: "Tell me about Puradak."
- Returns:
  - Brand/corporation basic info
  - Latest year store count
  - Latest year growth/churn/closure rate
  - Latest available average sales
  - Store types
  - Startup cost summary (latest year, by store type)

2. Brand Comparison (2 brands)
- Example: "Compare BBQ and Kyochon."
- Returns side-by-side values for selected year (default: latest common year):
  - Store count
  - New/closed/net change
  - Growth/closure/churn rate
  - Average sales
  - Representative startup total cost (if store type specified, compare that type; otherwise compare each brand's `Standard` type if present)

3. Conditional Brand Search (filters)
- Example: "Find brands with startup cost under 80,000,000 KRW and churn below 5%."
- Supports filters on:
  - `year`
  - `store_count`
  - `new_store_count`
  - `closed_store_count`
  - `net_store_change`
  - `store_growth_rate`
  - `closure_rate`
  - `churn_rate`
  - `avg_sales_amt`
  - `total_initial_cost` (from cost table)
- Output:
  - Matching brands
  - Key values that satisfied filters
  - Sort option (e.g., lowest churn first)

4. Trend Questions (single brand over time)
- Example: "How has NeNe's churn changed from 2022 to 2024?"
- Returns year-by-year table and trend direction for:
  - store count
  - growth/closure/churn rates
  - net change
  - average sales

5. Cost Breakdown Questions
- Example: "Show startup cost breakdown for BBQ 올리브치킨 in 2024."
- Returns:
  - Cost categories (`initial_fee`, `interior`, `equipment`, `other`, etc.)
  - Total initial cost if present
  - Currency-formatted amounts

## V1 Response Rules
- Default year: latest year available for requested metric.
- If `avg_sales_amt` is `NULL` for a year, answer "not disclosed in source data."
- Rates should be shown as percentages with 1-2 decimals.
- Numeric money values should be shown with KRW comma format.
- If the user mentions a brand alias/case variation, normalize to known `brand_name`.

## V1 Out of Scope
- Forecasting ("predict 2025 stores")
- Causal reasoning not in data ("why did churn rise?")
- Advice/recommendation ("which is best to invest in?")
- Multi-brand ranking with weighted scoring (save for V2)
- Joining external sources beyond this Excel

## Canonical Intents for LangChain Router
- `brand_overview`
- `brand_compare`
- `brand_filter_search`
- `brand_trend`
- `brand_cost_breakdown`
- `data_help` (schema/definitions questions)

## Suggested V1 Acceptance Checks
1. Ask one overview query for each brand in file.
2. Ask 5 compare queries with mixed Korean/English brand names.
3. Ask 5 filter queries with thresholds and year constraints.
4. Ask 3 trend queries with explicit year ranges.
5. Ask 3 cost breakdown queries including missing fields.
6. Verify chatbot clearly reports missing data instead of hallucinating.

## Next Step (after scope approval)
Create a data contract file in `db_chatbot` that maps Excel columns to normalized internal field names and data types.
