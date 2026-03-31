# DB Chatbot

Korean franchise-brand chatbot backed by local FTC API data.

Active runtime status:

- Active source data: `db_chatbot/api_data/*_selected.json`
- Active normalized build: `db_chatbot/build_api_selected/`
- Archived and not part of normal runtime: `archive/legacy/`

The current project centers on `api_data/*_selected.json` and the normalized build in
`build_api_selected/`. The older `api_sync` flow mentioned in previous docs is not the
active path in this repo.

## Project Layout

- `api_data/`
  FTC fetch scripts plus per-table outputs such as `raw.json`, `selected.json`, and `xlsx`.
- `api_data_multiyear/`
  Staging area for collecting historical years before appending deduped rows.
- `build_api_selected/`
  Normalized JSON tables built from `api_data/*_selected.json`.
- `testing/`
  Active eval scripts. Case files live in `testing/cases/` and generated outputs live in `testing/artifacts/`.
- `data_access.py`
  Core data layer. `BrandDataStore` loads and joins brand data.
- `tools.py`
  Tool definitions used by the chat app.
- `chat_app.py`
  LangChain/OpenAI chat entrypoint.

## Data Flow

1. Fetch FTC API data into `api_data/<table>/output/`.
2. Keep `*_selected.json` as the main local source artifact.
3. Optionally collect additional years into `api_data_multiyear/` and append deduped rows.
4. Optionally build normalized tables into `build_api_selected/`.
5. Run the chatbot against either:
   - `api_selected` mode: reads `api_data/*_selected.json` directly
   - `build` mode: reads normalized JSON tables

## Setup

```bash
cd /Users/yerincho/Desktop/26/WinWin/AI
python3 -m venv .venv
source .venv/bin/activate
pip install -U pyyaml langchain langchain-openai pydantic openpyxl
```

Create `db_chatbot/.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
FTC_SERVICE_KEY=your_ftc_service_key_here
```

## Fetch FTC Data

Run the table-specific fetchers directly. Examples:

```bash
python db_chatbot/api_data/brand_list_info/fetch_brand_list_info.py --year 2024
python db_chatbot/api_data/brand_frcs_stats/fetch_brand_frcs_stats.py --year 2024
python db_chatbot/api_data/brand_fntn_stats/fetch_brand_fntn_stats.py --year 2024
python db_chatbot/api_data/brand_brand_stats/fetch_brand_brand_stats.py --year 2024
python db_chatbot/api_data/brand_interior_cost/fetch_brand_interior_cost.py --year 2024
```

Outputs are written under each table’s `output/` directory.

## Collect Multi-Year Data

Collect historical years into staging without overwriting current outputs:

```bash
python db_chatbot/api_data/multiyear/collect_existing_brands_multiyear.py --years 2021 2022 2023
```

Other useful scripts:

- `api_data/multiyear/extend_brand_list_years_existing_brands.py`
- `api_data/multiyear/extend_other_tables_years_existing_brands.py`
- `api_data/multiyear/append_staging_into_existing_selected.py`

## Build Normalized JSON

Build normalized tables from the selected API JSON files:

```bash
python db_chatbot/scripts/build_from_api_selected_json.py
```

This writes:

- `db_chatbot/build_api_selected/brand_master.json`
- `db_chatbot/build_api_selected/brand_year_stats.json`
- `db_chatbot/build_api_selected/brand_store_types.json`
- `db_chatbot/build_api_selected/brand_store_type_costs.json`

There is also an older Excel-contract pipeline, but it is archived under `archive/legacy/` and is not part of the normal chatbot runtime:

```bash
python archive/legacy/db_chatbot/scripts/build_contract_workbook_from_api_excels.py
python archive/legacy/db_chatbot/scripts/load_validate_data.py
```

That path writes to the archived `archive/legacy/db_chatbot/build/`, but the active repo workflow is the API-selected JSON path above.

## Run Chatbot

Directly from selected API data:

```bash
python db_chatbot/chat_app.py --source-mode api_selected --query "교촌치킨 요약해줘"
```

From normalized build tables:

```bash
python db_chatbot/chat_app.py --source-mode build --build-dir db_chatbot/build_api_selected --query "교촌치킨이랑 BBQ 비교해줘"
```

Agent modes:

```bash
python db_chatbot/chat_app.py --agent-mode openai_api --query "교촌치킨 요약해줘"
python db_chatbot/chat_app.py --agent-mode simple --query "교촌치킨이랑 BBQ 비교해줘"
python db_chatbot/chat_app.py --agent-mode advanced --max-tool-rounds 5 --query "교촌치킨이랑 BBQ 비교하고 최근 추이도 같이 보여줘"
```

- `openai_api`: direct model answer without tools
- `simple`: one-shot tool use, then final answer
- `advanced`: iterative tool loop until the model stops calling tools or the round cap is reached

Interactive mode:

```bash
python db_chatbot/chat_app.py --source-mode api_selected
```

## Supported Tool Paths

- `brand_overview`
- `brand_compare`
- `brand_filter_search`
- `brand_trend`
- `brand_fallback_lookup`

## Testing

Resolver utilities:

```bash
python db_chatbot/testing/resolver_debug.py --query "비비큐"
python db_chatbot/testing/resolver_calibrate.py --query "교촌" --query "BBQ"
```

Top-level deterministic eval helpers:

```bash
python testing/run_store_eval.py
python testing/run_agent_mode_benchmark.py
python testing/run_agent_mode_eval.py
```

Case sets currently include:

- `testing/cases/overview_cases.json`
- `testing/cases/compare_cases.json`
- `testing/cases/filter_cases.json`
- `testing/cases/trend_cases.json`
- `testing/cases/resolver_cases.json`
- `testing/cases/agent_mode_cases.json`

Generated benchmark outputs, reports, and visuals are written under `testing/artifacts/`.
