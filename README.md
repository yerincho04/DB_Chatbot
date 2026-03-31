# AI Repo

This repository is organized around the active `db_chatbot` workflow and a separated archive for older assets.

## Active

- `db_chatbot/`
  - Main application code
  - Active data source: `api_data/*_selected.json`
  - Active normalized build: `build_api_selected/`
- `testing/cases/`
  - Active evaluation inputs
- `testing/artifacts/`
  - Generated benchmark outputs, reports, and visuals

## Archived

- `archive/legacy/`
  - Older Excel-contract pipeline
  - Archived workbook/PDF assets
  - Old `db_chatbot/build/` output

## Common Entry Points

- Chat app: `python db_chatbot/chat_app.py --query "..."`
- Deterministic eval: `python testing/run_store_eval.py`
- Agent benchmark: `python testing/run_agent_mode_benchmark.py`
- LLM accuracy eval: `python testing/run_llm_accuracy_eval.py`

For details:
- `db_chatbot/README.md`
- `testing/README.md`
- `archive/legacy/README.md`
