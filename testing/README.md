# Testing Assets

This folder contains deterministic smoke evals and generated evaluation artifacts for the current `db_chatbot` project.

These cases are intentionally scoped to the repo's supported data contract:

- `brand_overview`
- `brand_compare`
- `brand_filter_search`
- `brand_trend`
- brand resolver diagnostics

The older imported files were removed because they tested unsupported domains such as legal violations, financing policy, regional franchise counts, and other text-heavy disclosure content that this project does not currently model.

## Layout

- `cases/`
- `artifacts/benchmarks/`
- `artifacts/reports/`
- `artifacts/visuals/`
- `run_store_eval.py`
- `run_agent_mode_benchmark.py`
- `run_agent_mode_eval.py`
- `run_llm_accuracy_eval.py`

## Active Case Files

- `cases/overview_cases.json`
- `cases/compare_cases.json`
- `cases/filter_cases.json`
- `cases/trend_cases.json`
- `cases/resolver_cases.json`
- `run_store_eval.py`
- `cases/agent_mode_cases.json`
- `run_agent_mode_benchmark.py`
- `run_agent_mode_eval.py`
- `cases/llm_accuracy_cases.json`
- `run_llm_accuracy_eval.py`

Generated outputs and examples live under `artifacts/`.

Notable generated files:

- `artifacts/benchmarks/agent_mode_benchmark_output.json`
- `artifacts/benchmarks/llm_accuracy_benchmark_output.json`
- `artifacts/reports/llm_accuracy_eval_output.json`
- `artifacts/visuals/model_response_examples.md`
- `artifacts/visuals/mode_comparison_chart.png`

## Run

```bash
python testing/run_store_eval.py
```

You can also run one file at a time:

```bash
python testing/run_store_eval.py --cases testing/cases/resolver_cases.json
```

## Agent Benchmark

Run the same prompt set across:

- `db_chatbot/chat_app_openai_api.py`
- `db_chatbot/chat_app.py`
- `db_chatbot/chat_app_advanced.py`

```bash
python testing/run_agent_mode_benchmark.py --cases testing/cases/agent_mode_cases.json --out testing/artifacts/benchmarks/agent_mode_benchmark_output.json
```

The current lightweight heuristic scorer is:

```bash
python testing/run_agent_mode_eval.py --cases testing/cases/agent_mode_cases.json --input testing/artifacts/benchmarks/agent_mode_benchmark_output.json
```

## LLM Accuracy Eval

For expected-answer-based accuracy, create or edit `testing/cases/llm_accuracy_cases.json`.

Each item should contain:

- `index`
- `difficulty`
- `question`
- `expected_response`
- optional `reference_points`
- optional `grading_notes`

Then run:

```bash
python testing/run_agent_mode_benchmark.py --cases testing/cases/llm_accuracy_cases.json --out testing/artifacts/benchmarks/llm_accuracy_benchmark_output.json
python testing/run_llm_accuracy_eval.py --cases testing/cases/llm_accuracy_cases.json --input testing/artifacts/benchmarks/llm_accuracy_benchmark_output.json --out testing/artifacts/reports/llm_accuracy_eval_output.json
```

The LLM judge reports:

- `is_correct`
- `score`
- `accuracy` by mode
- average score by mode
- per-case rationales, missing points, and unsupported claims
