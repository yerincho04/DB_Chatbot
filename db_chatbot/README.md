# DB Chatbot (Korean Chicken Brand QA)

LangChain-based chatbot that answers chicken franchise questions in Korean using local data from `brand_data.xlsx`.

## Quick Start

1. Create and activate virtual environment:

```bash
cd /Users/yerincho/Desktop/26/WinWin/AI
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -U pyyaml langchain langchain-openai pydantic
```

3. Set API key in `db_chatbot/.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Build Normalized Data

```bash
python db_chatbot/scripts/load_validate_data.py
```

## Run Chatbot

- Single query:

```bash
python db_chatbot/chat_app.py --query "교촌이랑 비비큐 비교해줘."
```

- Interactive mode:

```bash
python db_chatbot/chat_app.py
```

## Supported Intents (V1)

- `brand_overview`: 단일 브랜드 요약
- `brand_compare`: 두 브랜드 비교
- `brand_filter_search`: 조건 검색
- `brand_trend`: 연도별 추이 분석

## Testing Utilities

```bash
python db_chatbot/testing/resolver_debug.py --query "비비큐"
python db_chatbot/testing/resolver_calibrate.py --query "교촌" --query "비비큐"
```
