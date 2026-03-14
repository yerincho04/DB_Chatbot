#!/usr/bin/env python3
"""Minimal LangChain chat app for brand overview queries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from data_access import BrandDataStore, BrandResolutionError
from tools import (
    create_brand_compare_tool,
    create_brand_fallback_lookup_tool,
    create_brand_filter_search_tool,
    create_brand_overview_tool,
    create_brand_trend_tool,
)


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


def run_once(
    query: str,
    model: str = "gpt-4.1-mini",
    build_dir: Path | str = Path("db_chatbot/build"),
    source_mode: str = "api_selected",
    api_data_root: Path | str = Path("db_chatbot/api_data"),
) -> str:
    load_env_file(Path("db_chatbot/.env"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in db_chatbot/.env or environment.")

    store = BrandDataStore(
        build_dir=Path(build_dir),
        source_mode=source_mode,
        api_data_root=Path(api_data_root),
    )
    overview_tool = create_brand_overview_tool(store)
    compare_tool = create_brand_compare_tool(store)
    filter_search_tool = create_brand_filter_search_tool(store)
    trend_tool = create_brand_trend_tool(store)
    fallback_tool = create_brand_fallback_lookup_tool(store)
    tools_by_name = {
        overview_tool.name: overview_tool,
        compare_tool.name: compare_tool,
        filter_search_tool.name: filter_search_tool,
        trend_tool.name: trend_tool,
        fallback_tool.name: fallback_tool,
    }

    llm = ChatOpenAI(model=model, temperature=0).bind_tools(list(tools_by_name.values()))

    system = SystemMessage(
        content=(
            "당신은 데이터 기반 프랜차이즈 브랜드 분석 어시스턴트입니다. "
            "최종 답변은 반드시 한국어로 작성하세요. "
            "도구 결과를 근거로만 답변하고, 데이터가 없으면 추정하지 말고 "
            "'원천 데이터에 없음'이라고 명확히 말하세요. "
            "가능하면 기준 연도와 사용한 매장 유형을 함께 표시하세요. "
            "조건 검색 요청은 brand_filter_search 도구를 사용하세요. "
            "연도별 추이 요청은 brand_trend 도구를 사용하세요. "
            "다른 도구에 명확히 맞지 않거나 브랜드명이 모호하면 brand_fallback_lookup 도구를 사용하세요. "
            "도구 결과에 error가 있으면 오류 내용을 한국어로 풀어 설명하고 "
            "사용자가 다시 시도할 입력 예시를 1개 제시하세요. "
            "error_type이 brand_resolution이고 resolution_status가 ambiguous이면 "
            "후보 브랜드를 번호 목록으로 보여주고, 하나를 선택해 달라고 안내하세요."
        )
    )
    messages = [system, HumanMessage(content=query)]
    ai_msg = llm.invoke(messages)

    if ai_msg.tool_calls:
        messages.append(ai_msg)
        for call in ai_msg.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                continue
            try:
                tool_result = tool.invoke(call["args"])
            except BrandResolutionError as exc:
                payload = exc.to_payload()
                payload["tool_name"] = call["name"]
                payload["input_args"] = call["args"]
                tool_result = payload
            except Exception as exc:  # noqa: BLE001
                tool_result = {
                    "error": str(exc),
                    "tool_name": call["name"],
                    "input_args": call["args"],
                }
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_call_id=call["id"],
                )
            )
        final_msg = llm.invoke(messages)
        return str(final_msg.content)

    # Deterministic fallback: if model did not call any tool, invoke fallback tool explicitly.
    fallback_result = fallback_tool.invoke({"query": query, "top_k": 5, "include_overview": True})
    fallback_prompt = HumanMessage(
        content=(
            "아래는 사용자 질의에 대한 fallback 브랜드 조회 결과(JSON)입니다.\n"
            f"{json.dumps(fallback_result, ensure_ascii=False)}\n\n"
            "이 데이터를 근거로 한국어로 답변하세요. "
            "브랜드가 모호하면 후보를 번호 목록으로 제시하고 선택을 요청하세요. "
            "데이터가 없으면 원천 데이터에 없다고 명시하세요."
        )
    )
    final_msg = llm.invoke([system, HumanMessage(content=query), fallback_prompt])
    return str(final_msg.content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the franchise brand chatbot.")
    parser.add_argument("--query", type=str, default=None, help="Single prompt to run.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model name.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("db_chatbot/build"),
        help="Directory containing normalized build JSON tables.",
    )
    parser.add_argument(
        "--source-mode",
        type=str,
        choices=["api_selected", "build"],
        default="api_selected",
        help="Data source mode. 'api_selected' reads api_data/*_selected.json directly.",
    )
    parser.add_argument(
        "--api-data-root",
        type=Path,
        default=Path("db_chatbot/api_data"),
        help="Root directory containing API selected JSON outputs.",
    )
    args = parser.parse_args()

    if args.query:
        print(
            run_once(
                args.query,
                model=args.model,
                build_dir=args.build_dir,
                source_mode=args.source_mode,
                api_data_root=args.api_data_root,
            )
        )
        return 0

    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        user = input("> ").strip()
        if user.lower() in {"exit", "quit"}:
            return 0
        if not user:
            continue
        try:
            print(
                run_once(
                    user,
                    model=args.model,
                    build_dir=args.build_dir,
                    source_mode=args.source_mode,
                    api_data_root=args.api_data_root,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
