#!/usr/bin/env python3
"""Iterative tool-calling chatbot with planning and self-check."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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


def build_tools(store: BrandDataStore) -> tuple[dict[str, object], object]:
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
    return tools_by_name, fallback_tool


def build_system_prompt() -> SystemMessage:
    return SystemMessage(
        content=(
            "당신은 데이터 기반 프랜차이즈 브랜드 분석 어시스턴트입니다. "
            "최종 답변은 반드시 한국어로 작성하세요. "
            "도구 결과를 근거로만 답변하고, 데이터가 없으면 추정하지 말고 "
            "'원천 데이터에 없음'이라고 명확히 말하세요. "
            "가능하면 기준 연도와 사용한 매장 유형을 함께 표시하세요. "
            "사용자 질문에 여러 요구사항이 섞여 있으면 각 요구사항을 분해해서 체크하세요. "
            "아직 확인하지 못한 요구사항이 하나라도 남아 있으면 최종 답변을 쓰지 말고 추가 도구 호출을 계속하세요. "
            "조건 검색 요청은 brand_filter_search 도구를 사용하세요. "
            "연도별 추이 요청은 brand_trend 도구를 사용하세요. "
            "브랜드명이 모호하면 brand_fallback_lookup 도구를 사용하세요. "
            "최종 답변 전에는 이미 확보한 정보와 아직 비어 있는 정보를 스스로 점검하세요. "
            "질문이 '가장 높다/낮다/많다/적다'를 요구하면 반드시 해당 기준으로 정렬된 결과를 근거로 대표 브랜드를 고르세요. "
            "질문이 '비슷한 규모'를 요구하면 store_count 차이가 가장 작은 후보를 우선 선택하세요. "
            "질문이 여러 후보 중 대표 1개를 고르라고 하면, 가능하면 먼저 유효 후보를 2개 이상 밝힌 뒤 선택 근거를 짧게 설명하세요. "
            "질문이 최신 기준 비교를 요구하면 가능할 때 store_count, avg_sales_krw, churn_rate, startup_total_initial_cost_krw를 함께 포함하세요. "
            "극값을 찾는 질문이 아니라면 avg_sales_krw가 0이거나 핵심 지표가 비어 있는 후보보다 데이터가 더 충실한 후보를 우선하세요."
        )
    )


def build_llm(model: str, tools_by_name: dict[str, object] | None = None) -> ChatOpenAI:
    llm = ChatOpenAI(model=model, temperature=0)
    if tools_by_name:
        return llm.bind_tools(list(tools_by_name.values()))
    return llm


def _extract_text_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _json_only_response(llm: ChatOpenAI, prompt: str) -> dict[str, Any]:
    response = llm.invoke([HumanMessage(content=prompt)])
    text = _extract_text_content(response).strip()
    if "```" in text:
        chunks = [chunk.strip() for chunk in text.split("```") if chunk.strip()]
        for chunk in chunks:
            if chunk.startswith("json"):
                text = chunk[4:].strip()
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def invoke_tool_call(tools_by_name: dict[str, object], call: dict[str, object]) -> dict[str, object]:
    tool = tools_by_name.get(str(call["name"]))
    if tool is None:
        return {
            "error": f"unknown_tool:{call['name']}",
            "tool_name": call["name"],
            "input_args": call.get("args"),
        }
    try:
        return tool.invoke(call["args"])
    except BrandResolutionError as exc:
        payload = exc.to_payload()
        payload["tool_name"] = call["name"]
        payload["input_args"] = call["args"]
        return payload
    except Exception as exc:  # noqa: BLE001
        return {
            "error": str(exc),
            "tool_name": call["name"],
            "input_args": call.get("args"),
        }


def append_tool_results(
    messages: list[SystemMessage | HumanMessage | AIMessage | ToolMessage],
    ai_msg: AIMessage,
    tools_by_name: dict[str, object],
) -> list[dict[str, Any]]:
    messages.append(ai_msg)
    tool_events: list[dict[str, Any]] = []
    for call in ai_msg.tool_calls:
        tool_result = invoke_tool_call(tools_by_name, call)
        messages.append(
            ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=call["id"],
            )
        )
        tool_events.append(
            {
                "tool_name": call["name"],
                "input_args": call.get("args", {}),
                "result": tool_result,
            }
        )
    return tool_events


def plan_query(query: str, model: str) -> dict[str, Any]:
    planner = build_llm(model=model)
    prompt = (
        "아래 사용자 질문을 처리하기 위한 작업 계획을 JSON으로만 작성하세요.\n"
        "반드시 다음 키를 포함하세요: sub_questions, required_slots, suggested_tools.\n"
        "sub_questions와 required_slots는 문자열 리스트여야 합니다.\n"
        "suggested_tools는 brand_overview, brand_compare, brand_filter_search, brand_trend, brand_fallback_lookup 중에서만 고르세요.\n"
        "JSON 외의 설명은 쓰지 마세요.\n\n"
        f"질문: {query}"
    )
    plan = _json_only_response(planner, prompt)
    sub_questions = plan.get("sub_questions")
    required_slots = plan.get("required_slots")
    suggested_tools = plan.get("suggested_tools")
    return {
        "sub_questions": [str(x) for x in sub_questions if str(x).strip()] if isinstance(sub_questions, list) else [query],
        "required_slots": [str(x) for x in required_slots if str(x).strip()] if isinstance(required_slots, list) else [query],
        "suggested_tools": [str(x) for x in suggested_tools if str(x).strip()] if isinstance(suggested_tools, list) else [],
    }


def summarize_tool_events(tool_events: list[dict[str, Any]]) -> dict[str, Any]:
    brand_names: set[str] = set()
    tools_used: list[str] = []
    errors: list[str] = []
    for event in tool_events:
        tool_name = str(event.get("tool_name") or "")
        if tool_name:
            tools_used.append(tool_name)
        result = event.get("result")
        if isinstance(result, dict):
            if result.get("error"):
                errors.append(str(result["error"]))
            brand = result.get("brand")
            if isinstance(brand, dict) and brand.get("brand_name"):
                brand_names.add(str(brand["brand_name"]))
            for key in ("brand_a", "brand_b"):
                side = result.get(key)
                if isinstance(side, dict):
                    b = side.get("brand")
                    if isinstance(b, dict) and b.get("brand_name"):
                        brand_names.add(str(b["brand_name"]))
            for row in result.get("results", []) or []:
                if isinstance(row, dict):
                    brand = row.get("brand")
                    if isinstance(brand, dict) and brand.get("brand_name"):
                        brand_names.add(str(brand["brand_name"]))
    return {
        "tools_used": tools_used,
        "brands_seen": sorted(brand_names),
        "errors": errors,
    }


def build_query_guidance(query: str) -> list[str]:
    lowered = re.sub(r"\s+", "", query.lower())
    guidance: list[str] = []
    if any(token in lowered for token in ["가장높", "가장낮", "가장많", "가장적"]):
        guidance.append("랭킹형 질문이므로 정확한 정렬 기준을 확인한 뒤 최종 브랜드를 골라야 합니다.")
    if "비슷한규모" in lowered:
        guidance.append("비슷한 규모는 store_count 차이가 가장 작은 후보를 우선 선택해야 합니다.")
    if any(token in lowered for token in ["몇개", "후보", "브랜드들"]):
        guidance.append("후보를 몇 개 찾으라는 질문이면 가능할 때 후보 브랜드를 2개 이상 명시해야 합니다.")
    if any(token in lowered for token in ["추이", "흐름", "2021년부터", "2022년부터", "최근몇년"]):
        guidance.append("추이 질문이므로 brand_trend 결과가 필요합니다.")
    if "최신기준으로비교" in lowered or ("최신" in query and "비교" in query):
        guidance.append(
            "최신 기준 비교는 가능하면 store_count, avg_sales_krw, churn_rate, startup_total_initial_cost_krw를 함께 담아야 합니다."
        )
    return guidance


def deterministic_review(query: str, tool_events: list[dict[str, Any]]) -> dict[str, Any]:
    lowered = re.sub(r"\s+", "", query.lower())
    tools_used = [str(event.get("tool_name") or "") for event in tool_events]
    brands_seen = set(summarize_tool_events(tool_events).get("brands_seen", []))
    missing_slots: list[str] = []

    filter_result_sizes = [
        len(result.get("results", []) or [])
        for event in tool_events
        for result in [event.get("result")]
        if isinstance(result, dict) and isinstance(result.get("results"), list)
    ]

    if any(token in lowered for token in ["추이", "흐름", "2021년부터", "2022년부터", "최근몇년"]) and "brand_trend" not in tools_used:
        missing_slots.append("연도별 추이 확인이 아직 없습니다.")

    if "비교" in query and len(brands_seen) < 2:
        missing_slots.append("비교 대상 브랜드가 아직 충분히 확정되지 않았습니다.")

    if any(token in lowered for token in ["몇개", "후보", "브랜드들"]) and max(filter_result_sizes or [0]) < 2:
        missing_slots.append("후보 브랜드를 여러 개 확보하지 못했습니다.")

    if ("최신기준으로비교" in lowered or ("최신" in query and "비교" in query)) and "brand_compare" not in tools_used:
        missing_slots.append("최신 기준 핵심 지표 비교가 아직 불충분할 수 있습니다.")

    return {
        "all_answered": not missing_slots,
        "missing_slots": missing_slots,
        "should_call_more_tools": bool(missing_slots),
    }


def review_progress(query: str, plan: dict[str, Any], tool_events: list[dict[str, Any]], model: str) -> dict[str, Any]:
    reviewer = build_llm(model=model)
    compact = [
        {
            "tool_name": event.get("tool_name"),
            "input_args": event.get("input_args"),
            "result": event.get("result"),
        }
        for event in tool_events[-6:]
    ]
    query_guidance = build_query_guidance(query)
    prompt = (
        "아래는 사용자 질문, 작업 계획, 그리고 지금까지의 도구 결과입니다.\n"
        "현재 시점의 진행 상태를 JSON으로만 평가하세요.\n"
        "반드시 다음 키를 포함하세요: all_answered, missing_slots, should_call_more_tools.\n"
        "missing_slots는 문자열 리스트여야 합니다.\n"
        "질문별 체크포인트를 엄격하게 적용하세요.\n"
        "JSON 외의 설명은 쓰지 마세요.\n\n"
        f"질문: {query}\n"
        f"계획: {json.dumps(plan, ensure_ascii=False)}\n"
        f"체크포인트: {json.dumps(query_guidance, ensure_ascii=False)}\n"
        f"도구 결과: {json.dumps(compact, ensure_ascii=False)}"
    )
    review = _json_only_response(reviewer, prompt)
    missing_slots = review.get("missing_slots")
    heuristic = deterministic_review(query=query, tool_events=tool_events)
    llm_missing = [str(x) for x in missing_slots if str(x).strip()] if isinstance(missing_slots, list) else []
    combined_missing = []
    for item in llm_missing + heuristic["missing_slots"]:
        if item not in combined_missing:
            combined_missing.append(item)
    return {
        "all_answered": bool(review.get("all_answered")) and not combined_missing,
        "missing_slots": combined_missing,
        "should_call_more_tools": bool(
            review.get("should_call_more_tools", not bool(review.get("all_answered")))
        )
        or bool(combined_missing),
    }


def run_once(
    query: str,
    model: str = "gpt-4.1-mini",
    build_dir: Path | str = Path("db_chatbot/build"),
    source_mode: str = "api_selected",
    api_data_root: Path | str = Path("db_chatbot/api_data"),
    max_tool_rounds: int = 5,
) -> str:
    load_env_file(Path("db_chatbot/.env"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in db_chatbot/.env or environment.")

    store = BrandDataStore(
        build_dir=Path(build_dir),
        source_mode=source_mode,
        api_data_root=Path(api_data_root),
    )
    tools_by_name, _ = build_tools(store)
    llm = build_llm(model=model, tools_by_name=tools_by_name)
    plan = plan_query(query=query, model=model)
    messages: list[SystemMessage | HumanMessage | AIMessage | ToolMessage] = [
        build_system_prompt(),
        HumanMessage(
            content=(
                "아래는 이 질문을 처리하기 위한 사전 계획입니다. "
                "실제 답변은 반드시 도구 결과를 근거로 작성하세요.\n"
                f"{json.dumps(plan, ensure_ascii=False)}"
            )
        ),
        HumanMessage(content=query),
    ]
    all_tool_events: list[dict[str, Any]] = []

    for step in range(max(1, max_tool_rounds)):
        ai_msg = llm.invoke(messages)
        if not ai_msg.tool_calls:
            if not all_tool_events:
                fallback_result = invoke_tool_call(
                    tools_by_name,
                    {
                        "name": "brand_fallback_lookup",
                        "args": {"query": query, "top_k": 5, "include_overview": True},
                    },
                )
                messages.append(ai_msg)
                messages.append(
                    ToolMessage(
                        content=json.dumps(fallback_result, ensure_ascii=False),
                        tool_call_id=f"advanced-fallback-{step}",
                    )
                )
                all_tool_events.append(
                    {
                        "tool_name": "brand_fallback_lookup",
                        "input_args": {"query": query, "top_k": 5, "include_overview": True},
                        "result": fallback_result,
                    }
                )
                continue
            review = review_progress(query=query, plan=plan, tool_events=all_tool_events, model=model)
            if review["all_answered"] or not review["should_call_more_tools"]:
                messages.append(ai_msg)
                break
            messages.append(ai_msg)
            messages.append(
                HumanMessage(
                    content=(
                        "아직 최종 답변을 쓰지 마세요. 남은 요구사항을 채우기 위해 추가 도구를 호출하세요.\n"
                        f"{json.dumps(review, ensure_ascii=False)}"
                    )
                )
            )
            continue

        tool_events = append_tool_results(messages, ai_msg, tools_by_name)
        all_tool_events.extend(tool_events)
        if step < max_tool_rounds - 1:
            review = review_progress(query=query, plan=plan, tool_events=all_tool_events, model=model)
            if review["all_answered"]:
                messages.append(
                    HumanMessage(
                        content=(
                            "현재까지의 정보로 질문의 요구사항이 모두 채워졌습니다. "
                            "이제 최종 답변을 작성해도 됩니다."
                        )
                    )
                )
                break
            if review["missing_slots"]:
                messages.append(
                    HumanMessage(
                        content=(
                            "아직 최종 답변을 쓰지 말고 남은 요구사항을 우선 해결하세요.\n"
                            f"진행 상태: {json.dumps(review, ensure_ascii=False)}\n"
                            f"수집 상태: {json.dumps(summarize_tool_events(all_tool_events), ensure_ascii=False)}"
                        )
                    )
                )

    messages.append(
        HumanMessage(
            content=(
                "지금까지의 도구 결과를 바탕으로 최종 답변을 작성하세요. "
                "질문의 각 요구사항을 빠짐없이 점검하고, 아직 없는 정보는 '원천 데이터에 없음'으로 명시하세요.\n"
                f"질문별 체크포인트: {json.dumps(build_query_guidance(query), ensure_ascii=False)}\n"
                f"계획: {json.dumps(plan, ensure_ascii=False)}\n"
                f"수집 상태: {json.dumps(summarize_tool_events(all_tool_events), ensure_ascii=False)}"
            )
        )
    )
    for _ in range(3):
        final_msg = llm.invoke(messages)
        if final_msg.tool_calls:
            tool_events = append_tool_results(messages, final_msg, tools_by_name)
            all_tool_events.extend(tool_events)
            messages.append(
                HumanMessage(
                    content=(
                        "도구 결과가 추가되었습니다. 이제 최종 답변을 한국어로 작성하세요. "
                        "더 이상 필요한 정보가 없다면 추가 도구 호출 없이 답변만 출력하세요.\n"
                        f"질문별 체크포인트: {json.dumps(build_query_guidance(query), ensure_ascii=False)}\n"
                        f"수집 상태: {json.dumps(summarize_tool_events(all_tool_events), ensure_ascii=False)}"
                    )
                )
            )
            continue

        content = _extract_text_content(final_msg).strip()
        if content:
            return content
        messages.append(
            HumanMessage(
                content=(
                    "방금 응답이 비어 있었습니다. 추가 도구 호출 없이 최종 답변 본문만 한국어로 작성하세요."
                )
            )
        )

    return "최종 답변을 생성하지 못했습니다."


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the iterative advanced chatbot.")
    parser.add_argument("--query", type=str, default=None, help="Single prompt to run.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model name.")
    parser.add_argument("--max-tool-rounds", type=int, default=5)
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
                max_tool_rounds=args.max_tool_rounds,
            )
        )
        return 0

    print("Advanced chatbot ready. Type 'exit' to quit.")
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
                    max_tool_rounds=args.max_tool_rounds,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
