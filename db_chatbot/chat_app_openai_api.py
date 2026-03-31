#!/usr/bin/env python3
"""Direct OpenAI baseline without tool calls."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


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


def run_once(query: str, model: str = "gpt-4.1-mini") -> str:
    load_env_file(Path("db_chatbot/.env"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in db_chatbot/.env or environment.")

    llm = ChatOpenAI(model=model, temperature=0)
    system = SystemMessage(
        content=(
            "당신은 프랜차이즈 브랜드 질의응답 어시스턴트입니다. "
            "최종 답변은 반드시 한국어로 작성하세요. "
            "도구는 사용할 수 없으며, 답변은 사용자의 요청을 직접 처리하는 형태로 작성하세요. "
            "확실하지 않은 정보는 추정하지 말고 불확실하다고 말하세요."
        )
    )
    final_msg = llm.invoke([system, HumanMessage(content=query)])
    return str(final_msg.content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the direct OpenAI baseline without tools.")
    parser.add_argument("--query", type=str, default=None, help="Single prompt to run.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model name.")
    args = parser.parse_args()

    if args.query:
        print(run_once(args.query, model=args.model))
        return 0

    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        user = input("> ").strip()
        if user.lower() in {"exit", "quit"}:
            return 0
        if not user:
            continue
        try:
            print(run_once(user, model=args.model))
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
