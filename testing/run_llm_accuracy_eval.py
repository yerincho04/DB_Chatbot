#!/usr/bin/env python3
"""Evaluate benchmark output against expected answers using an LLM judge."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "testing" / "cases" / "llm_accuracy_cases.json"
DEFAULT_INPUT = PROJECT_ROOT / "testing" / "artifacts" / "benchmarks" / "agent_mode_benchmark_output.json"
DEFAULT_OUT = PROJECT_ROOT / "testing" / "artifacts" / "reports" / "llm_accuracy_eval_output.json"
MODE_LABELS = {
    "openai_api": "db_chatbot/chat_app_openai_api.py",
    "simple": "db_chatbot/chat_app.py",
    "advanced": "db_chatbot/chat_app_advanced.py",
}

HARD_CASE_RULES: dict[str, dict[str, list[list[str]]]] = {
    "llm_accuracy:011": {
        "expected_selected_brands": [["온누리약국"], ["파리바게뜨"]],
        "required_logic_groups": [
            ["가장낮은두브랜드", "가장낮은2개브랜드"],
            ["202", "2.02"],
            ["222", "2.22"],
            ["2143", "2082", "감소"],
            ["3390", "3419", "증가"],
        ],
    },
    "llm_accuracy:012": {
        "expected_selected_brands": [["교촌치킨"]],
        "required_logic_groups": [
            ["1377"],
            ["500개이상많", "500개이상"],
            ["조건에맞는브랜드", "온누리약국", "이디야커피", "크린토피아"],
            ["139402"],
        ],
    },
    "llm_accuracy:013": {
        "expected_selected_brands": [["교촌치킨"], ["카카오t블루", "kakaotblue"]],
        "required_logic_groups": [
            ["1377"],
            ["500개이상많", "500개이상"],
            ["가장많은브랜드는카카오t블루", "카카오t블루가가장많"],
            ["27177", "1000"],
            ["139402"],
        ],
    },
    "llm_accuracy:014": {
        "expected_selected_brands": [["bbq"], ["지에스25", "gs25", "gs 25"]],
        "required_logic_groups": [
            ["90789", "523"],
            ["창업비용은더낮고직원수는더많", "직원수가가장많"],
            ["73030", "7563", "17272"],
            ["641457"],
        ],
    },
    "llm_accuracy:015": {
        "expected_selected_brands": [["비에이치씨", "bhc"], ["60계"]],
        "required_logic_groups": [
            ["80000천원이상", "90000천원이하", "80000", "90000"],
            ["점포수가가장큰브랜드는bhc", "bhc가가장크", "2291"],
            ["60계", "661"],
            ["521035", "546729", "증가"],
            ["519867", "468124", "감소"],
        ],
    },
    "llm_accuracy:016": {
        "expected_selected_brands": [["교촌치킨"], ["맘스터치", "굽네치킨"]],
        "required_logic_groups": [
            ["가족운영비율", "원천데이터에없"],
            ["비슷한규모브랜드"],
            ["1377", "694300", "153"],
            ["교촌치킨이더높", "교촌치킨이더낮"],
        ],
    },
    "llm_accuracy:018": {
        "expected_selected_brands": [["버거킹", "burgerking"]],
        "required_logic_groups": [
            ["평균매출은낮지만창업비용도더낮", "창업비용도더낮"],
            ["1077407", "542539"],
            ["성장", "추이"],
        ],
    },
    "llm_accuracy:019": {
        "expected_selected_brands": [["씨유", "cu"], ["지에스25", "gs25", "gs 25"]],
        "required_logic_groups": [
            ["17576", "17272"],
            ["627969", "641457"],
            ["1142", "1149"],
            ["72700", "73030"],
            ["2839", "2679"],
            ["씨유가더많이늘", "씨유가더크"],
        ],
    },
    "llm_accuracy:020": {
        "expected_selected_brands": [["gsthefresh", "gs the fresh"], ["oliveyoung", "olive young", "올리브영"]],
        "required_logic_groups": [
            ["160에서313", "153개증가"],
            ["2536871", "2479052", "감소"],
            ["236에서226", "10개감소"],
            ["1314950", "2096276", "증가"],
            ["둘다churnrate하락", "두브랜드모두churnrate하락", "둘다낮아졌"],
        ],
    },
}

DIFFICULTY_WEIGHTS = {
    "easy": {
        "factual_accuracy_score": 0.55,
        "completeness_score": 0.2,
        "constraint_satisfaction_score": 0.1,
        "helpfulness_score": 0.1,
        "insightfulness_score": 0.05,
        "value_add_score": 0.0,
    },
    "medium": {
        "factual_accuracy_score": 0.43,
        "completeness_score": 0.2,
        "constraint_satisfaction_score": 0.15,
        "helpfulness_score": 0.1,
        "insightfulness_score": 0.1,
        "value_add_score": 0.02,
    },
    "hard": {
        "factual_accuracy_score": 0.37,
        "completeness_score": 0.2,
        "constraint_satisfaction_score": 0.25,
        "helpfulness_score": 0.05,
        "insightfulness_score": 0.1,
        "value_add_score": 0.03,
    },
}

HARD_RULE_BLEND_WEIGHTS = {
    "judge_constraint_weight": 0.55,
    "hard_rule_weight": 0.45,
}


def compute_final_score(
    difficulty: str,
    factual_accuracy_score: float,
    completeness_score: float,
    constraint_satisfaction_score: float,
    helpfulness_score: float,
    insightfulness_score: float,
    value_add_score: float,
) -> float:
    weights = DIFFICULTY_WEIGHTS.get(difficulty, DIFFICULTY_WEIGHTS["medium"])
    weighted = (
        factual_accuracy_score * weights["factual_accuracy_score"]
        + completeness_score * weights["completeness_score"]
        + constraint_satisfaction_score * weights["constraint_satisfaction_score"]
        + helpfulness_score * weights["helpfulness_score"]
        + insightfulness_score * weights["insightfulness_score"]
        + value_add_score * weights["value_add_score"]
    )

    if difficulty == "hard":
        # Hard questions should be dominated by whether the response actually
        # satisfied the selection/filter/ranking constraints, not by fluency.
        bottleneck = min(
            factual_accuracy_score,
            completeness_score,
            constraint_satisfaction_score,
        )
        return round(weighted * (0.55 + 0.45 * bottleneck), 4)

    if difficulty == "medium":
        bottleneck = min(factual_accuracy_score, completeness_score)
        return round(weighted * (0.7 + 0.3 * bottleneck), 4)

    return round(weighted, 4)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(score, 1.0))


def normalize_match_text(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum() or ("가" <= ch <= "힣"))


def any_group_match(normalized_response: str, options: list[str]) -> bool:
    return any(normalize_match_text(option) in normalized_response for option in options if option)


def compute_hard_rule_score(case_index: str, response_text: str) -> float | None:
    rules = HARD_CASE_RULES.get(case_index)
    if not rules:
        return None
    normalized_response = normalize_match_text(response_text)
    checks: list[float] = []
    for group in rules.get("expected_selected_brands", []):
        checks.append(1.0 if any_group_match(normalized_response, group) else 0.0)
    for group in rules.get("required_logic_groups", []):
        checks.append(1.0 if any_group_match(normalized_response, group) else 0.0)
    if not checks:
        return None
    return round(sum(checks) / len(checks), 4)


def extract_numeric_tokens(text: str) -> list[str]:
    seen: list[str] = []
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text)):
        token = raw.replace(",", "")
        if len(token) < 2:
            continue
        if token not in seen:
            seen.append(token)
    return seen


def has_strong_numeric_core_match(case: dict[str, Any], response_text: str) -> bool:
    expected_response = str(case.get("expected_response") or "")
    expected_tokens = extract_numeric_tokens(expected_response)
    if not expected_tokens:
        return False
    normalized_response = str(response_text).replace(",", "")
    matched = sum(1 for token in expected_tokens if token in normalized_response)
    return matched / len(expected_tokens) >= 0.85


def compute_reference_coverage(case: dict[str, Any], response_text: str) -> float:
    normalized_response = normalize_match_text(response_text)
    points = [str(x) for x in case.get("reference_points", []) if str(x).strip()]
    if not points:
        return 0.0
    checks = 0
    hits = 0
    for point in points:
        options = [tok.replace(",", "") for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", point)]
        hangul_chunks = re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9() ]+", point)
        candidate_chunks = [normalize_match_text(chunk) for chunk in hangul_chunks if normalize_match_text(chunk)]
        local_hit = False
        if options:
            checks += 1
            if any(opt in normalized_response for opt in options):
                local_hit = True
        elif candidate_chunks:
            checks += 1
            if any(chunk in normalized_response for chunk in candidate_chunks):
                local_hit = True
        if local_hit:
            hits += 1
    if checks == 0:
        return 0.0
    return round(hits / checks, 4)


def judge_response(
    client: Any,
    model: str,
    case: dict[str, Any],
    case_index: str,
    response_text: str,
) -> dict[str, Any]:
    difficulty = str(case.get("difficulty") or "unknown")
    prompt = {
        "difficulty": difficulty,
        "question": case.get("question"),
        "expected_response": case.get("expected_response"),
        "reference_points": case.get("reference_points", []),
        "grading_notes": case.get("grading_notes", []),
        "actual_response": response_text,
    }
    system = (
        "당신은 한국어 데이터 질의응답 결과를 평가하는 엄격하지만 공정한 채점기입니다.\n"
        "사용자 질문, 기대 답변, 참고 포인트, 실제 응답을 비교해서 채점하세요.\n"
        "반드시 한국어로 판단 근거를 작성하세요.\n"
        "표현 방식이 달라도 의미가 같으면 정답으로 인정하세요. 문장 순서, 표 형식, 목록 형식 차이는 감점 사유가 아닙니다.\n"
        "특히 금액 표기는 다음을 유연하게 해석하세요: 원, 천원, 만원, 억 원 등 단위가 달라도 실제 의미가 같으면 맞는 것으로 보세요.\n"
        "예: 150,800천원, 1억 5,080만 원, 150,800,000원은 같은 값으로 취급할 수 있습니다.\n"
        "기대 답변과 실제 응답의 금액 단위가 다르면 반드시 환산 후 비교하세요. 표면 문자열이 다르다는 이유만으로 오답 처리하면 안 됩니다.\n"
        "실제 응답이 기대 답변보다 더 많은 설명, 보조 맥락, 추가 비교를 포함하더라도 직접적인 모순이 없다면 감점하지 마세요.\n"
        "특히 상세한 답변은 핵심 정답이 맞는 한, 누락 없는 요약보다 불리하게 채점하면 안 됩니다.\n"
        "질문이 열린 선택형이면 기대 답변에 등장한 브랜드만 정답이라고 단정하지 마세요.\n"
        "사용자 질문이 특정 브랜드를 명시적으로 요구하지 않았고, 실제 응답이 원문 제약을 만족하는 다른 브랜드를 골랐다면 정답 가능성을 열어두세요.\n"
        "다만 '가장 높다/낮다/많다/적다' 또는 '가장 비슷하다'처럼 결정 규칙이 분명한 경우에는 그 규칙 위반을 감점하세요.\n"
        "반대로 실제 수치가 다르거나, 핵심 비교 결과가 바뀌거나, 데이터에 없는 내용을 단정적으로 추가하면 감점하세요.\n"
        "다음 4개 점수를 각각 0.0~1.0 사이로 채점하세요.\n"
        "1) factual_accuracy_score: 핵심 사실과 수치의 정확성\n"
        "2) completeness_score: 질문이 요구한 항목을 얼마나 빠짐없이 답했는지\n"
        "3) constraint_satisfaction_score: 질문의 필터링, 선택, 랭킹, 비교 조건을 정확히 만족했는지\n"
        "4) helpfulness_score: 사용자가 바로 이해하고 활용하기 쉬운 답변인지\n"
        "5) insightfulness_score: 단순 나열을 넘어 비교, 요약, 흐름 해석이 유의미한지\n"
        "6) value_add_score: 정답을 맞춘 뒤 추가로 제공한 비교, 해석, 맥락이 실제로 유용한지\n"
        "특히 hard 난이도에서는 constraint_satisfaction_score를 중요하게 보세요.\n"
        "hard 질문에서 잘못된 브랜드를 고르거나, 필터/랭킹/선정 논리를 빠뜨리면 감점하세요.\n"
        "다만 중간 과정이나 표현이 조금 달라도, 핵심 브랜드 선택과 주요 비교 결론이 대체로 맞으면 과도하게 깎지 마세요.\n"
        "is_correct는 형식이 다르더라도 핵심 사실과 결론이 대체로 맞으면 true로 줄 수 있습니다. hard 질문도 완벽한 문장이나 모든 보조 수치가 있어야만 true인 것은 아닙니다.\n"
        "value_add_score는 많이 말한 것 자체가 아니라, 핵심 답이 이미 맞는 상태에서 추가 설명이 실제로 도움이 될 때만 높게 주세요.\n"
        "핵심 답이 틀렸거나 제약 조건을 제대로 만족하지 못한 경우에는 value_add_score를 높게 주면 안 됩니다.\n"
        "strict JSON만 반환하세요. 반드시 다음 키를 포함하세요:\n"
        "is_correct, factual_accuracy_score, completeness_score, constraint_satisfaction_score, helpfulness_score, insightfulness_score, value_add_score, final_score, rationale, matched_points, missing_points, unsupported_claims.\n"
        "matched_points, missing_points, unsupported_claims는 짧은 한국어 문자열 배열이어야 합니다."
    )
    user = json.dumps(prompt, ensure_ascii=False)
    result = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = (result.choices[0].message.content or "").strip()
    parsed = json.loads(text)
    factual_accuracy_score = round(clamp_score(parsed.get("factual_accuracy_score")), 4)
    completeness_score = round(clamp_score(parsed.get("completeness_score")), 4)
    constraint_satisfaction_score = round(clamp_score(parsed.get("constraint_satisfaction_score")), 4)
    helpfulness_score = round(clamp_score(parsed.get("helpfulness_score")), 4)
    insightfulness_score = round(clamp_score(parsed.get("insightfulness_score")), 4)
    value_add_score = round(clamp_score(parsed.get("value_add_score")), 4)
    reference_coverage = compute_reference_coverage(case=case, response_text=response_text)
    strong_numeric = has_strong_numeric_core_match(case, response_text)

    if difficulty in {"easy", "medium"} and strong_numeric:
        factual_accuracy_score = max(factual_accuracy_score, 0.97)
        completeness_score = max(completeness_score, 0.96)
        constraint_satisfaction_score = max(constraint_satisfaction_score, 0.95)
        helpfulness_score = max(helpfulness_score, 0.9)

    if reference_coverage >= 0.75:
        factual_accuracy_score = max(factual_accuracy_score, 0.9)
        completeness_score = max(completeness_score, 0.88)
        helpfulness_score = max(helpfulness_score, 0.85)

    if difficulty == "hard" and reference_coverage >= 0.6:
        constraint_satisfaction_score = max(constraint_satisfaction_score, 0.72)
        insightfulness_score = max(insightfulness_score, 0.65)

    if not (
        factual_accuracy_score >= 0.8
        and completeness_score >= 0.75
        and constraint_satisfaction_score >= 0.7
    ):
        value_add_score = min(value_add_score, 0.25)
    elif reference_coverage >= 0.75:
        value_add_score = max(value_add_score, 0.7)

    hard_rule_score = None
    if difficulty == "hard":
        hard_rule_score = compute_hard_rule_score(case_index=case_index, response_text=response_text)
        if hard_rule_score is not None:
            constraint_satisfaction_score = round(
                (
                    HARD_RULE_BLEND_WEIGHTS["judge_constraint_weight"] * constraint_satisfaction_score
                    + HARD_RULE_BLEND_WEIGHTS["hard_rule_weight"] * hard_rule_score
                ),
                4,
            )

    final_score = compute_final_score(
        difficulty=difficulty,
        factual_accuracy_score=factual_accuracy_score,
        completeness_score=completeness_score,
        constraint_satisfaction_score=constraint_satisfaction_score,
        helpfulness_score=helpfulness_score,
        insightfulness_score=insightfulness_score,
        value_add_score=value_add_score,
    )
    judge_is_correct = bool(parsed.get("is_correct"))
    if not judge_is_correct and strong_numeric and reference_coverage >= 0.75 and factual_accuracy_score >= 0.9:
        judge_is_correct = True
    if difficulty == "hard":
        is_correct = (
            judge_is_correct
            and factual_accuracy_score >= 0.78
            and completeness_score >= 0.68
            and constraint_satisfaction_score >= 0.62
            and final_score >= 0.68
        )
    elif difficulty == "medium":
        is_correct = (
            judge_is_correct
            and factual_accuracy_score >= 0.85
            and completeness_score >= 0.8
            and constraint_satisfaction_score >= 0.72
            and final_score >= 0.78
        )
    else:
        is_correct = (
            judge_is_correct
            and factual_accuracy_score >= 0.85
            and completeness_score >= 0.78
            and final_score >= 0.78
        )
    return {
        "judge_is_correct": judge_is_correct,
        "is_correct": is_correct,
        "factual_accuracy_score": factual_accuracy_score,
        "completeness_score": completeness_score,
        "constraint_satisfaction_score": constraint_satisfaction_score,
        "hard_rule_score": hard_rule_score,
        "reference_coverage_score": reference_coverage,
        "helpfulness_score": helpfulness_score,
        "insightfulness_score": insightfulness_score,
        "value_add_score": value_add_score,
        "final_score": final_score,
        "score": final_score,
        "rationale": str(parsed.get("rationale") or ""),
        "matched_points": [str(x) for x in parsed.get("matched_points", []) if str(x).strip()],
        "missing_points": [str(x) for x in parsed.get("missing_points", []) if str(x).strip()],
        "unsupported_claims": [
            str(x) for x in parsed.get("unsupported_claims", []) if str(x).strip()
        ],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        bucket = by_mode.setdefault(
            mode,
            {
                "mode": mode,
                "entrypoint": MODE_LABELS.get(mode, mode),
                "count": 0,
                "correct_count": 0,
                "accuracy": 0.0,
                "avg_score": 0.0,
                "avg_factual_accuracy_score": 0.0,
                "avg_completeness_score": 0.0,
                "avg_constraint_satisfaction_score": 0.0,
                "avg_helpfulness_score": 0.0,
                "avg_insightfulness_score": 0.0,
                "avg_value_add_score": 0.0,
                "avg_latency_sec": 0.0,
                "error_count": 0,
                "judge_error_count": 0,
            },
        )
        bucket["count"] += 1
        bucket["correct_count"] += int(bool(row.get("is_correct")))
        bucket["avg_score"] += float(row.get("score") or 0.0)
        bucket["avg_factual_accuracy_score"] += float(row.get("factual_accuracy_score") or 0.0)
        bucket["avg_completeness_score"] += float(row.get("completeness_score") or 0.0)
        bucket["avg_constraint_satisfaction_score"] += float(row.get("constraint_satisfaction_score") or 0.0)
        bucket["avg_helpfulness_score"] += float(row.get("helpfulness_score") or 0.0)
        bucket["avg_insightfulness_score"] += float(row.get("insightfulness_score") or 0.0)
        bucket["avg_value_add_score"] += float(row.get("value_add_score") or 0.0)
        bucket["avg_latency_sec"] += float(row.get("latency_sec") or 0.0)
        if row.get("error"):
            bucket["error_count"] += 1
        if row.get("judge_error"):
            bucket["judge_error_count"] += 1

    output = []
    for bucket in by_mode.values():
        count = max(1, int(bucket["count"]))
        bucket["accuracy"] = round(bucket["correct_count"] / count, 4)
        bucket["avg_score"] = round(bucket["avg_score"] / count, 4)
        bucket["avg_factual_accuracy_score"] = round(bucket["avg_factual_accuracy_score"] / count, 4)
        bucket["avg_completeness_score"] = round(bucket["avg_completeness_score"] / count, 4)
        bucket["avg_constraint_satisfaction_score"] = round(bucket["avg_constraint_satisfaction_score"] / count, 4)
        bucket["avg_helpfulness_score"] = round(bucket["avg_helpfulness_score"] / count, 4)
        bucket["avg_insightfulness_score"] = round(bucket["avg_insightfulness_score"] / count, 4)
        bucket["avg_value_add_score"] = round(bucket["avg_value_add_score"] / count, 4)
        bucket["avg_latency_sec"] = round(bucket["avg_latency_sec"] / count, 4)
        output.append(bucket)
    return sorted(output, key=lambda item: item["mode"])


def summarize_rows_by_difficulty(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        difficulty = str(row.get("difficulty") or "unknown")
        mode = str(row.get("mode") or "unknown")
        key = (difficulty, mode)
        bucket = by_key.setdefault(
            key,
            {
                "difficulty": difficulty,
                "mode": mode,
                "entrypoint": MODE_LABELS.get(mode, mode),
                "count": 0,
                "correct_count": 0,
                "accuracy": 0.0,
                "avg_score": 0.0,
                "avg_factual_accuracy_score": 0.0,
                "avg_completeness_score": 0.0,
                "avg_constraint_satisfaction_score": 0.0,
                "avg_helpfulness_score": 0.0,
                "avg_insightfulness_score": 0.0,
                "avg_value_add_score": 0.0,
                "avg_latency_sec": 0.0,
                "error_count": 0,
                "judge_error_count": 0,
            },
        )
        bucket["count"] += 1
        bucket["correct_count"] += int(bool(row.get("is_correct")))
        bucket["avg_score"] += float(row.get("score") or 0.0)
        bucket["avg_factual_accuracy_score"] += float(row.get("factual_accuracy_score") or 0.0)
        bucket["avg_completeness_score"] += float(row.get("completeness_score") or 0.0)
        bucket["avg_constraint_satisfaction_score"] += float(row.get("constraint_satisfaction_score") or 0.0)
        bucket["avg_helpfulness_score"] += float(row.get("helpfulness_score") or 0.0)
        bucket["avg_insightfulness_score"] += float(row.get("insightfulness_score") or 0.0)
        bucket["avg_value_add_score"] += float(row.get("value_add_score") or 0.0)
        bucket["avg_latency_sec"] += float(row.get("latency_sec") or 0.0)
        if row.get("error"):
            bucket["error_count"] += 1
        if row.get("judge_error"):
            bucket["judge_error_count"] += 1

    output = []
    for bucket in by_key.values():
        count = max(1, int(bucket["count"]))
        bucket["accuracy"] = round(bucket["correct_count"] / count, 4)
        bucket["avg_score"] = round(bucket["avg_score"] / count, 4)
        bucket["avg_factual_accuracy_score"] = round(bucket["avg_factual_accuracy_score"] / count, 4)
        bucket["avg_completeness_score"] = round(bucket["avg_completeness_score"] / count, 4)
        bucket["avg_constraint_satisfaction_score"] = round(bucket["avg_constraint_satisfaction_score"] / count, 4)
        bucket["avg_helpfulness_score"] = round(bucket["avg_helpfulness_score"] / count, 4)
        bucket["avg_insightfulness_score"] = round(bucket["avg_insightfulness_score"] / count, 4)
        bucket["avg_value_add_score"] = round(bucket["avg_value_add_score"] / count, 4)
        bucket["avg_latency_sec"] = round(bucket["avg_latency_sec"] / count, 4)
        output.append(bucket)
    return sorted(output, key=lambda item: (item["difficulty"], item["mode"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM-judged accuracy eval for agent-mode benchmark output.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / "db_chatbot" / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in db_chatbot/.env or environment.")

    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Missing dependency 'openai'. Install it in the active environment.") from exc

    cases_payload = load_json(args.cases)
    benchmark_rows = load_json(args.input)
    case_map = {
        str(item.get("index")): item
        for item in cases_payload.get("items", [])
        if isinstance(item, dict) and item.get("index")
    }

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    detailed_rows: list[dict[str, Any]] = []

    for row in benchmark_rows:
        index = str(row.get("index") or "")
        case = case_map.get(index)
        if not case:
            continue

        response = str(row.get("response") or "")
        runtime_error = row.get("error")
        judge_error: str | None = None

        if runtime_error or not response.strip():
            judge_result = {
                "is_correct": False,
                "factual_accuracy_score": 0.0,
                "completeness_score": 0.0,
                "constraint_satisfaction_score": 0.0,
                "helpfulness_score": 0.0,
                "insightfulness_score": 0.0,
                "value_add_score": 0.0,
                "final_score": 0.0,
                "score": 0.0,
                "rationale": "실행 오류 또는 빈 응답으로 인해 평가 가능한 답변이 없었습니다.",
                "matched_points": [],
                "missing_points": ["평가 가능한 유효 답변이 없습니다."],
                "unsupported_claims": [],
            }
        else:
            try:
                judge_result = judge_response(
                    client=client,
                    model=args.judge_model,
                    case=case,
                    case_index=index,
                    response_text=response,
                )
            except Exception as exc:  # noqa: BLE001
                judge_error = str(exc)
                judge_result = {
                    "is_correct": False,
                    "judge_is_correct": False,
                    "factual_accuracy_score": 0.0,
                    "completeness_score": 0.0,
                    "constraint_satisfaction_score": 0.0,
                    "helpfulness_score": 0.0,
                    "insightfulness_score": 0.0,
                    "value_add_score": 0.0,
                    "final_score": 0.0,
                    "score": 0.0,
                    "rationale": "채점 모델 호출에 실패했습니다.",
                    "matched_points": [],
                    "missing_points": [],
                    "unsupported_claims": [],
                }

        detailed_rows.append(
            {
                "index": index,
                "difficulty": row.get("difficulty") or str(case.get("difficulty") or "unknown"),
                "question": row.get("question") or case.get("question"),
                "expected_response": case.get("expected_response"),
                "judge_expected_difficulty": case.get("difficulty"),
                "mode": row.get("mode"),
                "entrypoint": MODE_LABELS.get(str(row.get("mode") or ""), str(row.get("mode") or "")),
                "latency_sec": row.get("latency_sec"),
                "response_chars": row.get("response_chars"),
                "error": runtime_error,
                "judge_error": judge_error,
                "response": response,
                **judge_result,
            }
        )
        status = "PASS" if judge_result["is_correct"] else "FAIL"
        print(f"[{status}] {index} / {row.get('mode')} score={judge_result['score']:.2f}")

    output = {
        "cases_file": str(args.cases),
        "input_file": str(args.input),
        "judge_model": args.judge_model,
        "summary_by_mode": summarize_rows(detailed_rows),
        "summary_by_difficulty": summarize_rows_by_difficulty(detailed_rows),
        "detailed_rows": detailed_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"wrote: {args.out}")
    print()
    print("accuracy by mode:")
    for row in output["summary_by_mode"]:
        print(
            f"- {row['mode']} ({row['entrypoint']}): "
            f"accuracy={row['accuracy']:.3f}, avg_final={row['avg_score']:.3f}, "
            f"avg_factual={row['avg_factual_accuracy_score']:.3f}, "
            f"avg_constraint={row['avg_constraint_satisfaction_score']:.3f}, "
            f"avg_helpfulness={row['avg_helpfulness_score']:.3f}, "
            f"avg_insight={row['avg_insightfulness_score']:.3f}, "
            f"avg_value_add={row['avg_value_add_score']:.3f}, "
            f"latency={row['avg_latency_sec']:.3f}s, errors={row['error_count']}, judge_errors={row['judge_error_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
