#!/usr/bin/env python3
"""Generate a Korean business-plan-style mode comparison chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "testing" / "artifacts" / "reports" / "llm_accuracy_eval_output.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "testing" / "artifacts" / "visuals" / "mode_comparison_chart.png"

MODE_LABELS = {
    "openai_api": "ChatGPT",
    "simple": "자사 모델(기본형)",
    "advanced": "자사 모델(고급형)",
}

METRIC_LABELS = {
    "accuracy": "정답률",
    "avg_score": "평균 종합 점수",
}

METRIC_COLORS = {
    "accuracy": "#1F6D8C",
    "avg_score": "#2E86A7",
}

FONT_CANDIDATES = [
    "Apple SD Gothic Neo",
    "Malgun Gothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_matplotlib_fonts() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def extract_summary_rows(payload: dict) -> list[dict]:
    rows = payload.get("summary_by_mode", [])
    if not isinstance(rows, list):
        raise ValueError("summary_by_mode 형식이 올바르지 않습니다.")
    filtered = [row for row in rows if isinstance(row, dict) and row.get("mode") in MODE_LABELS]
    if len(filtered) != 3:
        raise ValueError("비교 대상 모드(openai_api, simple, advanced) 3개를 모두 찾지 못했습니다.")
    order = ["openai_api", "simple", "advanced"]
    return sorted(filtered, key=lambda row: order.index(str(row["mode"])))


def draw_chart(rows: list[dict], output_path: Path, title: str, subtitle: str | None) -> None:
    configure_matplotlib_fonts()

    models = [MODE_LABELS[str(row["mode"])] for row in rows]
    accuracy_vals = [float(row.get("accuracy") or 0.0) for row in rows]
    score_vals = [float(row.get("avg_score") or 0.0) for row in rows]

    x = list(range(len(models)))
    bar_width = 0.24

    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    bars_accuracy = ax.bar(
        [v - bar_width / 2 for v in x],
        accuracy_vals,
        width=bar_width,
        color=METRIC_COLORS["accuracy"],
        label=METRIC_LABELS["accuracy"],
    )
    bars_score = ax.bar(
        [v + bar_width / 2 for v in x],
        score_vals,
        width=bar_width,
        color=METRIC_COLORS["avg_score"],
        label=METRIC_LABELS["avg_score"],
    )

    ax.set_title(title, fontsize=20, fontweight="bold", pad=18)
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha="center", va="center", fontsize=11, color="#7A7A7A")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight="bold", color="#6F6F6F")
    ax.set_ylabel("점수", fontsize=12, color="#6F6F6F")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="#FFFFFF", alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors="#8A8A8A", labelsize=11, length=0)
    ax.tick_params(axis="x", colors="#6F6F6F", labelsize=12, length=0, pad=8)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#FFFFFF")
    ax.spines["bottom"].set_linewidth(0.8)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=11,
        labelcolor="#7A7A7A",
        handlelength=0.7,
        handletextpad=0.4,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    def annotate(bars) -> None:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#8A8A8A",
                fontweight="bold",
            )

    annotate(bars_accuracy)
    annotate(bars_score)

    footer = "출처: 내부 LLM 정확도 평가 결과"
    fig.text(0.99, 0.015, footer, ha="right", va="bottom", fontsize=9, color="#666666")

    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Korean comparison chart from eval JSON.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to eval JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output image.")
    parser.add_argument(
        "--title",
        default="모델별 질의응답 성능 비교",
        help="Chart title in Korean.",
    )
    parser.add_argument(
        "--subtitle",
        default="정답률 및 평균 종합 점수 기준",
        help="Optional subtitle in Korean.",
    )
    args = parser.parse_args()

    payload = load_json(args.input)
    rows = extract_summary_rows(payload)
    draw_chart(rows=rows, output_path=args.output, title=args.title, subtitle=args.subtitle)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
