#!/usr/bin/env python3
"""Generate a presentation-style diagram for the testing/eval pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "testing" / "artifacts" / "visuals" / "agent_architecture_comparison.png"

FONT_CANDIDATES = [
    "Apple SD Gothic Neo",
    "Malgun Gothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
]


def configure_fonts() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def round_box(ax, x, y, w, h, fc, ec, lw=2.0, radius=0.025):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def labeled_box(ax, x, y, w, h, title, subtitle="", fc="#FFFFFF", ec="#D0DCE8", lw=2.0, title_fs=13, sub_fs=10):
    round_box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, radius=0.02)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_fs, weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=sub_fs, color="#52606D")


def add_arrow(ax, start, end, text=None, text_xy=None, lw=2.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=18,
        linewidth=lw,
        color="black",
        shrinkA=2,
        shrinkB=2,
        connectionstyle="arc3",
    )
    ax.add_patch(patch)
    if text and text_xy:
        ax.text(text_xy[0], text_xy[1], text, fontsize=10.5, weight="bold", ha="center", va="center")


def add_chip(ax, x, y, w, h, text, fc, ec):
    round_box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.6, radius=0.018)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, weight="bold")


def draw_diagram(output_path: Path) -> None:
    configure_fonts()

    fig, ax = plt.subplots(figsize=(8.5, 8.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    round_box(ax, 0.03, 0.12, 0.94, 0.84, fc="white", ec="#A9CBE9", lw=4.0, radius=0.04)

    ax.text(0.5, 0.93, "LLM 응답 평가 파이프라인", ha="center", va="center", fontsize=22, weight="bold")

    # Top input box
    labeled_box(
        ax,
        0.24,
        0.79,
        0.23,
        0.11,
        "질문 / 기대답변 세트",
        "cases/agent_mode_cases.json\ncases/llm_accuracy_cases.json",
        fc="#CBF1C2",
        ec="#CBF1C2",
        lw=1.8,
        title_fs=12.5,
    )

    # Main pipeline box
    round_box(ax, 0.12, 0.40, 0.46, 0.24, fc="white", ec="#F4A7A7", lw=3.0, radius=0.025)
    add_chip(ax, 0.265, 0.575, 0.17, 0.036, "벤치마크 실행 시스템", fc="#FDF0F0", ec="#FDF0F0")

    labeled_box(ax, 0.145, 0.455, 0.12, 0.09, "OpenAI API", "chat_app_openai_api.py", fc="#F6FAFF", ec="#C9DDF2", lw=1.5, title_fs=11.5, sub_fs=9.2)
    labeled_box(ax, 0.29, 0.455, 0.12, 0.09, "자사 모델 기본", "chat_app.py", fc="#F6FAFF", ec="#C9DDF2", lw=1.5, title_fs=11.5, sub_fs=9.2)
    labeled_box(ax, 0.435, 0.455, 0.12, 0.09, "자사 모델 고급", "chat_app_advanced.py", fc="#F6FAFF", ec="#C9DDF2", lw=1.5, title_fs=11.5, sub_fs=9.2)

    # Right judging box
    labeled_box(
        ax,
        0.69,
        0.44,
        0.18,
        0.14,
        "LLM Judge",
        "run_llm_accuracy_eval.py\n정답 여부 + 점수 산출",
        fc="#EEF6FF",
        ec="#B6D1EC",
        lw=2.0,
        title_fs=13,
        sub_fs=9.5,
    )

    # Bottom sources / outputs
    labeled_box(ax, 0.15, 0.22, 0.15, 0.10, "benchmark output", "artifacts/benchmarks/\nllm_accuracy_benchmark_output.json", fc="#FFF7EC", ec="#F1D5AA", lw=1.6, title_fs=11.5, sub_fs=8.8)
    labeled_box(ax, 0.34, 0.22, 0.15, 0.10, "eval output", "artifacts/reports/\nllm_accuracy_eval_output.json", fc="#FFF7EC", ec="#F1D5AA", lw=1.6, title_fs=11.5, sub_fs=8.8)
    labeled_box(ax, 0.53, 0.22, 0.15, 0.10, "response examples", "artifacts/visuals/\nmodel_response_examples.md", fc="#FFF7EC", ec="#F1D5AA", lw=1.6, title_fs=11.5, sub_fs=8.8)
    labeled_box(ax, 0.72, 0.22, 0.15, 0.10, "comparison chart", "generate_mode_comparison_graph.py", fc="#FFF7EC", ec="#F1D5AA", lw=1.6, title_fs=11.5, sub_fs=8.8)

    # Flow arrows
    add_arrow(ax, (0.355, 0.79), (0.355, 0.64), text="입력 케이스", text_xy=(0.29, 0.71))
    add_arrow(ax, (0.59, 0.50), (0.69, 0.50), text="응답 결과 전달", text_xy=(0.64, 0.54))
    add_arrow(ax, (0.355, 0.40), (0.225, 0.32), text="benchmark 저장", text_xy=(0.25, 0.37))
    add_arrow(ax, (0.78, 0.44), (0.415, 0.32), text="judge 결과 저장", text_xy=(0.62, 0.38))
    add_arrow(ax, (0.415, 0.22), (0.605, 0.22), text="예시/그래프 생성", text_xy=(0.51, 0.18))

    # Small descriptive notes
    ax.text(0.35, 0.355, "run_agent_mode_benchmark.py", ha="center", va="center", fontsize=10.5, color="#596774", weight="bold")
    ax.text(0.78, 0.405, "정답 기반 평가", ha="center", va="center", fontsize=10.5, color="#596774", weight="bold")

    ax.text(0.5, 0.07, "(b) 내 파이프라인", ha="center", va="center", fontsize=16, weight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a pipeline diagram.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path.")
    args = parser.parse_args()
    draw_diagram(args.output)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
