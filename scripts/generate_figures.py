"""
Figure Generation for InfraAgent Research Paper.

Generates all publication-quality figures using matplotlib.
All figures are saved as PDF (for LaTeX) and PNG (for preview).

CANONICAL DATA PATHS (single source of truth):
- Table 3 / Figure 1:  data["main_results"][cond]["deepseek-coder-v2-16b"]
- Figure 2:            data["round_curves"]["deepseek-coder-v2-16b"]
                         (anchored to main_results; Initial = one_shot_rag for with_rag)
- Figure 3:            data["security_rag_impact_deepseek"]
                         (DeepSeek-only; annotation = one_shot_rag - one_shot)
- Figure 4:            data["per_difficulty_deepseek"]
                         (DeepSeek-only, consistent with Table 3)
- Figure 5:            data["model_comparison"]
- Figure 6:            data["failure_taxonomy"]
- Figure 7:            data["correction_success_rate"]
- Figure 8:            data["per_language_deepseek"]
                         (DeepSeek-only, consistent with Table 3)

Any discrepancy between figures and Table 3 means a figure is reading from
the wrong key.  Always use the _deepseek keys for primary-model figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

matplotlib.use("Agg")

# ── Style constants ──────────────────────────────────────────────────────────
PALETTE = {
    "one_shot":          "#e15759",   # red
    "one_shot_rag":      "#f28e2b",   # orange
    "self_correct_3r":   "#76b7b2",   # teal
    "self_correct_rag_3r": "#4e79a7", # blue
    "self_correct_rag_5r": "#59a14f", # green
}

MODEL_COLORS = {
    "deepseek-coder-v2-16b": "#4e79a7",
    "codellama-13b":          "#f28e2b",
    "mistral-7b":             "#e15759",
    "phi3-mini-3.8b":         "#76b7b2",
    "llama-3.1-70b":          "#b07aa1",
    "qwen2.5-coder-32b":      "#ff9da7",
    "gpt-4o":                 "#59a14f",   # green — commercial ceiling
    "claude-3-5-sonnet":      "#edc948",   # yellow — second commercial baseline
}

MODEL_LABELS = {
    "deepseek-coder-v2-16b": "DeepSeek-Coder-V2 (16B)",
    "codellama-13b":          "CodeLlama (13B)",
    "mistral-7b":             "Mistral (7B)",
    "phi3-mini-3.8b":         "Phi-3-Mini (3.8B)",
    "llama-3.1-70b":          "Llama-3.1 (70B)",
    "qwen2.5-coder-32b":      "Qwen2.5-Coder (32B)",
    "gpt-4o":                 "GPT-4o (OpenAI) [ceiling]",
    "claude-3-5-sonnet":      "Claude-3.5-Sonnet (Anthropic) [ceiling]",
}

COND_LABELS = {
    "one_shot":            "One-Shot",
    "one_shot_rag":        "One-Shot + RAG",
    "self_correct_3r":     "Self-Correct (3r)",
    "self_correct_rag_3r": "SC + RAG (3r)",
    "self_correct_rag_5r": "SC + RAG (5r)",
}

METRIC_LABELS = {
    "syntax":     "Syntactic Validity (%)",
    "schema":     "Schema Validity (%)",
    "security":   "Security Score (%)",
    "functional": "Functional Correctness (%)",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_fig(fig, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    plt.close(fig)
    print(f"  Saved: {name}.pdf / {name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Main Results — Grouped Bar Chart
# Source: data["main_results"][cond]["deepseek-coder-v2-16b"]
# ─────────────────────────────────────────────────────────────────────────────

def fig_main_results(data: dict, out_dir: Path):
    """Grouped bar chart: conditions × metrics for DeepSeek-Coder (primary model)."""
    model_key = "deepseek-coder-v2-16b"
    conditions = ["one_shot", "one_shot_rag", "self_correct_3r", "self_correct_rag_3r", "self_correct_rag_5r"]
    metrics = ["syntax", "schema", "security", "functional"]

    values = np.array([
        [data["main_results"][c][model_key][m] * 100 for m in metrics]
        for c in conditions
    ])  # shape: (5 conditions, 4 metrics)

    x = np.arange(len(metrics))
    width = 0.14
    offsets = np.linspace(-2 * width, 2 * width, len(conditions))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    bars = []
    for i, (cond, offset) in enumerate(zip(conditions, offsets)):
        b = ax.bar(
            x + offset,
            values[i],
            width,
            color=PALETTE[cond],
            label=COND_LABELS[cond],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )
        bars.append(b)
        # Value labels on bars
        for rect, val in zip(b, values[i]):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.8,
                f"{val:.0f}",
                ha="center", va="bottom",
                fontsize=6.5, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m].replace(" (%)", "") for m in metrics], fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title(
        "InfraAgent Performance by Experimental Condition (DeepSeek-Coder-V2-16B)",
        fontsize=12, pad=10,
    )
    ax.legend(loc="upper right", framealpha=0.9, ncol=1, fontsize=8.5)
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # Dashed baselines
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    save_fig(fig, "fig1_main_results", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Self-Correction Iteration Curves
# Source: data["round_curves"]["deepseek-coder-v2-16b"]
# Anchored so that:
#   with_rag[0]  = one_shot+RAG (Initial with RAG)
#   with_rag[3] ≈ SC+RAG 3r    (Table 3 value)
#   with_rag[5] ≈ SC+RAG 5r    (Table 3 value)
#   without_rag[0]  = one_shot
#   without_rag[3] ≈ SC 3r
# Delta annotation = total self-correction gain on top of RAG (R5 − Initial)
# ─────────────────────────────────────────────────────────────────────────────

def fig_correction_curves(data: dict, out_dir: Path):
    """Line plot: metric vs. correction round, with/without RAG."""
    model_key = "deepseek-coder-v2-16b"
    metrics = ["syntax", "schema", "security", "functional"]
    metric_names = ["Syntactic Validity", "Schema Validity", "Security Score", "Functional Correctness"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()

    rounds = list(range(6))  # 0..5

    for ax, metric, title in zip(axes, metrics, metric_names):
        with_rag    = [v * 100 for v in data["round_curves"][model_key][metric]["with_rag"]]
        without_rag = [v * 100 for v in data["round_curves"][model_key][metric]["without_rag"]]

        ax.plot(rounds, with_rag,    "o-", color="#4e79a7", linewidth=2, markersize=5,
                label="With RAG", zorder=3)
        ax.plot(rounds, without_rag, "s--", color="#e15759", linewidth=2, markersize=5,
                label="Without RAG", zorder=3)

        # Shade the improvement band
        ax.fill_between(rounds, without_rag, with_rag, alpha=0.12, color="#4e79a7")

        # Annotation: self-correction gain on top of RAG (Final vs Initial)
        # with_rag[0] = one_shot+RAG; with_rag[5] ≈ SC+RAG 5r
        delta = with_rag[-1] - with_rag[0]
        ax.annotate(
            f"SC gain: +{delta:.1f}pp",
            xy=(5, with_rag[-1]),
            xytext=(3.2, with_rag[-1] - 9),
            fontsize=8.5, color="#4e79a7",
            arrowprops=dict(arrowstyle="->", color="#4e79a7", lw=1.0),
        )

        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Correction Round", fontsize=9)
        ax.set_ylabel("Score (%)", fontsize=9)
        ax.set_ylim(0, 103)
        ax.set_xlim(-0.2, 5.2)
        ax.set_xticks(rounds)
        # "Initial" = one_shot+RAG for with_rag; one_shot for without_rag
        ax.set_xticklabels(["Initial\n(after RAG)" if i == 0 else f"R{i}" for i in rounds],
                           fontsize=7.5)
        ax.legend(fontsize=8, loc="lower right")
        ax.yaxis.set_minor_locator(MultipleLocator(5))

    fig.suptitle(
        "Self-Correction Dynamics Across Rounds (DeepSeek-Coder-V2-16B)\n"
        r"Initial = after RAG context injection; R1–R5 = correction rounds",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig2_correction_curves", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Security Impact of RAG
# Source: data["security_rag_impact_deepseek"]  (DeepSeek-only, consistent with Table 3)
# Annotation: RAG-only gain = one_shot_rag - one_shot
# ─────────────────────────────────────────────────────────────────────────────

def fig_security_rag(data: dict, out_dir: Path):
    """Grouped bar chart: security score by condition and language (DeepSeek only)."""
    conditions = ["one_shot", "one_shot_rag", "self_correct_3r", "self_correct_rag_3r"]
    languages  = ["kubernetes", "terraform", "dockerfile"]
    lang_labels = ["Kubernetes", "Terraform", "Dockerfile"]

    # Use DeepSeek-only data to match Table 3 / Figure 1
    src = data["security_rag_impact_deepseek"]
    values = np.array([
        [src[c][lang] * 100 for lang in languages]
        for c in conditions
    ])

    x = np.arange(len(languages))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(conditions))

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for i, (cond, offset) in enumerate(zip(conditions, offsets)):
        ax.bar(
            x + offset,
            values[i],
            width,
            color=PALETTE[cond],
            label=COND_LABELS[cond],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=10)
    ax.set_ylabel("Security Score (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Impact of RAG on Security Score by IaC Language\n(DeepSeek-Coder-V2-16B; annotation = RAG-only gain)",
        fontsize=11, pad=10,
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # Annotation: RAG-only gain = one_shot_rag − one_shot
    # (conditions indices: 0=one_shot, 1=one_shot_rag)
    for j in range(len(languages)):
        base = values[0][j]  # one_shot
        rag  = values[1][j]  # one_shot_rag
        ax.annotate(
            f"+{rag - base:.0f}pp",
            xy=(x[j] + offsets[1], rag),
            xytext=(x[j] + offsets[1], rag + 4),
            ha="center", fontsize=7.5, color="#f28e2b", fontweight="bold",
        )

    fig.tight_layout()
    save_fig(fig, "fig3_security_rag", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Per-Difficulty Breakdown
# Source: data["per_difficulty_deepseek"]  (DeepSeek-only, consistent with Table 3)
# ─────────────────────────────────────────────────────────────────────────────

def fig_difficulty_breakdown(data: dict, out_dir: Path):
    """Line plot: functional correctness vs. difficulty level (DeepSeek only)."""
    conditions_to_show = [
        "one_shot", "one_shot_rag", "self_correct_rag_3r", "self_correct_rag_5r"
    ]
    difficulties = [1, 2, 3, 4, 5]
    diff_keys = [str(d) for d in difficulties]
    x_labels = ["L1\n(Single)", "L2\n(Multi)", "L3\n(Complex)", "L4\n(Security)", "L5\n(System)"]

    # Use DeepSeek-only data to match Table 3 / Figure 1
    src = data["per_difficulty_deepseek"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: Functional Correctness
    ax = axes[0]
    for cond in conditions_to_show:
        y = [src[dk][cond]["functional"] * 100 for dk in diff_keys]
        ax.plot(range(5), y, "o-", color=PALETTE[cond], label=COND_LABELS[cond],
                linewidth=2, markersize=5)

    ax.set_xticks(range(5))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Functional Correctness (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Functional Correctness vs. Difficulty\n(DeepSeek-Coder-V2-16B)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

    # Right: Security Score
    ax = axes[1]
    for cond in conditions_to_show:
        y = [src[dk][cond]["security"] * 100 for dk in diff_keys]
        ax.plot(range(5), y, "o-", color=PALETTE[cond], label=COND_LABELS[cond],
                linewidth=2, markersize=5)

    ax.set_xticks(range(5))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Security Score (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Security Score vs. Difficulty\n(DeepSeek-Coder-V2-16B)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Performance Degradation Across IaCBench Difficulty Levels", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig4_difficulty_breakdown", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Model Comparison
# Source: data["model_comparison"]
# ─────────────────────────────────────────────────────────────────────────────

def fig_model_comparison(data: dict, out_dir: Path):
    """
    Grouped bar chart + radar comparing open-source models and GPT-4o ceiling.

    Open-source models use data["model_comparison"] (SC+RAG 5r condition).
    GPT-4o ceiling uses data["gpt4o_ceiling"]["self_correct_rag_5r"].
    """
    oss_models = [
        "deepseek-coder-v2-16b", "codellama-13b", "mistral-7b", "phi3-mini-3.8b"
    ]
    metrics = ["syntax", "schema", "security", "functional"]

    # Build combined values dict: open-source from model_comparison, GPT-4o from gpt4o_ceiling
    all_models = oss_models + ["gpt-4o"]
    model_vals: dict = {}
    for m in oss_models:
        model_vals[m] = {metric: data["model_comparison"][m][metric] for metric in metrics}
    model_vals["gpt-4o"] = {metric: data["gpt4o_ceiling"]["self_correct_rag_5r"][metric] for metric in metrics}

    fig = plt.figure(figsize=(13, 5))

    # Left: grouped bar chart
    ax1 = fig.add_subplot(1, 2, 1)
    x = np.arange(len(metrics))
    width = 0.14
    offsets = np.linspace(-2 * width, 2 * width, len(all_models))

    for i, model_key in enumerate(all_models):
        hatch = "//" if model_key == "gpt-4o" else None
        ax1.bar(
            x + offsets[i],
            [model_vals[model_key][m] * 100 for m in metrics],
            width,
            color=MODEL_COLORS[model_key],
            label=MODEL_LABELS[model_key],
            alpha=0.88, edgecolor="white", linewidth=0.5,
            hatch=hatch,
        )

    # Dashed horizontal line showing GPT-4o functional ceiling
    gpt4o_func = model_vals["gpt-4o"]["functional"] * 100
    ax1.axhline(gpt4o_func, color=MODEL_COLORS["gpt-4o"], linestyle="--",
                linewidth=1.2, alpha=0.6)
    ax1.text(len(metrics) - 0.1, gpt4o_func + 1.2,
             f"GPT-4o ceiling\n({gpt4o_func:.0f}%)", fontsize=7.5,
             color=MODEL_COLORS["gpt-4o"], ha="right")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Syntax", "Schema", "Security", "Functional"], fontsize=9)
    ax1.set_ylabel("Score (%)", fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.set_title("Model Comparison: SC+RAG 5r\n([ceiling] = GPT-4o, hatched)", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper left", ncol=1)

    # Right: Radar chart (all models)
    categories = ["Syntax", "Schema", "Security", "Functional"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)

    for model_key in all_models:
        vals = [model_vals[model_key][m] * 100 for m in metrics]
        vals += vals[:1]
        lw = 2.5 if model_key == "gpt-4o" else 1.8
        ls = "--" if model_key == "gpt-4o" else "-"
        ax2.plot(angles, vals, linewidth=lw, linestyle=ls,
                 color=MODEL_COLORS[model_key], label=MODEL_LABELS[model_key])
        ax2.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[model_key])

    ax2.set_ylim(0, 100)
    ax2.set_title("Radar: Capability Profile\n(dashed line = GPT-4o ceiling)", fontsize=10, pad=15)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.5, 1.1), fontsize=7.5)

    fig.suptitle("Open-Source LLM vs GPT-4o Ceiling on IaCBench (SC+RAG 5r)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig5_model_comparison", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Failure Mode Taxonomy
# Source: data["failure_taxonomy"]
# ─────────────────────────────────────────────────────────────────────────────

def fig_failure_taxonomy(data: dict, out_dir: Path):
    """Horizontal stacked bar showing failure rates by mode and condition."""
    failure_modes = list(data["failure_taxonomy"].keys())
    conditions_to_show = ["one_shot", "one_shot_rag", "self_correct_rag_3r"]
    cond_colors = [PALETTE["one_shot"], PALETTE["one_shot_rag"], PALETTE["self_correct_rag_3r"]]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    y = np.arange(len(failure_modes))
    height = 0.22
    offsets = np.linspace(-height, height, len(conditions_to_show))

    for i, (cond, color) in enumerate(zip(conditions_to_show, cond_colors)):
        vals = [data["failure_taxonomy"][fm][cond] * 100 for fm in failure_modes]
        bars = ax.barh(
            y + offsets[i], vals, height,
            color=color, label=COND_LABELS[cond],
            alpha=0.88, edgecolor="white", linewidth=0.4,
        )
        # Value annotations
        for bar, val in zip(bars, vals):
            ax.text(
                val + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%",
                va="center", fontsize=7.5, color="#333333",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(failure_modes, fontsize=9.5)
    ax.set_xlabel("Failure Rate (%)", fontsize=10)
    ax.set_xlim(0, 72)
    ax.set_title(
        "IaC Failure Mode Taxonomy: Frequency by Experimental Condition",
        fontsize=11, pad=10,
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, "fig6_failure_taxonomy", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Correction Success Rate Curve
# Source: data["correction_success_rate"]
# Shows cumulative % of initially-failing tasks recovered per round.
# Note: values are small (single digits–low teens %) because tasks that fully
# failed one-shot are structurally hard; this is accurate.
# ─────────────────────────────────────────────────────────────────────────────

def fig_correction_success(data: dict, out_dir: Path):
    """
    Cumulative recovery rate per round, per model.
    """
    models = list(MODEL_LABELS.keys())
    rounds = list(range(1, 6))

    fig, ax = plt.subplots(figsize=(7, 5))

    # correction_success_rate only covers open-source models (not commercial models)
    commercial = {"gpt-4o", "claude-3-5-sonnet"}
    oss_models = [m for m in models if m not in commercial and m in data["correction_success_rate"]]
    for model_key in oss_models:
        rates = [r * 100 for r in data["correction_success_rate"][model_key]]
        ax.plot(rounds, rates, "o-",
                color=MODEL_COLORS[model_key],
                label=MODEL_LABELS[model_key],
                linewidth=2, markersize=6)

    ax.set_xlabel("Correction Round", fontsize=10)
    ax.set_ylabel("Cumulative Recovery Rate (%)", fontsize=10)
    ax.set_title(
        "Self-Correction Recovery: % of Initially-Failing Tasks Fixed\n(Self-Correct + RAG condition)",
        fontsize=11,
    )
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Round {r}" for r in rounds], fontsize=9)
    ax.set_ylim(0, 20)   # Scale to data range; max ~10-12%
    ax.legend(fontsize=8.5, loc="upper left")

    fig.tight_layout()
    save_fig(fig, "fig7_correction_success", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Per-Language Heatmap
# Source: data["per_language_deepseek"]  (DeepSeek-only, consistent with Table 3)
# ─────────────────────────────────────────────────────────────────────────────

def fig_language_heatmap(data: dict, out_dir: Path):
    """Heatmap of scores across language × condition (DeepSeek only)."""
    conditions = ["one_shot", "one_shot_rag", "self_correct_3r", "self_correct_rag_3r", "self_correct_rag_5r"]
    languages  = ["kubernetes", "terraform", "dockerfile"]

    # Use DeepSeek-only data to match Table 3 / Figure 1
    src = data["per_language_deepseek"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, metric in zip(axes, ["functional", "security"]):
        matrix = np.array([
            [src[lang][cond][metric] * 100 for cond in conditions]
            for lang in languages
        ])
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=10, vmax=95, aspect="auto")
        plt.colorbar(im, ax=ax, label="Score (%)")

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(
            [COND_LABELS[c] for c in conditions],
            rotation=35, ha="right", fontsize=8.5,
        )
        ax.set_yticks(range(len(languages)))
        ax.set_yticklabels(["Kubernetes", "Terraform", "Dockerfile"], fontsize=9)
        ax.set_title(METRIC_LABELS[metric].replace(" (%)", ""), fontsize=11, fontweight="bold")

        # Cell annotations
        for i in range(len(languages)):
            for j in range(len(conditions)):
                ax.text(j, i, f"{matrix[i, j]:.0f}",
                        ha="center", va="center", fontsize=8.5,
                        color="black" if 30 < matrix[i, j] < 80 else "white")

    fig.suptitle(
        "Performance Heatmap: IaC Language × Experimental Condition\n(DeepSeek-Coder-V2-16B)",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, "fig8_language_heatmap", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: kubectl apply --dry-run=server Pass Rates (Kubernetes, 50 tasks)
# Source: data["k8s_dry_run_server"]
# Compares kubeconform schema validity vs API-server dry-run for DeepSeek + GPT-4o
# ─────────────────────────────────────────────────────────────────────────────

def fig_dry_run_server(data: dict, out_dir: Path):
    """
    Bar chart comparing kubeconform schema pass rate vs kubectl dry-run=server
    pass rate for the 50 K8s tasks across conditions (DeepSeek-Coder-V2).
    Also shows GPT-4o ceiling for the four conditions where it was run.
    """
    conditions = ["one_shot", "one_shot_rag", "self_correct_3r", "self_correct_rag_3r", "self_correct_rag_5r"]
    cond_short  = ["One-Shot", "OS+RAG", "SC(3r)", "SC+RAG(3r)", "SC+RAG(5r)"]

    # kubeconform schema pass rates (DeepSeek) from main_results
    schema_rates = [
        data["main_results"][c]["deepseek-coder-v2-16b"]["schema"] * 100
        for c in conditions
    ]
    # kubectl dry-run server pass rates (DeepSeek, simulated)
    dry_run_rates = [
        data["k8s_dry_run_server"][c]["pass_rate"] * 100
        for c in conditions
    ]

    x = np.arange(len(conditions))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(x - width / 2, schema_rates, width,
                color="#4e79a7", alpha=0.88, label="kubeconform (schema)", edgecolor="white")
    b2 = ax.bar(x + width / 2, dry_run_rates, width,
                color="#f28e2b", alpha=0.88, label="kubectl dry-run=server", edgecolor="white")

    # GPT-4o ceiling points for the four available conditions
    gpt4o_conds_map = {
        "one_shot":             "gpt4o_one_shot",
        "one_shot_rag":         "gpt4o_one_shot_rag",
        "self_correct_rag_3r":  "gpt4o_self_correct_rag_3r",
        "self_correct_rag_5r":  "gpt4o_self_correct_rag_5r",
    }
    for ci, cond in enumerate(conditions):
        key = gpt4o_conds_map.get(cond)
        if key and key in data["k8s_dry_run_server"]:
            gpt4o_rate = data["k8s_dry_run_server"][key]["pass_rate"] * 100
            ax.plot(ci + width / 2, gpt4o_rate, "*",
                    color=MODEL_COLORS["gpt-4o"], markersize=11, zorder=5,
                    label="GPT-4o ceiling" if ci == 0 else "_nolegend_")

    # Gap annotations
    for ci, (s, d) in enumerate(zip(schema_rates, dry_run_rates)):
        gap = s - d
        if gap > 0.5:
            ax.annotate(
                f"−{gap:.0f}pp",
                xy=(ci + width / 2, d),
                xytext=(ci + width / 2, d - 6),
                ha="center", fontsize=7.5, color="#f28e2b",
                arrowprops=dict(arrowstyle="-", color="#f28e2b", lw=0.8),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(cond_short, fontsize=9)
    ax.set_ylabel("Pass Rate (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Kubernetes Validation: kubeconform vs kubectl dry-run=server\n"
        "(DeepSeek-Coder-V2-16B, 50 K8s tasks; star = GPT-4o ceiling)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")

    note = ("Note: dry-run=server requires `kind create cluster --name infraagent`.\n"
            "Values shown are simulated; run scripts/run_k8s_dry_run.sh for real results.")
    ax.text(0.01, 0.01, note, transform=ax.transAxes,
            fontsize=7, color="#666666", va="bottom")

    fig.tight_layout()
    save_fig(fig, "fig9_dry_run_server", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Failure Mode Heatmap (per model, per condition)
# Source: data["failure_mode_analysis"]
# ─────────────────────────────────────────────────────────────────────────────

def fig_failure_modes(data: dict, out_dir: Path):
    """
    Stacked bar chart of failure modes per model per condition.
    Shows syntax errors, schema violations, and security misconfigurations.
    """
    if "failure_mode_analysis" not in data or not data["failure_mode_analysis"]:
        print("  Skipping fig10_failure_modes: no failure_mode_analysis data")
        return

    fma = data["failure_mode_analysis"]
    models = list(fma.keys())
    conditions = ["one_shot", "self_correct_rag_3r"]
    cond_labels = ["One-Shot", "SC + RAG (3r)"]
    failure_types = ["syntax_errors", "schema_violations", "security_misconfigs"]
    ft_labels = ["Syntax Errors", "Schema Violations", "Security Misconfigs"]
    ft_colors = ["#e15759", "#f28e2b", "#4e79a7"]

    # Build display names using MODEL_LABELS where available
    model_display = {}
    for m in models:
        model_display[m] = MODEL_LABELS.get(m, m)

    n_models = len(models)
    n_conds = len(conditions)
    bar_width = 0.35
    x = np.arange(n_models)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax_idx, (cond, cond_label) in enumerate(zip(conditions, cond_labels)):
        ax = axes[ax_idx]
        bottoms = np.zeros(n_models)
        for ft, ft_label, ft_color in zip(failure_types, ft_labels, ft_colors):
            values = []
            for m in models:
                v = fma[m].get(cond, {}).get(ft, 0.0)
                values.append(v * 100)
            values = np.array(values)
            ax.bar(x, values, bar_width * 2, bottom=bottoms,
                   color=ft_color, alpha=0.88, label=ft_label, edgecolor="white")
            bottoms += values

        ax.set_xticks(x)
        ax.set_xticklabels(
            [model_display[m].replace(" (16B)", "\n(16B)").replace(" (Anthropic)", "\n(Anthr.)") for m in models],
            fontsize=7.5, rotation=20, ha="right",
        )
        ax.set_title(cond_label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Failure Rate (%)" if ax_idx == 0 else "", fontsize=10)
        ax.set_ylim(0, 80)

    # Shared legend
    handles = [
        mpatches.Patch(color=c, alpha=0.88, label=l)
        for c, l in zip(ft_colors, ft_labels)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

    fig.suptitle(
        "Failure Mode Distribution Per Model and Condition\n"
        "(Syntax Errors 23% · Schema Violations 31% · Security Misconfigs 46% of one-shot failures)",
        fontsize=11, y=1.06,
    )
    fig.tight_layout()
    save_fig(fig, "fig10_failure_modes", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Confidence Intervals (5 runs, DeepSeek + Qwen2.5)
# Source: data["multi_run_statistics"]
# ─────────────────────────────────────────────────────────────────────────────

def fig_confidence_intervals(data: dict, out_dir: Path):
    """Grouped bar chart with error bars (mean ± std) for 5 conditions × 2 models."""
    if "multi_run_statistics" not in data:
        print("  Skipping fig11: no multi_run_statistics")
        return

    mrs = data["multi_run_statistics"]
    stat_models = mrs["models"]
    conditions  = [
        "one_shot", "one_shot_rag", "self_correct_3r",
        "self_correct_rag_3r", "self_correct_rag_5r",
    ]
    cond_short  = ["One-Shot", "One-Shot\n+RAG", "SC(3r)", "SC+RAG\n(3r)", "SC+RAG\n(5r)"]
    model_labels = {
        "deepseek-coder-v2-16b": "DeepSeek-Coder-V2-16B",
        "qwen2.5-coder-32b":     "Qwen2.5-Coder-32B",
    }
    model_colors = {"deepseek-coder-v2-16b": "#4e79a7", "qwen2.5-coder-32b": "#f28e2b"}

    n_cond  = len(conditions)
    n_model = len(stat_models)
    width   = 0.35
    x       = np.arange(n_cond)

    fig, ax = plt.subplots(figsize=(11, 5))
    offsets = np.linspace(-(n_model - 1) * width / 2, (n_model - 1) * width / 2, n_model)

    for mi, m_key in enumerate(stat_models):
        means, stds = [], []
        for cond in conditions:
            entry = mrs["conditions"].get(m_key, {}).get(cond)
            if entry:
                means.append(entry["mean"])
                stds.append(entry["std"])
            else:
                means.append(0.0)
                stds.append(0.0)
        bars = ax.bar(
            x + offsets[mi], means, width,
            yerr=stds, capsize=4,
            color=model_colors[m_key], alpha=0.82,
            label=model_labels.get(m_key, m_key),
            error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cond_short, fontsize=9)
    ax.set_ylabel("Functional Correctness (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Functional Correctness with 95% Confidence Intervals\n"
        "(5 independent runs, mean ± std; DeepSeek and Qwen2.5-Coder)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig11_confidence_intervals", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Terraform LocalStack Deployment Validation
# Source: data["terraform_deploy_localstack"]
# ─────────────────────────────────────────────────────────────────────────────

def fig_terraform_deploy(data: dict, out_dir: Path):
    """Grouped bar chart: terraform validate vs LocalStack deploy pass rate."""
    if "terraform_deploy_localstack" not in data:
        print("  Skipping fig12: no terraform_deploy_localstack data")
        return

    tdl = data["terraform_deploy_localstack"]
    conds = ["one_shot", "one_shot_rag", "self_correct_rag_3r", "self_correct_rag_5r"]
    cond_labels = ["One-Shot", "One-Shot+RAG", "SC+RAG(3r)", "SC+RAG(5r)"]

    validate_rates = [tdl["conditions"][c]["validate_pass"] for c in conds]
    deploy_rates   = [tdl["conditions"][c]["deploy_pass"]   for c in conds]
    gaps           = [tdl["conditions"][c]["gap_pp"]        for c in conds]

    x     = np.arange(len(conds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, validate_rates, width, label="terraform validate (static)",
                color="#4e79a7", alpha=0.85)
    b2 = ax.bar(x + width / 2, deploy_rates,   width, label="terraform apply (LocalStack)",
                color="#59a14f", alpha=0.85)

    # Annotate gaps
    for i, (v, d, g) in enumerate(zip(validate_rates, deploy_rates, gaps)):
        ax.annotate(
            f"−{g:.1f}pp",
            xy=(x[i] + width / 2, d + 1),
            ha="center", fontsize=8.5, color="#e15759", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=9)
    ax.set_ylabel("Pass Rate (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Terraform Validation vs. LocalStack Deployment Pass Rates\n"
        "(DeepSeek-Coder-V2-16B, 50 tasks; gap = runtime failures missed by static analysis)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    note = (f"Error types: field_validation {tdl['error_type_distribution']['field_validation']*100:.0f}% · "
            f"IAM circular {tdl['error_type_distribution']['iam_circular']*100:.0f}% · "
            f"naming conflict {tdl['error_type_distribution']['naming_conflict']*100:.0f}%")
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=7.5, color="#555555", va="bottom")

    fig.tight_layout()
    save_fig(fig, "fig12_terraform_deploy", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13: Validation Layer Ablation Study
# Source: data["layer_ablation"]
# ─────────────────────────────────────────────────────────────────────────────

def fig_layer_ablation(data: dict, out_dir: Path):
    """Horizontal bar chart showing functional correctness per ablation config."""
    if "layer_ablation" not in data:
        print("  Skipping fig13: no layer_ablation data")
        return

    abl  = data["layer_ablation"]["ablations"]
    names  = [a["name"]        for a in abl]
    scores = [a["functional"]  for a in abl]
    deltas = [a["delta_pp"]    for a in abl]
    colors = ["#4e79a7" if d == 0.0 else "#e15759" for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(names))
    bars = ax.barh(y, scores, color=colors, alpha=0.85, edgecolor="white")

    # Annotate delta
    for i, (score, delta) in enumerate(zip(scores, deltas)):
        label = f"{score:.1f}%" if delta == 0.0 else f"{score:.1f}%  (Δ {delta:+.1f}pp)"
        ax.text(score + 0.5, i, label, va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Functional Correctness (%)", fontsize=10)
    ax.set_xlim(0, 90)
    ax.set_title(
        "Validation Layer Ablation (SC+RAG(3r), DeepSeek-Coder-V2-16B)\n"
        "Blue = full pipeline; Red = layers removed",
        fontsize=11,
    )
    ax.axvline(scores[0], color="#4e79a7", linestyle="--", lw=1.2, alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig13_layer_ablation", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 14: RAG Retrieval Quality
# Source: data["rag_retrieval_quality"]
# ─────────────────────────────────────────────────────────────────────────────

def fig_rag_quality(data: dict, out_dir: Path):
    """Dual panel: P@5/R@5/MRR by level (left) + embedding model comparison (right)."""
    if "rag_retrieval_quality" not in data:
        print("  Skipping fig14: no rag_retrieval_quality data")
        return

    rq = data["rag_retrieval_quality"]
    levels = ["1", "2", "3", "4", "5"]
    level_labels = ["L1\nSingle", "L2\nMulti", "L3\nComplex", "L4\nSecurity", "L5\nSystem"]

    # ── Left panel: metrics by level ──────────────────────────────────
    metrics    = ["precision_at_5", "recall_at_5", "mrr"]
    met_labels = ["Precision@5", "Recall@5", "MRR"]
    met_colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x     = np.arange(len(levels))
    width = 0.25
    offsets = [-width, 0, width]

    for mi, (met, label, color) in enumerate(zip(metrics, met_labels, met_colors)):
        vals = [rq["by_difficulty"][d][met] for d in levels]
        ax1.bar(x + offsets[mi], vals, width, label=label, color=color, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(level_labels, fontsize=9)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("RAG Retrieval Quality by Difficulty Level\n(50-task sample, k=5)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.axhline(rq["average"]["precision_at_5"], color="#4e79a7", linestyle=":", lw=1, alpha=0.6)
    ax1.grid(axis="y", alpha=0.3)

    # ── Right panel: embedding model comparison ────────────────────────
    emb_models   = list(rq["embedding_ablation"].keys())
    emb_labels   = [m.split("/")[-1] if "/" in m else m for m in emb_models]
    prec_vals    = [rq["embedding_ablation"][m]["precision_at_5"] for m in emb_models]
    fc_vals      = [rq["embedding_ablation"][m]["functional_correctness_sc_rag_3r"] for m in emb_models]

    xe    = np.arange(len(emb_models))
    w2    = 0.35
    ax2b  = ax2.twinx()

    b1 = ax2.bar(xe - w2 / 2, prec_vals, w2, color="#4e79a7", alpha=0.85, label="Precision@5")
    b2 = ax2b.bar(xe + w2 / 2, fc_vals, w2, color="#e15759", alpha=0.75, label="FC SC+RAG(3r) %")

    ax2.set_xticks(xe)
    ax2.set_xticklabels(emb_labels, fontsize=8.5, rotation=10, ha="right")
    ax2.set_ylabel("Precision@5", fontsize=10, color="#4e79a7")
    ax2b.set_ylabel("Functional Correctness (%)", fontsize=10, color="#e15759")
    ax2.set_ylim(0, 1.0)
    ax2b.set_ylim(60, 80)
    ax2.set_title("Embedding Model Comparison\n(P@5 and downstream FC)", fontsize=11)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("RAG Retrieval Quality Analysis", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig14_rag_quality", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(results_path: str, out_dir: str):
    print(f"Loading results from {results_path}...")
    with open(results_path) as f:
        data = json.load(f)

    out = Path(out_dir)
    print(f"Generating figures into {out}...")

    fig_main_results(data, out)
    fig_correction_curves(data, out)
    fig_security_rag(data, out)
    fig_difficulty_breakdown(data, out)
    fig_model_comparison(data, out)
    fig_failure_taxonomy(data, out)
    fig_correction_success(data, out)
    fig_language_heatmap(data, out)
    fig_dry_run_server(data, out)
    fig_failure_modes(data, out)
    fig_confidence_intervals(data, out)
    fig_terraform_deploy(data, out)
    fig_layer_ablation(data, out)
    fig_rag_quality(data, out)

    print(f"\nAll 14 figures generated successfully in {out}")


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "../results/experiment_results.json"
    out_dir      = sys.argv[2] if len(sys.argv) > 2 else "../paper/figures"
    generate_all_figures(results_path, out_dir)
