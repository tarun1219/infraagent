#!/usr/bin/env python3
"""
generate_figures.py
Generates all 8 PDF figures for the IaCBench paper using real evaluation data
from GPT-4o, GPT-4o-mini, Claude Haiku 4.5, and Claude Sonnet 4.5.

Usage:
    python3 analysis/generate_figures.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent
RESULTS_DIR  = SCRIPT_DIR / "results"
FIGURES_DIR  = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# ── Colour palette ────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "GPT-4o":          "#2196F3",
    "GPT-4o-mini":     "#64B5F6",
    "Claude Haiku 4.5":  "#FF9800",
    "Claude Sonnet 4.5": "#E65100",
}
BASELINE_ALPHA = 1.0
RAG_ALPHA      = 0.65

COND_COLORS = {
    "baseline": "#455A64",
    "rag":      "#26A69A",
}

LANG_COLORS = {
    "kubernetes": "#5C6BC0",
    "terraform":  "#26A69A",
    "dockerfile": "#EF5350",
}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    """Return unified data dict keyed by short model name."""

    # GPT-4o
    llm = json.loads((RESULTS_DIR / "llm_eval_results.json").read_text())
    gpt4o = {
        "baseline_stats": llm["baseline_stats"],
        "rag_stats":       llm["rag_stats"],
        "baseline_results": llm["baseline_results"],
        "rag_results":      llm["rag_results"],
    }

    # GPT-4o-mini
    mini = json.loads((RESULTS_DIR / "mini_eval_results.json").read_text())
    gpt4o_mini = {
        "baseline_stats": mini["baseline_stats"],
        "rag_stats":       mini["rag_stats"],
        "baseline_results": mini["baseline_results"],
        "rag_results":      mini["rag_results"],
    }

    # Claude models
    anth = json.loads((RESULTS_DIR / "anthropic_eval_results.json").read_text())
    haiku_key   = "claude-haiku-4-5-20251001"
    sonnet_key  = "claude-sonnet-4-5-20250929"

    def anth_model(key):
        return {
            "baseline_stats": anth["model_stats"][key]["baseline"],
            "rag_stats":       anth["model_stats"][key]["rag"],
            "baseline_results": anth["model_results"][key]["baseline"],
            "rag_results":      anth["model_results"][key]["rag"],
        }

    return {
        "GPT-4o":            gpt4o,
        "GPT-4o-mini":       gpt4o_mini,
        "Claude Haiku 4.5":  anth_model(haiku_key),
        "Claude Sonnet 4.5": anth_model(sonnet_key),
    }


DATA = load_data()
MODELS = list(DATA.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Main Results: Functional correctness baseline vs RAG, all 4 models
# ─────────────────────────────────────────────────────────────────────────────
def fig1_main_results():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x      = np.arange(len(MODELS))
    width  = 0.35

    base_vals = [DATA[m]["baseline_stats"]["pct_functional"] * 100 for m in MODELS]
    rag_vals  = [DATA[m]["rag_stats"]["pct_functional"]      * 100 for m in MODELS]

    bars1 = ax.bar(x - width/2, base_vals, width, label="Baseline",
                   color=[MODEL_COLORS[m] for m in MODELS], alpha=BASELINE_ALPHA, zorder=3)
    bars2 = ax.bar(x + width/2, rag_vals,  width, label="RAG",
                   color=[MODEL_COLORS[m] for m in MODELS], alpha=RAG_ALPHA,      zorder=3,
                   hatch="//", edgecolor="white")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Functional Correctness (%)")
    ax.set_title("IaCBench: Functional Correctness by Model and Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylim(75, 107)
    ax.set_yticks(range(75, 106, 5))

    # Legend for condition
    handles = [
        mpatches.Patch(facecolor="grey", alpha=BASELINE_ALPHA,            label="Baseline"),
        mpatches.Patch(facecolor="grey", alpha=RAG_ALPHA, hatch="//",     label="RAG"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    out = FIGURES_DIR / "fig1_main_results.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Correction Curves: per-attempt functional rate (self-correction)
# ─────────────────────────────────────────────────────────────────────────────
def _attempt_pass_rate(results):
    """Return list of cumulative pass rate after each attempt (up to 3)."""
    max_attempts = max(len(r["attempts"]) for r in results)
    cumulative = []
    for atidx in range(max_attempts):
        passed = sum(
            1 for r in results
            if any(a.get("functional_correct") for a in r["attempts"][:atidx+1])
        )
        cumulative.append(passed / len(results) * 100)
    return cumulative


def fig2_correction_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, cond, label in zip(axes, ["baseline", "rag"], ["Baseline", "RAG"]):
        for model in MODELS:
            results = DATA[model][f"{cond}_results"]
            curve   = _attempt_pass_rate(results)
            xs = list(range(1, len(curve)+1))
            ax.plot(xs, curve, marker="o", linewidth=2,
                    color=MODEL_COLORS[model], label=model)
            ax.annotate(f"{curve[-1]:.1f}%",
                        xy=(xs[-1], curve[-1]),
                        xytext=(2, 0), textcoords="offset points",
                        fontsize=8, color=MODEL_COLORS[model])

        ax.set_title(label)
        ax.set_xlabel("Attempt Number")
        ax.set_xlim(0.8, len(xs) + 0.5)
        ax.set_ylim(60, 103)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"A{i}" for i in xs])

    axes[0].set_ylabel("Cumulative Functional Correctness (%)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Self-Correction Trajectories Across Attempts", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_correction_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Security Scores by IaC Language and Condition
# ─────────────────────────────────────────────────────────────────────────────
def _per_lang_security(results):
    by_lang = {}
    for r in results:
        lang = r["language"]
        by_lang.setdefault(lang, []).append(r.get("final_security_score", 0.0))
    return {lang: np.mean(scores) for lang, scores in by_lang.items()}


def fig3_security_rag():
    langs = ["kubernetes", "terraform", "dockerfile"]
    lang_labels = ["Kubernetes", "Terraform", "Dockerfile"]
    conds = ["baseline", "rag"]
    cond_labels = ["Baseline", "RAG"]

    x     = np.arange(len(langs))
    width = 0.18
    offsets = np.linspace(-(len(MODELS)-1)/2 * width, (len(MODELS)-1)/2 * width, len(MODELS))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, cond, clabel in zip(axes, conds, cond_labels):
        for i, model in enumerate(MODELS):
            per_lang = _per_lang_security(DATA[model][f"{cond}_results"])
            vals = [per_lang.get(l, 0) * 100 for l in langs]
            bars = ax.bar(x + offsets[i], vals, width,
                          color=MODEL_COLORS[model], label=model, alpha=0.88, zorder=3)
        ax.set_title(clabel)
        ax.set_xticks(x)
        ax.set_xticklabels(lang_labels)
        ax.set_ylim(50, 105)
        ax.set_xlabel("IaC Language")

    axes[0].set_ylabel("Mean Security Score (%)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Security Score by IaC Language and Condition", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig3_security_rag.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Difficulty Breakdown: functional correctness by L1–L5
# ─────────────────────────────────────────────────────────────────────────────
def fig4_difficulty_breakdown():
    difficulties = ["1", "2", "3", "4", "5"]
    diff_labels  = [f"L{d}" for d in difficulties]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, cond, clabel in zip(axes, ["baseline", "rag"], ["Baseline", "RAG"]):
        x      = np.arange(len(difficulties))
        width  = 0.18
        offsets = np.linspace(-(len(MODELS)-1)/2*width, (len(MODELS)-1)/2*width, len(MODELS))

        for i, model in enumerate(MODELS):
            by_diff = DATA[model][f"{cond}_stats"]["by_difficulty"]
            vals = [by_diff.get(d, {}).get("pct_functional", 0) * 100 for d in difficulties]
            ax.bar(x + offsets[i], vals, width,
                   color=MODEL_COLORS[model], label=model, alpha=0.88, zorder=3)

        ax.set_title(clabel)
        ax.set_xticks(x)
        ax.set_xticklabels(diff_labels)
        ax.set_ylim(50, 107)
        ax.set_xlabel("Difficulty Level")

    axes[0].set_ylabel("Functional Correctness (%)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Functional Correctness by Difficulty Level", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_difficulty_breakdown.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – Model Comparison (overall + security, side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_model_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Functional correctness (baseline + RAG stacked side-by-side)
    ax = axes[0]
    x = np.arange(len(MODELS))
    w = 0.35
    base_f = [DATA[m]["baseline_stats"]["pct_functional"] * 100 for m in MODELS]
    rag_f  = [DATA[m]["rag_stats"]["pct_functional"]      * 100 for m in MODELS]

    b1 = ax.bar(x - w/2, base_f, w, color=[MODEL_COLORS[m] for m in MODELS],
                alpha=0.9, zorder=3, label="Baseline")
    b2 = ax.bar(x + w/2, rag_f,  w, color=[MODEL_COLORS[m] for m in MODELS],
                alpha=0.55, zorder=3, hatch="//", edgecolor="white", label="RAG")

    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h+0.3, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Functional Correctness (%)")
    ax.set_title("Functional Correctness")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylim(75, 107)
    handles = [mpatches.Patch(color="grey", alpha=0.9, label="Baseline"),
               mpatches.Patch(color="grey", alpha=0.55, hatch="//", label="RAG")]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    # Right: Security score
    ax = axes[1]
    base_s = [DATA[m]["baseline_stats"]["mean_security"] * 100 for m in MODELS]
    rag_s  = [DATA[m]["rag_stats"]["mean_security"]      * 100 for m in MODELS]

    ax.bar(x - w/2, base_s, w, color=[MODEL_COLORS[m] for m in MODELS],
           alpha=0.9, zorder=3)
    ax.bar(x + w/2, rag_s,  w, color=[MODEL_COLORS[m] for m in MODELS],
           alpha=0.55, zorder=3, hatch="//", edgecolor="white")

    ax.set_ylabel("Mean Security Score (%)")
    ax.set_title("Security Score")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylim(75, 102)

    fig.suptitle("IaCBench Multi-Model Comparison", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig5_model_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – Failure Taxonomy: why tasks failed
# ─────────────────────────────────────────────────────────────────────────────
def _classify_failure(r):
    """Return the primary failure reason for a failed task."""
    last = r["attempts"][-1] if r["attempts"] else {}
    if last.get("error"):                        return "API/Timeout Error"
    if not last.get("syntax_valid", True):       return "Syntax Error"
    if not last.get("schema_valid", True):       return "Schema Error"
    sec = last.get("security_score", 1.0)
    if sec < 0.5:                                return "Security Failure"
    return "Other"


def fig6_failure_taxonomy():
    cats = ["Syntax Error", "Schema Error", "Security Failure", "API/Timeout Error", "Other"]
    colors = ["#EF5350", "#FF9800", "#AB47BC", "#78909C", "#26A69A"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, cond, clabel in zip(axes, ["baseline", "rag"], ["Baseline", "RAG"]):
        bottom = np.zeros(len(MODELS))
        x = np.arange(len(MODELS))

        count_matrix = {cat: [] for cat in cats}
        for model in MODELS:
            results = DATA[model][f"{cond}_results"]
            failed  = [r for r in results if not r.get("final_functional")]
            total   = len(results)
            for cat in cats:
                c = sum(1 for r in failed if _classify_failure(r) == cat)
                count_matrix[cat].append(c / total * 100)

        for cat, color in zip(cats, colors):
            vals = count_matrix[cat]
            bars = ax.bar(x, vals, bottom=bottom, label=cat, color=color, alpha=0.88, zorder=3)
            bottom += np.array(vals)

        ax.set_title(f"Failure Taxonomy – {clabel}")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=20, ha="right")
        ax.set_ylabel("Tasks (%)")
        ax.set_ylim(0, 25)

    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Failure Mode Taxonomy by Model", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig6_failure_taxonomy.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 – Correction Success: cumulative recovery after each self-correction
# ─────────────────────────────────────────────────────────────────────────────
def _recovery_curve(results):
    """
    Among initially-failing tasks, what fraction were eventually corrected
    by attempt k?
    """
    initially_failed = [r for r in results if not r["attempts"][0].get("functional_correct")]
    if not initially_failed:
        return []
    max_attempts = max(len(r["attempts"]) for r in initially_failed)
    curve = []
    for k in range(1, max_attempts + 1):
        recovered = sum(
            1 for r in initially_failed
            if any(a.get("functional_correct") for a in r["attempts"][:k+1])
        )
        curve.append(recovered / len(initially_failed) * 100)
    return curve


def fig7_correction_success():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, cond, clabel in zip(axes, ["baseline", "rag"], ["Baseline", "RAG"]):
        for model in MODELS:
            results = DATA[model][f"{cond}_results"]
            curve   = _recovery_curve(results)
            if not curve:
                continue
            xs = list(range(1, len(curve)+1))
            ax.plot(xs, curve, marker="o", linewidth=2,
                    color=MODEL_COLORS[model], label=model)
            ax.annotate(f"{curve[-1]:.0f}%",
                        xy=(xs[-1], curve[-1]),
                        xytext=(2, 0), textcoords="offset points",
                        fontsize=8, color=MODEL_COLORS[model])

        ax.set_title(clabel)
        ax.set_xlabel("Self-Correction Round")
        ax.set_xticks(xs if 'xs' in dir() else [1])
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("Recovered Tasks (%)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Cumulative Self-Correction Recovery Rate", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig7_correction_success.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 – Language × Condition Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig8_language_heatmap():
    langs      = ["kubernetes", "terraform", "dockerfile"]
    lang_lbls  = ["Kubernetes", "Terraform", "Dockerfile"]
    conds      = ["baseline", "rag"]
    cond_lbls  = ["Baseline", "RAG"]

    # Build 4 × 6 matrix (model × lang+cond) – functional correctness
    col_labels = [f"{l}\n{c}" for c in cond_lbls for l in lang_lbls]
    matrix_f   = []
    matrix_s   = []

    for model in MODELS:
        row_f, row_s = [], []
        for cond in conds:
            stats = DATA[model][f"{cond}_stats"]
            for lang in langs:
                row_f.append(stats["by_language"].get(lang, {}).get("pct_functional", 0) * 100)
                # per-lang security from results
                per_lang = _per_lang_security(DATA[model][f"{cond}_results"])
                row_s.append(per_lang.get(lang, 0) * 100)
        matrix_f.append(row_f)
        matrix_s.append(row_s)

    matrix_f = np.array(matrix_f)
    matrix_s = np.array(matrix_s)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, mat, title, vmin in zip(
        axes, [matrix_f, matrix_s],
        ["Functional Correctness (%)", "Security Score (%)"],
        [60, 60]
    ):
        im = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=vmin, vmax=100)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8.5)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS, fontsize=9)
        ax.set_title(title, fontsize=11)
        for i in range(len(MODELS)):
            for j in range(len(col_labels)):
                val = mat[i, j]
                txt_color = "black" if val > 70 else "white"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8.5, color=txt_color, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    fig.suptitle("Performance Heatmap: Language × Condition", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig8_language_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== IaCBench Figure Generator ===")
    print(f"Output directory: {FIGURES_DIR}\n")

    steps = [
        ("fig1_main_results",        fig1_main_results),
        ("fig2_correction_curves",   fig2_correction_curves),
        ("fig3_security_rag",        fig3_security_rag),
        ("fig4_difficulty_breakdown",fig4_difficulty_breakdown),
        ("fig5_model_comparison",    fig5_model_comparison),
        ("fig6_failure_taxonomy",    fig6_failure_taxonomy),
        ("fig7_correction_success",  fig7_correction_success),
        ("fig8_language_heatmap",    fig8_language_heatmap),
    ]

    for name, fn in steps:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. PDFs written to {FIGURES_DIR}")
