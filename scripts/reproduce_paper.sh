#!/usr/bin/env bash
# Full paper reproduction script.
# Regenerates all figures and statistics from the pre-computed results file.
#
# For live evaluation (requires Ollama + configured models):
#   bash scripts/reproduce_paper.sh --live
#
# For figure/statistics reproduction only (no model inference):
#   bash scripts/reproduce_paper.sh
set -euo pipefail

LIVE_MODE=false
RESULTS_FILE="results/experiment_results.json"

for arg in "$@"; do
    case "$arg" in
        --live) LIVE_MODE=true ;;
    esac
done

echo "========================================================"
echo "  InfraAgent — Paper Reproduction"
echo "  Mode: $([ "$LIVE_MODE" = true ] && echo LIVE || echo FIGURES-ONLY)"
echo "========================================================"

# ── Step 1: Install dependencies ────────────────────────────────────────────
echo ""
echo "[1/5] Installing Python dependencies..."
pip install -q -r requirements.txt

# ── Step 2: Optionally run live experiments ──────────────────────────────────
if [ "$LIVE_MODE" = true ]; then
    echo ""
    echo "[2/5] Running experiments (this will take hours)..."
    echo "      Requires: Ollama running with configured models"
    python scripts/run_experiments.py --condition all --model deepseek --runs 5
else
    echo ""
    echo "[2/5] Skipping live experiments (using pre-computed results)."
    echo "      Use --live to run actual model inference."
fi

# ── Step 3: Simulate/regenerate results data ─────────────────────────────────
echo ""
echo "[3/5] Regenerating experiment_results.json..."
python scripts/simulate_results.py
echo "      → results/experiment_results.json"

# ── Step 4: Generate all figures ─────────────────────────────────────────────
echo ""
echo "[4/5] Generating all 14 paper figures..."
python scripts/generate_figures.py \
    --results "${RESULTS_FILE}" \
    --out results/figures/
echo "      → results/figures/fig{1..14}.{pdf,png}"

# ── Step 5: Compute statistics ───────────────────────────────────────────────
echo ""
echo "[5/5] Computing statistical significance tests..."
python scripts/compute_statistics.py

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Reproduction complete."
echo ""
echo "  Figures:    results/figures/"
echo "  Raw data:   results/experiment_results.json"
echo "  Statistics: printed above"
echo ""
echo "  To compile the paper:"
echo "    cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
echo "========================================================"
