#!/usr/bin/env python3
"""
Wrapper script to regenerate all paper figures from experiment_results.json.

Usage:
  python scripts/plot_figures.py
  python scripts/plot_figures.py --results results/experiment_results.json --out results/figures/
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all paper figures")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/experiment_results.json"),
        help="Path to experiment_results.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Re-run simulate_results.py before plotting (default: True)",
    )
    args = parser.parse_args()

    if args.simulate:
        logger.info("Regenerating experiment_results.json from simulate_results.py ...")
        ret = subprocess.run(
            [sys.executable, "scripts/simulate_results.py"],
            capture_output=False,
        )
        if ret.returncode != 0:
            logger.error("simulate_results.py failed")
            sys.exit(1)

    if not args.results.exists():
        logger.error(f"Results file not found: {args.results}")
        sys.exit(1)

    logger.info(f"Generating figures from {args.results} → {args.out}")
    ret = subprocess.run(
        [sys.executable, "scripts/generate_figures.py",
         "--results", str(args.results),
         "--out",     str(args.out)],
        capture_output=False,
    )
    if ret.returncode != 0:
        logger.error("generate_figures.py failed")
        sys.exit(1)

    # List generated files
    out_dir = args.out
    if out_dir.exists():
        pdfs = sorted(out_dir.glob("*.pdf"))
        pngs = sorted(out_dir.glob("*.png"))
        logger.info(f"Generated {len(pdfs)} PDFs and {len(pngs)} PNGs in {out_dir}/")
        for f in pdfs:
            logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
