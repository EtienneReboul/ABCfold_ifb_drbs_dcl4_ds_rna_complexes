#!/usr/bin/env python3
"""
aggregate_summaries.py
======================
Merges all per-model summary.csv files (produced by pliparser) that belong to
a single protein into one CSV, tagging each row with the replica and model it
came from.

Expected path structure for each input file:
    {output_dir}/{protein}/{replica}/{model}_report/csv/summary.csv

So walking upward from the file:
    parts[-1] = summary.csv
    parts[-2] = csv
    parts[-3] = {model}_report      → strip "_report" suffix → model name
    parts[-4] = {replica}
    parts[-5] = {protein}           (not added as column; same for all rows)

Usage (called automatically by Snakemake, but also works standalone):
    python scripts/aggregate_summaries.py \
        --output results/proteinA/all_replica_summary.csv \
        results/proteinA/replica_1/AF3_report/csv/summary.csv \
        results/proteinA/replica_2/AF3_report/csv/summary.csv \
        results/proteinA/replica_1/RoseTTAFold_report/csv/summary.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-model pliparser summary CSVs for one protein."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths to individual summary.csv files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the merged output CSV.",
    )
    return parser.parse_args()


def extract_metadata(path: Path) -> tuple[str, str]:
    """
    Derive (replica, model) from the file path.

    Expected structure (from the file upward):
        summary.csv  ← parts[-1]
        csv/         ← parts[-2]
        {model}_report/  ← parts[-3]
        {replica}/   ← parts[-4]
        {protein}/   ← parts[-5]  (not used here; same within one run)
    """
    parts = path.parts
    if len(parts) < 5:
        raise ValueError(
            f"Cannot extract metadata from path '{path}'. "
            "Expected at least 5 path components: "
            ".../protein/replica/model_report/csv/summary.csv"
        )
    model_report_dir = parts[-4]   # e.g. "AF3_report"
    replica          = parts[-5]   # e.g. "replica_1"

    # Strip the "_report" suffix to recover the clean model name
    model = model_report_dir.removesuffix("_report")

    return replica, model


def load_summary(path: Path) -> pd.DataFrame:
    """Load a single summary CSV and prepend replica / model metadata columns."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARNING] Could not read '{path}': {exc}", file=sys.stderr)
        return pd.DataFrame()

    if df.empty:
        print(f"[WARNING] '{path}' is empty – skipping.", file=sys.stderr)
        return pd.DataFrame()

    replica, model = extract_metadata(path)

    # Prepend so these are always the first two columns
    df.insert(0, "replica", replica)
    df.insert(1, "model",   model)

    return df


def main() -> None:
    args  = parse_args()
    paths = [Path(p) for p in args.inputs]

    frames = []
    for p in paths:
        df = load_summary(p)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("[ERROR] No valid summary CSVs could be loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    aggregated = pd.concat(frames, ignore_index=True)

    # Sort for readability: replica first, then model
    if {"replica", "model"}.issubset(aggregated.columns):
        aggregated = aggregated.sort_values(["replica", "model"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(out_path, index=False)

    print(
        f"[OK] Aggregated {len(frames)} files → {out_path} "
        f"({len(aggregated)} rows, {len(aggregated.columns)} columns)."
    )


if __name__ == "__main__":
    main()
