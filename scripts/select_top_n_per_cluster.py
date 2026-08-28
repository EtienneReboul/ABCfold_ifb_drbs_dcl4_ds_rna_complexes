#!/usr/bin/env python3
"""
scripts/select_top_n_per_cluster.py
=====================================
Stage 3b of the ABCfold IFB DRB2/DCL4/dsRNA pipeline: pick the top N models
per pose cluster (scripts/pose_cluster_anchor.py's pose_clusters.csv),
ranked by --rank-by (default ranking_score, falling back to iptm for any
row where a backend didn't report one), and stage their CIFs for
minimize_cif/fix_pdb/run_plip under a flat, Snakemake-wildcard-friendly
layout.

Usage:
    python scripts/select_top_n_per_cluster.py \\
        --pose-clusters       results/rna_ds_dcl4_drb2_drb4/pose_clusters.csv \\
        --abcfold-output-root results/abcfold \\
        --out-root            results/rna_ds_dcl4_drb2_drb4 \\
        --top-n 20 --rank-by ranking_score
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-clusters", required=True)
    p.add_argument("--abcfold-output-root", default="results/abcfold")
    p.add_argument("--out-root", required=True)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--rank-by", default="ranking_score")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.pose_clusters)

    rank_col = args.rank_by
    if rank_col not in df.columns:
        raise ValueError(f"--rank-by {rank_col!r} not a column in {args.pose_clusters}")
    fallback_col = "iptm" if rank_col != "iptm" else "ptm"
    df["_rank_value"] = df[rank_col].fillna(df.get(fallback_col))

    abcfold_output_root = Path(args.abcfold_output_root)
    out_root = Path(args.out_root)
    selected_dir = out_root / "selected"

    rows = []
    for cluster, group in df.groupby("cluster"):
        top = group.sort_values("_rank_value", ascending=False).head(args.top_n)
        cluster_dir = selected_dir / f"cluster_{cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            src = abcfold_output_root / row["cif_path"]
            dest_name = f"rank_{rank:02d}_{row['backend']}_seed{row['seed']}_sample{row['sample_index']}.cif"
            dest = cluster_dir / dest_name
            shutil.copy2(src, dest)
            rows.append({
                "cluster": cluster, "rank": rank, "backend": row["backend"],
                "seed": row["seed"], "sample_index": row["sample_index"],
                "source_cif": str(src), "staged_cif": str(dest),
                rank_col: row[rank_col], "ptm": row.get("ptm"), "iptm": row.get("iptm"),
            })
        print(f"[select] cluster {cluster}: {len(top)}/{len(group)} model(s) selected -> {cluster_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "selected_models.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[select] {len(rows)} model(s) total -> {out_csv}")


if __name__ == "__main__":
    main()
