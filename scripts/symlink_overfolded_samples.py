#!/usr/bin/env python3
"""
scripts/symlink_overfolded_samples.py
=======================================
Pick the N most over-folded models per backend (highest presence of long
alpha helix inside the disordered domains -- see notebooks/
drb2_drb4_domain_analysis.ipynb's over-folding investigation) and symlink
them into a flat per-backend directory for quick visual inspection in
ChimeraX.

Ranking: per model, over both disordered domains (DRB2 189-434, DRB4
151-291) combined --

DRB2's boundary was corrected from the original PROSITE-based 156-434 call:
residues 156-188 turned out to be genuinely folded (near-universal helix
across all 5 surviving backends, plus a low AIUpred disorder score, both
transitioning sharply at 189, not 156) rather than a fold-upon-binding
event -- see notebooks/drb2_drb4_domain_analysis.ipynb's fold-upon-binding
cross-validation section for the full evidence. DRB4's boundary is
unrevised.
    1. longest single contiguous helical run (primary key -- "relatively
       long alpha helix"), then
    2. total helix residue count (tiebreaker -- "high presence")
both descending, so the top picks are both long AND pervasive, not just one
short-but-technically-longest fluke.

Only draws from models that survived the energy filter (dssp_summary.csv
was itself only ever computed for that set -- see dssp_manifest.txt) --
consistent with the notebook, and correctly excludes rosettafold3 here too
if it has zero valid models.

Usage:
    python scripts/symlink_overfolded_samples.py \\
        --complex drb2_drb4 --top-n 20 \\
        --out-dir results/drb2_drb4/overfolding_inspection
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

DRB2_DISORDERED = (189, 434)  # corrected from the original 156-434 PROSITE-based call
DRB4_DISORDERED = (151, 291)
CHAIN_DOMAINS = {"A": DRB2_DISORDERED, "B": DRB4_DISORDERED}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--complex", required=True, help="e.g. drb2_drb4")
    p.add_argument("--results-root", default="results")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def helix_runs(dssp, chain, resnum_range):
    lo, hi = resnum_range
    sub = dssp[(dssp["chain"] == chain) & (dssp["resnum"].between(lo, hi))].copy()
    sub = sub.sort_values(["fname", "cluster", "resnum"])
    sub["is_helix"] = sub["ss_type"] == "helix"
    grp = sub.groupby(["fname", "cluster"], sort=False)["is_helix"]
    sub["run_id"] = (sub["is_helix"] != grp.shift()).groupby([sub["fname"], sub["cluster"]]).cumsum()
    helix_only = sub[sub["is_helix"]]
    return (
        helix_only.groupby(["fname", "cluster", "run_id"])
        .agg(length=("resnum", "size"))
        .reset_index()
    )


def main():
    args = parse_args()
    results_dir = Path(args.results_root) / args.complex

    dssp_csv = results_dir / "dssp_summary.csv"
    if not dssp_csv.exists():
        raise SystemExit(
            f"{dssp_csv} not found -- run scripts/dssp_summary.py first "
            f"(see notebooks/{args.complex}_domain_analysis.ipynb's over-folding section)."
        )
    dssp = pd.read_csv(dssp_csv)
    dssp["cluster"] = dssp["cluster"].astype(int)

    sel = pd.read_csv(results_dir / "selected_models.csv")
    sel["fname"] = sel["staged_cif"].apply(lambda p: Path(p).stem)
    backend_lookup = sel[["fname", "cluster", "backend"]].drop_duplicates()

    all_runs = pd.concat(
        [helix_runs(dssp, chain, rng) for chain, rng in CHAIN_DOMAINS.items()],
        ignore_index=True,
    )

    # Per-model summary: longest single run (primary rank key) + total helix
    # residues across both domains (tiebreaker), then backend.
    per_model = (
        all_runs.groupby(["fname", "cluster"])
        .agg(max_helix_run=("length", "max"), total_helix_residues=("length", "sum"))
        .reset_index()
        .merge(backend_lookup, on=["fname", "cluster"], how="left")
        .dropna(subset=["backend"])
    )

    out_root = Path(args.out_dir)
    if out_root.exists():
        shutil.rmtree(out_root)  # clean re-generation, not accumulation across runs
    out_root.mkdir(parents=True)

    minimized_root = (results_dir / "minimized").resolve()

    print(f"{len(per_model)} model(s) with >=1 helical run in a disordered domain, "
          f"across {per_model['backend'].nunique()} backend(s)")

    for backend in sorted(sel["backend"].dropna().unique()):
        sub = per_model[per_model["backend"] == backend]
        top = sub.sort_values(
            ["max_helix_run", "total_helix_residues"], ascending=False
        ).head(args.top_n)

        backend_dir = out_root / backend
        backend_dir.mkdir(parents=True, exist_ok=True)

        for rank, row in enumerate(top.itertuples(index=False), start=1):
            src = minimized_root / str(row.cluster) / row.fname / f"{row.fname}.pdb"
            if not src.exists():
                continue
            link_name = f"{rank:02d}_maxhelix{int(row.max_helix_run)}_totalhelix{int(row.total_helix_residues)}_{row.fname}.pdb"
            (backend_dir / link_name).symlink_to(src)

        print(f"  {backend:12s}: {len(top)} / {args.top_n} symlinked "
              f"(available candidates: {len(sub)})" +
              (f"  [max_helix_run range {top['max_helix_run'].min():.0f}-{top['max_helix_run'].max():.0f}]"
               if len(top) else "  -- no models with any helical run in the disordered domains"))

    print(f"\nDone: {out_root}")


if __name__ == "__main__":
    main()
