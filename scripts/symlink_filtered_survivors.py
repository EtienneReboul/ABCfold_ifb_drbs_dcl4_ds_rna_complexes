#!/usr/bin/env python3
"""
scripts/symlink_filtered_survivors.py
=======================================
Symlink the models that **pass** the energy + non-MoRF over-folding filter
(notebooks/rna_complexes_energy_and_overfolding_filter.ipynb) into a flat
per-backend directory for quick visual inspection in ChimeraX, cleanest
first.

Single source of truth is that notebook's
``filtered_models.csv``:

    results/<complex>/figures/domain_analysis/energy_overfolding_filter/filtered_models.csv

so this script is just a projection of it -- re-run the notebook after new
AIUpred/ANCHOR2 data or a threshold/region change, then re-run this script
(or the notebook cell that calls it) and the symlink tree is regenerated
from scratch: newly-passing models appear, no-longer-passing ones vanish.

Companion to scripts/symlink_overfolded_samples.py (which works off
dssp_summary.csv + a hard-coded DRB2/DRB4 rule for the *binary* drb2_drb4
complex). This one is region-agnostic: it reads whatever regions the CSV
carries (`pct_helix_net::<region>` / `max_helix_run_net::<region>` columns),
so it works for both rna_ds_drb2_drb4 and rna_ds_dcl4_drb2_drb4.

Layout written (mirrors the existing overfolding_inspection_survivors/ dirs):

    <out-dir>/<backend>/NN_maxhelix<worst_run>_pct<worst_pct>_<fname>.pdb -> minimized/<cluster>/<fname>/<fname>.pdb
    <out-dir>/_survivors.tsv     (manifest: fname, cluster, backend, worst_pct, worst_run, per-region stats)

Usage:
    # after re-running the notebook
    python scripts/symlink_filtered_survivors.py --complex rna_ds_dcl4_drb2_drb4
    python scripts/symlink_filtered_survivors.py --complex rna_ds_drb2_drb4 --top-n 20

    # also dump the N worst offenders into the sibling overfolding_inspection/ dir
    python scripts/symlink_filtered_survivors.py --complex rna_ds_drb2_drb4 --also-worst 20
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

PCT_PREFIX = "pct_helix_net::"
RUN_PREFIX = "max_helix_run_net::"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complex", required=True, help="e.g. rna_ds_dcl4_drb2_drb4")
    p.add_argument("--results-root", default="results")
    p.add_argument("--filtered-csv", default=None,
                   help="override path to filtered_models.csv "
                        "(default: results/<complex>/figures/domain_analysis/"
                        "energy_overfolding_filter/filtered_models.csv)")
    p.add_argument("--out-dir", default=None,
                   help="default: results/<complex>/overfolding_inspection_survivors")
    p.add_argument("--top-n", type=int, default=0,
                   help="survivors per backend to link, cleanest first (0 = all)")
    p.add_argument("--also-worst", type=int, default=0, metavar="N",
                   help="additionally link the N most over-folded models per backend "
                        "into results/<complex>/overfolding_inspection/ (0 = skip)")
    p.add_argument("--source", choices=("minimized-pdb", "staged-cif"),
                   default="minimized-pdb",
                   help="what to point the symlinks at (default: the minimized .pdb, "
                        "as used for DSSP / ChimeraX)")
    return p.parse_args()


def region_cols(df):
    pct = [c for c in df.columns if c.startswith(PCT_PREFIX)]
    run = [c for c in df.columns if c.startswith(RUN_PREFIX)]
    regions = sorted({c[len(PCT_PREFIX):] for c in pct} & {c[len(RUN_PREFIX):] for c in run})
    if not regions:
        raise SystemExit("no per-region pct_helix_net::/max_helix_run_net:: columns in the CSV")
    return regions


def add_worst(df, regions):
    df = df.copy()
    df["worst_pct"] = df[[PCT_PREFIX + r for r in regions]].max(axis=1).fillna(0.0)
    df["worst_run"] = df[[RUN_PREFIX + r for r in regions]].max(axis=1).fillna(0).astype(int)
    return df


def src_path(row, results_dir, mode):
    if mode == "staged-cif":
        return results_dir / "selected" / f"cluster_{int(row.cluster)}" / f"{row.fname}.cif"
    return results_dir / "minimized" / str(int(row.cluster)) / row.fname / f"{row.fname}.pdb"


def link_group(df, out_root, results_dir, mode, top_n, cleanest_first, name_fn):
    if out_root.exists():
        shutil.rmtree(out_root)               # clean regeneration, never accumulate
    out_root.mkdir(parents=True)
    ext = ".cif" if mode == "staged-cif" else ".pdb"
    total = 0
    for backend in sorted(df["backend"].dropna().unique()):
        g = df[df["backend"] == backend].sort_values(
            ["worst_run", "worst_pct", "fname"], ascending=[cleanest_first, cleanest_first, True])
        if top_n:
            g = g.head(top_n)
        (out_root / backend).mkdir(parents=True, exist_ok=True)
        n = 0
        for rk, row in enumerate(g.itertuples(index=False), start=1):
            src = src_path(row, results_dir, mode)
            if not src.exists():
                print(f"  ! missing source, skipped: {src}")
                continue
            (out_root / backend / (name_fn(rk, row) + ext)).symlink_to(src.resolve())
            n += 1
        total += n
        print(f"  {backend:12s}: {n:3d} linked  (of {len(df[df['backend'] == backend])} in group)")
    return total


def main():
    args = parse_args()
    results_dir = Path(args.results_root) / args.complex
    csv = Path(args.filtered_csv) if args.filtered_csv else (
        results_dir / "figures" / "domain_analysis" / "energy_overfolding_filter"
        / "filtered_models.csv")
    if not csv.exists():
        raise SystemExit(
            f"{csv} not found -- run notebooks/rna_complexes_energy_and_overfolding_filter.ipynb "
            f"(the export_survivors() cell) for '{args.complex}' first.")

    df = pd.read_csv(csv)
    regions = region_cols(df)
    df = add_worst(df, regions)

    survivors = df[df["keep"].astype(bool)].copy()
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "overfolding_inspection_survivors"

    print(f"[{args.complex}] {len(survivors)} survivor(s) / {len(df)} energy-survivors  "
          f"(regions: {', '.join(regions)})")
    print(f"[survivors] -> {out_dir}")
    n = link_group(
        survivors, out_dir, results_dir, args.source, args.top_n, cleanest_first=True,
        name_fn=lambda rk, row: f"{rk:02d}_maxhelix{row.worst_run}_pct{row.worst_pct:04.1f}_{row.fname}")

    manifest_cols = (["fname", "cluster", "backend", "worst_pct", "worst_run"]
                     + [PCT_PREFIX + r for r in regions] + [RUN_PREFIX + r for r in regions])
    (survivors.sort_values(["backend", "worst_run", "worst_pct"])[manifest_cols]
     .to_csv(out_dir / "_survivors.tsv", sep="\t", index=False))
    print(f"[survivors] manifest -> {out_dir / '_survivors.tsv'}")

    if args.also_worst:
        worst_dir = results_dir / "overfolding_inspection"
        overfolded = df[~df["keep"].astype(bool)].copy()
        print(f"\n[overfolded] {len(overfolded)} flagged  -> {worst_dir}")
        link_group(
            overfolded, worst_dir, results_dir, args.source, args.also_worst, cleanest_first=False,
            name_fn=lambda rk, row: f"{rk:02d}_maxhelix{row.worst_run}_pct{row.worst_pct:04.1f}_{row.fname}")

    print(f"\nDone: {n} survivor symlink(s) under {out_dir}")


if __name__ == "__main__":
    main()
