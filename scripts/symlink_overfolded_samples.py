#!/usr/bin/env python3
"""
scripts/symlink_overfolded_samples.py
=======================================
Symlink minimized models into a flat per-backend directory for quick visual
inspection in ChimeraX, ranked by how much alpha helix they form inside the
**non-MoRF** residues of the disordered domains -- the same criterion as
notebooks/drb2_drb4_domain_analysis.ipynb's "Non-MoRF over-folding filter"
section.

Two modes (`--mode`):

  overfolded  (default) -- the N *most* over-folded models per backend
              (longest helical run, then most helix residues, both
              descending). The worst offenders, for eyeballing the
              hallucinated secondary structure.

  clean                 -- the mirror image: the N *least* over-folded models
              per backend, restricted to those that actually PASS the filter
              (`pct_helix <= --max-pct-helix` AND `max_helix_run <
              --min-long-run` in the non-MoRF disordered residues), sorted
              cleanest-first. The models the filter keeps, for eyeballing
              what "converged" looks like.

Disordered domains: DRB2 (chain A) 189-434, DRB4 (chain B) 151-291. DRB2's
boundary was corrected from the original PROSITE 156-434 call (residues
156-188 are genuinely folded -- see the notebook's fold-upon-binding
section). Genuine fold-upon-binding windows are carved out before the helix
count (disable with `--include-morf`): DRB2 425-434, DRB4 268-284, per that
notebook's verdict table -- note the notebook's own `MORF_WINDOWS` dict has
these two swapped by chain; the assignment here is the evidence-correct one.

Ranking metrics, per model:
    1. longest single contiguous helical run in EITHER disordered domain's
       non-MoRF residues  ("relatively long alpha helix")
    2. total helix residue count across both domains  ("high presence",
       tiebreaker)
%% helix is evaluated per domain (the worse of the two), matching the
notebook's `pct_helix_drb2 > T OR pct_helix_drb4 > T` filter rather than a
length-weighted pool. Helical runs are found per contiguous sub-range, so a
run is never counted as bridging the excised MoRF window.

Only draws from models present in dssp_summary.csv, which was itself only
ever computed for the energy-filter survivors (see dssp_manifest.txt) --
consistent with the notebook, and correctly excludes rosettafold3 if it has
zero valid models.

Usage:
    # worst offenders (unchanged default behaviour)
    python scripts/symlink_overfolded_samples.py \\
        --complex drb2_drb4 --top-n 20 \\
        --out-dir results/drb2_drb4/overfolding_inspection

    # the other way around: the filter's survivors, cleanest first
    python scripts/symlink_overfolded_samples.py \\
        --complex drb2_drb4 --mode clean --top-n 20 \\
        --out-dir results/drb2_drb4/overfolding_inspection_survivors
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

DRB2_DISORDERED = (189, 434)  # corrected from the original 156-434 PROSITE-based call
DRB4_DISORDERED = (151, 291)
CHAIN_DOMAINS = {"A": DRB2_DISORDERED, "B": DRB4_DISORDERED}

# Genuine fold-upon-binding windows (drb2_drb4_domain_analysis.ipynb verdict
# table): DRB2 = chain A, DRB4 = chain B. The notebook's own MORF_WINDOWS dict
# assigns these two to the wrong chains; this is the evidence-correct pairing.
MORF_WINDOWS = {"A": (425, 434), "B": (268, 284)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complex", required=True, help="e.g. drb2_drb4")
    p.add_argument("--results-root", default="results")
    p.add_argument("--mode", choices=("overfolded", "clean"), default="overfolded",
                   help="overfolded = worst offenders (default); clean = filter survivors, cleanest first")
    p.add_argument("--top-n", type=int, default=20,
                   help="models per backend to symlink (0 = all eligible)")
    p.add_argument("--max-pct-helix", type=float, default=20.0,
                   help="clean mode: max %% helix in non-MoRF disordered residues to pass")
    p.add_argument("--min-long-run", type=int, default=12,
                   help="clean mode: a non-MoRF helical run >= this fails the filter")
    p.add_argument("--include-morf", action="store_true",
                   help="do NOT carve out the genuine MoRF windows before counting helix")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def nonmorf_subranges(domain, morf):
    """`domain` (inclusive lo, hi) minus `morf` -> list of contiguous sub-ranges."""
    lo, hi = domain
    m0, m1 = morf
    subs = []
    if lo < m0:
        subs.append((lo, min(m0 - 1, hi)))
    if m1 < hi:
        subs.append((max(m1 + 1, lo), hi))
    return subs or [domain]  # MoRF window entirely outside the domain -> whole domain


def domain_helix_stats(dssp, chain, subranges):
    """Per (fname, cluster): residue count, helix-residue count, and longest
    contiguous helix run over `chain` restricted to `subranges`. Runs are
    computed per sub-range so a helix never bridges an excised gap."""
    in_any = pd.Series(False, index=dssp.index)
    is_chain = dssp["chain"] == chain
    for lo, hi in subranges:
        in_any |= is_chain & dssp["resnum"].between(lo, hi)
    sub = dssp[in_any]

    n_res = sub.groupby(["fname", "cluster"]).size().rename("n_res")
    n_helix = (sub[sub["ss_type"] == "helix"].groupby(["fname", "cluster"]).size()
               .rename("n_helix"))

    max_run = {}
    for lo, hi in subranges:
        s = sub[sub["resnum"].between(lo, hi)].sort_values(["fname", "cluster", "resnum"]).copy()
        if s.empty:
            continue
        s["is_helix"] = s["ss_type"] == "helix"
        grp = s.groupby(["fname", "cluster"], sort=False)["is_helix"]
        s["run_id"] = (s["is_helix"] != grp.shift()).groupby([s["fname"], s["cluster"]]).cumsum()
        runs = s[s["is_helix"]].groupby(["fname", "cluster", "run_id"]).size()
        if runs.empty:
            continue
        for (fn, cl), longest in runs.groupby(level=[0, 1]).max().items():
            max_run[(fn, cl)] = max(max_run.get((fn, cl), 0), int(longest))

    out = pd.concat([n_res, n_helix], axis=1)
    out["n_helix"] = out["n_helix"].fillna(0).astype(int)
    out["max_run"] = [max_run.get(k, 0) for k in out.index]
    return out


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

    exclude_morf = not args.include_morf
    parts = {}
    for chain, domain in CHAIN_DOMAINS.items():
        subs = nonmorf_subranges(domain, MORF_WINDOWS[chain]) if exclude_morf else [domain]
        s = domain_helix_stats(dssp, chain, subs)
        s["pct_helix"] = 100 * s["n_helix"] / s["n_res"]
        parts[chain] = s

    per_model = (
        parts["A"].add_suffix("_drb2")
        .join(parts["B"].add_suffix("_drb4"), how="outer")
        .reset_index()
        .merge(backend_lookup, on=["fname", "cluster"], how="left")
        .dropna(subset=["backend"])
    )
    for c in ["max_run_drb2", "max_run_drb4", "pct_helix_drb2", "pct_helix_drb4",
              "n_helix_drb2", "n_helix_drb4"]:
        per_model[c] = per_model[c].fillna(0)
    per_model["max_run"] = per_model[["max_run_drb2", "max_run_drb4"]].max(axis=1)
    per_model["worst_pct_helix"] = per_model[["pct_helix_drb2", "pct_helix_drb4"]].max(axis=1)
    per_model["total_helix"] = per_model["n_helix_drb2"] + per_model["n_helix_drb4"]

    if args.mode == "overfolded":
        sort_keys, ascending = ["max_run", "total_helix"], False

        def eligible(g):
            return g

        def link_name(rank, row):
            return (f"{rank:02d}_maxhelix{int(row.max_run)}_totalhelix{int(row.total_helix)}"
                    f"_{row.fname}.pdb")
    else:  # clean
        sort_keys, ascending = ["max_run", "total_helix", "worst_pct_helix"], True

        def eligible(g):
            return g[(g["worst_pct_helix"] <= args.max_pct_helix) & (g["max_run"] < args.min_long_run)]

        def link_name(rank, row):
            return (f"{rank:02d}_maxhelix{int(row.max_run)}_pct{row.worst_pct_helix:04.1f}"
                    f"_{row.fname}.pdb")

    out_root = Path(args.out_dir)
    if out_root.exists():
        shutil.rmtree(out_root)  # clean re-generation, not accumulation across runs
    out_root.mkdir(parents=True)

    minimized_root = (results_dir / "minimized").resolve()
    morf_note = "MoRF windows carved out" if exclude_morf else "MoRF windows INCLUDED (--include-morf)"
    print(f"mode={args.mode}  ({morf_note})")
    if args.mode == "clean":
        print(f"pass criterion: pct_helix <= {args.max_pct_helix} AND max_helix_run < {args.min_long_run} "
              f"(non-MoRF residues of DRB2 189-434 + DRB4 151-291)")
    print(f"{len(per_model)} model(s) in dssp_summary.csv, "
          f"{per_model['backend'].nunique()} backend(s)\n")

    for backend in sorted(sel["backend"].dropna().unique()):
        sub = per_model[per_model["backend"] == backend]
        elig = eligible(sub).sort_values(sort_keys, ascending=ascending)
        take = len(elig) if args.top_n in (0, None) else args.top_n
        picks = elig.head(take)

        backend_dir = out_root / backend
        backend_dir.mkdir(parents=True, exist_ok=True)

        n_linked = 0
        for rank, row in enumerate(picks.itertuples(index=False), start=1):
            src = minimized_root / str(row.cluster) / row.fname / f"{row.fname}.pdb"
            if not src.exists():
                continue
            (backend_dir / link_name(rank, row)).symlink_to(src)
            n_linked += 1

        rng = ""
        if len(picks):
            rng = (f"  [max_helix_run {picks['max_run'].min():.0f}-{picks['max_run'].max():.0f}, "
                   f"worst_pct_helix {picks['worst_pct_helix'].min():.1f}-{picks['worst_pct_helix'].max():.1f}]")
        print(f"  {backend:12s}: {n_linked} symlinked  "
              f"(eligible: {len(elig)} / {len(sub)} models){rng}")

    print(f"\nDone: {out_root}")


if __name__ == "__main__":
    main()
