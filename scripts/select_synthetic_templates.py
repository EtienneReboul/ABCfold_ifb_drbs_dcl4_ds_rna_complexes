#!/usr/bin/env python3
"""
scripts/select_synthetic_templates.py
=======================================
Pick the synthetic (custom) templates for the AF3+OpenFold3 re-run of the
full DCL4/DRB2/DRB4/dsRNA complex, and generate one
``configs/<target>_synthtmpl_NN.yaml`` per picked template.

Why
---
The full complex ``rna_ds_dcl4_drb2_drb4`` (~2600 tokens) only produced
structures from AlphaFold3, OpenFold3 and RosettaFold3 -- Chai-1, Boltz and
Protenix hit architectural token caps (see
``configs/rna_ds_dcl4_drb2_drb4.yaml`` header). The DCL4-free sub-assembly
``rna_ds_drb2_drb4`` (~900 tokens) ran all six backends, and its Chai-1 /
Boltz predictions produced DRB2/DRB4 conformations that pass the non-MoRF
over-folding filter
(``notebooks/rna_complexes_energy_and_overfolding_filter.ipynb`` ->
``results/rna_ds_drb2_drb4/overfolding_inspection_survivors/_survivors.tsv``).

This script selects the top-N of those survivors by backend-reported ipTM
(restricted to a chosen backend pool, default ``chai1,boltz``), copies each
one's minimized PDB into ``data/synthetic_templates/<target-family>/`` under
a stable name (``tmpl_01.pdb`` ...), and writes a manifest plus one
per-template complex config. ``scripts/inject_synthetic_template.py``
(preprocessing stage) then bakes those PDBs into each re-run's
``fold_input.resolved.json`` as AF3-dialect custom templates on the DRB2 and
DRB4 chains.

Chain-id mapping (source -> target)
-----------------------------------
    role   source rna_ds_drb2_drb4   target rna_ds_dcl4_drb2_drb4[_synthtmpl_*]
    DCL4   -- (absent)               A   (no custom template)
    DRB2   A                         B
    DRB4   B                         C
    dsRNA  C, D                      D, E

Idempotent: safe to re-run after new postprocessing data -- it rewrites the
copied PDBs, the manifest and every generated config from scratch.

Usage (run under envs/postprocessing.yaml -- needs pandas):
    python scripts/select_synthetic_templates.py
    python scripts/select_synthetic_templates.py --backends chai1,boltz --top-n 3
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

FNAME_RE = re.compile(r"^rank_(\d+)_(?P<backend>.+)_seed(?P<seed>\d+)_sample(?P<sample>[0-9.]+)$")

# target chain (in the DCL4 complex) -> chain id to pull from the template PDB
# (which is a rna_ds_drb2_drb4 model: A=DRB2, B=DRB4).
DEFAULT_MAPPING = {"B": "A", "C": "B"}
DEFAULT_STRIP = ["B", "C"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source-complex", default="rna_ds_drb2_drb4",
                   help="complex whose over-folding survivors are the template pool")
    p.add_argument("--donor-complex", default="rna_ds_dcl4_drb2_drb4",
                   help="complex whose MSAs / DCL4 templates the re-run reuses "
                        "(same sequences as the target family) -- its "
                        "fold_input.resolved.json is snapshotted into --out-dir so "
                        "the re-runs are pinned to that exact MSA, decoupled from "
                        "any later re-resolution of the baseline")
    p.add_argument("--fold-inputs-dir", default="data/fold_inputs",
                   help="where <donor-complex>/fold_input.resolved.json lives")
    p.add_argument("--target-family", default="rna_ds_dcl4_drb2_drb4_synthtmpl",
                   help="prefix for the generated complexes/configs "
                        "(-> <family>_01, <family>_02, ...)")
    p.add_argument("--template-source-config",
                   default="configs/rna_ds_dcl4_drb2_drb4.yaml",
                   help="config whose sequences/anchor/plip blocks are copied verbatim "
                        "into each generated config")
    p.add_argument("--backends", default="chai1,boltz",
                   help="comma-separated backend pool the templates are drawn from")
    p.add_argument("--top-n", type=int, default=3,
                   help="number of templates / generated configs")
    p.add_argument("--results-root", default="results")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--out-dir", default=None,
                   help="default: data/synthetic_templates/<target-family>")
    return p.parse_args()


def load_survivors_with_iptm(results_dir: Path) -> pd.DataFrame:
    surv_tsv = results_dir / "overfolding_inspection_survivors" / "_survivors.tsv"
    sel_csv = results_dir / "selected_models.csv"
    if not surv_tsv.exists():
        raise SystemExit(f"{surv_tsv} not found -- run scripts/symlink_filtered_survivors.py "
                         "for the source complex first.")
    if not sel_csv.exists():
        raise SystemExit(f"{sel_csv} not found -- run workflows/postprocessing/Snakefile "
                         "for the source complex first.")

    surv = pd.read_csv(surv_tsv, sep="\t")
    sel = pd.read_csv(sel_csv)
    sel["fname"] = sel["staged_cif"].apply(lambda p: Path(p).stem)

    merged = surv.merge(
        sel[["fname", "cluster", "backend", "seed", "sample_index",
             "iptm", "ptm", "ranking_score"]],
        on=["fname", "cluster", "backend"], how="left", validate="one_to_one",
    )
    missing = merged["iptm"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} survivor(s) had no matching row in {sel_csv} "
                         "-- survivors.tsv / selected_models.csv are out of sync.")
    return merged


def resolve_minimized_pdb(results_dir: Path, row) -> Path:
    pdb = results_dir / "minimized" / str(int(row.cluster)) / row.fname / f"{row.fname}.pdb"
    if not pdb.exists():
        raise SystemExit(f"minimized PDB missing: {pdb}")
    return pdb


def build_config(src_cfg: dict, name: str, args, template_pdb: Path,
                 donor_snapshot: Path) -> dict:
    """A copy of src_cfg's structural blocks + the synthtmpl-only blocks."""
    keep = ["sequences", "anchor_chain", "partner_chains",
            "plip", "plip_rna_ligands", "plip_drb2_drb4"]
    out = {"name": name}
    for k in keep:
        if k in src_cfg:
            out[k] = src_cfg[k]
    out["models"] = ["alphafold3", "openfold3"]
    out["synthetic_template"] = {
        "msa_donor_complex": args.donor_complex,          # provenance only
        "donor_resolved_json": donor_snapshot.as_posix(),  # what's actually read
        "template_source_complex": args.source_complex,
        "template_pdb": template_pdb.as_posix(),
        "mapping": dict(DEFAULT_MAPPING),      # target chain : template-PDB chain
        "strip_colabfold_templates": list(DEFAULT_STRIP),
    }
    return out


CONFIG_HEADER = """\
# ─────────────────────────────────────────────────────────────────────────────
# GENERATED by scripts/select_synthetic_templates.py -- do not edit by hand.
# Re-run that script to regenerate (selection is top-{top_n} survivors of
# {source} by ipTM, backend pool: {backends}).
#
# {name}: AF3 + OpenFold3 re-run of the full DCL4/DRB2/DRB4/dsRNA complex with
# a Chai-1/Boltz DRB2+DRB4 conformation injected as a custom template (see the
# synthetic_template: block below; consumed by
# scripts/inject_synthetic_template.py at the preprocessing stage).
#   template model : {fname}  (backend={backend}, ipTM={iptm:.3f})
# Structural blocks (sequences, anchor_chain, partner_chains, plip*) are copied
# verbatim from {src_cfg}.
# ─────────────────────────────────────────────────────────────────────────────
"""


def main():
    args = parse_args()
    results_dir = Path(args.results_root) / args.source_complex
    out_dir = Path(args.out_dir) if args.out_dir else Path("data/synthetic_templates") / args.target_family
    configs_dir = Path(args.configs_dir)
    src_cfg_path = Path(args.template_source_config)
    src_cfg = yaml.safe_load(src_cfg_path.read_text())
    backend_pool = [b.strip() for b in args.backends.split(",") if b.strip()]

    df = load_survivors_with_iptm(results_dir)
    pool = df[df["backend"].isin(backend_pool)].copy()
    if pool.empty:
        raise SystemExit(f"no survivors from backends {backend_pool} in {results_dir}")
    pool = pool.sort_values(["iptm", "ranking_score", "ptm"], ascending=False)
    picks = pool.head(args.top_n).reset_index(drop=True)
    if len(picks) < args.top_n:
        print(f"WARNING: only {len(picks)} survivor(s) available in the backend pool "
              f"(requested top-{args.top_n})", file=sys.stderr)

    donor_src = Path(args.fold_inputs_dir) / args.donor_complex / "fold_input.resolved.json"
    if not donor_src.exists():
        raise SystemExit(f"{donor_src} not found -- run workflows/preprocessing/Snakefile "
                         f"for the donor complex '{args.donor_complex}' first.")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    donor_snapshot = out_dir / "donor_resolved.json"
    shutil.copyfile(donor_src, donor_snapshot)
    print(f"[select] donor MSA snapshot: {donor_src} -> {donor_snapshot}\n")

    manifest_rows = []
    print(f"[select] source={args.source_complex}  backend pool={backend_pool}  "
          f"picking top {len(picks)} by ipTM\n")
    for i, row in enumerate(picks.itertuples(index=False), start=1):
        src_pdb = resolve_minimized_pdb(results_dir, row)
        dst_pdb = out_dir / f"tmpl_{i:02d}.pdb"
        shutil.copyfile(src_pdb, dst_pdb)

        name = f"{args.target_family}_{i:02d}"
        cfg = build_config(src_cfg, name, args, dst_pdb, donor_snapshot)
        cfg_path = configs_dir / f"{name}.yaml"
        header = CONFIG_HEADER.format(
            top_n=args.top_n, source=args.source_complex, backends=args.backends,
            name=name, fname=row.fname, backend=row.backend, iptm=row.iptm,
            src_cfg=src_cfg_path.as_posix(),
        )
        body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False,
                              width=10 ** 9, allow_unicode=True)
        cfg_path.write_text(header + "\n" + body)

        manifest_rows.append(dict(
            rank_idx=i, config=name, fname=row.fname, backend=row.backend,
            cluster=int(row.cluster), seed=int(row.seed), sample_index=row.sample_index,
            iptm=row.iptm, ptm=row.ptm, ranking_score=row.ranking_score,
            src_pdb=src_pdb.as_posix(), dst_pdb=dst_pdb.as_posix(),
        ))
        print(f"  tmpl_{i:02d}  {row.fname:42s} {row.backend:7s} "
              f"ipTM={row.iptm:.3f} ptm={row.ptm:.3f}  -> {cfg_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "templates_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"\n[select] manifest -> {manifest_path}")
    print(f"[select] {len(picks)} config(s) written under {configs_dir}/")
    print("\nNext: add these to config.yaml's complexes: list, then run "
          "workflows/preprocessing/Snakefile.")


if __name__ == "__main__":
    main()
