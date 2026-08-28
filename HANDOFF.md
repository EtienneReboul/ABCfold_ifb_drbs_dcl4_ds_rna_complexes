# Handoff prompt — paste this into a new Claude Code session opened on this project

I'm continuing work on `ABCfold_ifb_drbs_dcl4_ds_rna_complexes`. Read `README.md`
first for the full pipeline overview, then read this note for context the
README doesn't cover.

## What this project is

A rebuild of `../ab_initio_modelling_drbs_dcl4_ds_rna_complexes` (which
predicted DCL4/DRB2/DRB4/dsRNA complexes by manually submitting AF3-webserver
replicas one at a time). This rebuild automates input prep + inference using
the same protocol as the sibling `../../NPF-ab-initio-modelling/ABCfold_NPF_pipeline`
project: ABCfold (https://github.com/rigdenlab/ABCFold) launches AlphaFold3,
Boltz-2, Chai-1, OpenFold3, Protenix and RosettaFold3 **together** per complex
on the IFB cluster — not AF3-only resampling. (An earlier draft of this repo
briefly used the wrong sibling — `AF3_NPF_pipeline`, single-backend — as the
template; that was corrected before anything downstream got built on it,
`abcfold_backends.py` / `compress_abcfold_metadata.py` / `submit_abcfold.sh`
are all genuinely multi-backend.)

Two complexes only (a deliberately simplified subset of the old project's
six — see README's "Complexes modelled" table): `drb2_drb4` (binary, anchor
= DRB2 itself) and `rna_ds_dcl4_drb2_drb4` (5 chains, anchor = DCL4).

## Current state: scaffold complete, nothing has actually run yet

Every file in `configs/`, `workflows/`, `scripts/`, `envs/` exists and is
syntax-checked (YAML parses, Python compiles, `make_multimer_af3_input.py`
was smoke-tested and produces a correct `fold_input.json`). **No AF3/ABCfold
job, no PLIP run, no real data of any kind has been generated.** The
`.gitignore`'d `data/`, `results/`, `logs/` directories are empty.

Reused verbatim (already confirmed against real completed IFB runs, per
their own module docstrings) from `ABCfold_NPF_pipeline`:
`scripts/abcfold_backends.py`, `scripts/compress_abcfold_metadata.py`,
`scripts/parquet_utils.py`, and `workflows/processing/submit_abcfold.sh`'s IFB
infra (AF3 `.sif`/CUDA_HOME auto-discovery, node exclusions, `--prime` flow).
Reused verbatim from the old DRB2 project: `scripts/sanitize_cif.py`,
`minimize_cif.py`, `fix_pdb.py`, `aggregate_summaries.py`.

New for this rebuild: `scripts/make_multimer_af3_input.py` (generalizes
single-protein `make_af3_input.py` to N protein + N RNA chains),
`scripts/fetch_mmseqs2_msa.py` (simplified — no NPF-pocket-pipeline local
MSA reuse, this project has no such cache), `scripts/pose_cluster_anchor.py`
(rigid-anchor Kabsch + hierarchical RMSD clustering — forked from
`tm_helix_alignment.py`'s Procrustes machinery, ported from the old
project's `notebooks/*_domain_analysis.ipynb` cells 38-58, generalized to
an arbitrary anchor/partner chain split), `scripts/select_top_n_per_cluster.py`
(top 20 models/cluster by `ranking_score`), and both `workflows/*/Snakefile`s.

## Key design decisions worth knowing before you touch anything

- **RNA chains are excluded from the pose-clustering feature vector** —
  only the anchor's rigid core + partner *protein* chains drive cluster
  assignment (mixing protein Cα and RNA P/C1′ RMSD isn't meaningful). RNA
  still rides along in the minimized/PLIP structures.
- **PLIP runs receptor=anchor vs. ligand=everything-else in one call per
  model**, not the old project's per-chain-pair exploded configs
  (DCL4×DRB2, DCL4×DRB4 separately). Simpler; easy to add pairwise configs
  back later under `configs/*.yaml` if finer interaction breakdowns are
  needed.
- **`min_core_frac` (fraction of anchor length) replaces the old project's
  hardcoded `MIN_CORE_SIZE = 150` residues** — that constant was tuned for
  DRB2 (~400aa) and would have been wrong for DCL4 (~1700aa).
- Ranking for top-N selection uses `ranking_score` from
  `model_metadata.parquet` (each ABCfold backend's own native ranking
  score, unified by `compress_abcfold_metadata.py`), falling back to `iptm`
  when a backend doesn't report one.

## Open items — things that need a real IFB run to actually verify

1. **RNA complex GPU/mem/time sizing is not tuned** — `submit_abcfold.sh`'s
   defaults (`gpu:l40s:1`, 80G, 600min) are sized for the smaller
   `drb2_drb4` complex. This was an explicit "bridge to cross later
   together" from the user at the start of this work — don't just guess at
   numbers, ask first.
2. **`mmseqs2msa`'s multimer behavior is assumed, not tested.** I believe
   (per ABCFold source, `abcfold.scripts.add_mmseqs_msa`) that ABCfold's
   own `mmseqs2msa` CLI walks every `protein` entry in a multi-chain
   `fold_input.json` and fills in each chain's own MSA/templates
   correctly, but this pipeline has never actually run it against a
   3-protein+2-RNA complex. First `workflows/preprocessing/Snakefile` run
   is the first real test — inspect the resulting
   `fold_input.resolved.json` before submitting to the cluster.
3. **ABCfold's per-backend output directory layout** — `abcfold_backends.py`
   is copied verbatim from a project where it was confirmed against real
   completed runs of *that* project's (single-protein, apo/holo) jobs. It
   should generalize fine to multi-chain complexes (the layout is
   per-backend, not per-topology), but this hasn't been directly confirmed
   here yet. Run `submit_abcfold.sh --test` first and inspect
   `results/abcfold/<complex>/` before trusting the full array.
4. Neither `bash submit_abcfold.sh --prime` (backend env warm-up) nor the
   `metadata-compress` conda env has been created on IFB yet for this
   project — both are one-time, login-node, internet-required setup steps
   documented in the README's Quick Start and in `submit_abcfold.sh`'s own
   header comment.

## Where to pick up

Natural next step is likely: install `envs/pipeline.yaml`, run
`workflows/preprocessing/Snakefile` for both complexes, inspect the two
`fold_input.resolved.json` outputs by hand against the old project's
`*_job_request.json` sequences (open item #2 above), then move to IFB for
`--prime` + `--test`.
