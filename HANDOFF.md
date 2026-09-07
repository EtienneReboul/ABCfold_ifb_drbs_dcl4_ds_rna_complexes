# Handoff prompt — paste this into a new Claude Code session opened on this project

I'm continuing work on `ABCfold_ifb_drbs_dcl4_ds_rna_complexes`. Read `README.md`
for the pipeline overview, then this note for current state. The auto-memory
files (`project_abcfold_drbs_pipeline.md`, `reference_ifb_cluster.md`) carry the
full running history — this file is the "what's live right now" summary.

## What this project is

ABCfold (https://github.com/rigdenlab/ABCFold) launches AlphaFold3, Boltz-2,
Chai-1, OpenFold3, Protenix and RosettaFold3 **together** per complex on the
IFB cluster (rebuild of `../ab_initio_modelling_drbs_dcl4_ds_rna_complexes`'s
manual AF3-webserver approach; same protocol as the sibling
`../../NPF-ab-initio-modelling/ABCfold_NPF_pipeline`). Then: rigid-anchor pose
clustering → top-N/cluster → ChimeraX minimize → fix_pdb → 3× PLIP passes →
aggregate. Analysis lives in `notebooks/`.

## Complexes (`config.yaml` `complexes:`), all DONE end-to-end

| complex | chains | backends that produced structures | notes |
|---|---|---|---|
| `drb2_drb4` | DRB2, DRB4 | all 6 | binary, anchor = DRB2 |
| `rna_ds_dcl4_drb2_drb4` | DCL4,DRB2,DRB4,dsRNA×2 (~2600 tok) | AF3, OpenFold3, RosettaFold3 (3/6) | Chai/Protenix hit hard token caps, Boltz OOM even on H200 — see memory |
| `rna_ds_drb2_drb4` | DRB2,DRB4,dsRNA×2 (~900 tok) | all 6 | DCL4 dropped so every backend fits |
| `rna_ds_dcl4_drb2_drb4_synthtmpl_0{1,2,3}` | = the DCL4 complex + a custom template | AF3 + OpenFold3 only | **synthetic-template re-run, see below** |

Baseline postprocessing deliverables: `results/<complex>/all_selected_summary*.csv`.

## Synthetic-template re-run — the current focus (commits 9c886f6, cab4433, 276082c)

**Idea.** In `rna_ds_dcl4_drb2_drb4`, AF3 and OpenFold3 fold the DRB2/DRB4
disordered tails into ~60% helix → ~0 models survive the non-MoRF over-folding
filter. The DCL4-free `rna_ds_drb2_drb4` keeps those tails disordered; its
Chai-1/Boltz predictions pass. So: inject a top-ipTM Boltz DRB2+DRB4
conformation back into the full complex as an AF3-dialect per-chain custom
template on DRB2/DRB4, and re-run only AF3 + OpenFold3.

**Pipeline plumbing (all committed):**
- `scripts/select_synthetic_templates.py` — pick top-N chai1/boltz over-folding
  survivors of `rna_ds_drb2_drb4` by ipTM (top-3 were all **boltz**, ipTM
  0.73/0.67/0.67), snapshot the donor complex's resolved-MSA JSON, emit
  `configs/rna_ds_dcl4_drb2_drb4_synthtmpl_NN.yaml` (structural blocks copied
  verbatim from `configs/rna_ds_dcl4_drb2_drb4.yaml` + `synthetic_template:` +
  `models: [alphafold3, openfold3]`), + `data/synthetic_templates/<family>/`
  (`tmpl_0N.pdb`, `donor_resolved.json`, `templates_manifest.tsv`).
- `scripts/inject_synthetic_template.py` — build each synthtmpl
  `fold_input.resolved.json` from the frozen donor JSON: retitle, drop
  ColabFold templates from DRB2/DRB4 (keep MSAs), inject one **identity-mapped**
  single-chain mmCIF template per chain. Does the injection itself (BioPython
  `MMCIFIO` + synthetic `revision_date`), **not** via abcfold's
  `add_custom_template` — that path's `Bio.Align.PairwiseAligner`-based mapping
  is BioPython-version-fragile (gave a shredded 91-fragment map under biopython
  1.84). Valid because the template chain IS the exact same DRB2/DRB4 construct
  as the query (verified residue-exact via `gemmi.one_letter_code`).
- `workflows/preprocessing/Snakefile` — `rule resolve_synthtmpl`, gated by a
  `_synthtmpl_` wildcard constraint so it and `fetch_mmseqs2_msa` never contend
  for the shared `fold_input.resolved.json` output. synthtmpl complexes reuse
  the donor's MSAs, no ColabFold call.
- `workflows/processing/submit_abcfold.sh` — new `--only <csv>` complex
  allowlist.
- `envs/preprocessing.yaml` — + `biopython`, `gemmi`. `.gitignore` — all of
  `data/`.
- Postprocessing unchanged (synthtmpl complexes have identical topology and are
  picked up by every `{complex}`-wildcard rule).

**Run status:**
- **synthtmpl_01: DONE + postprocessed + analysed.** Its IFB run was cut short
  by an inode-quota failure at OpenFold3 seed 18/20, so the ensemble is AF3
  100/100 + OpenFold3 90/100 (190 models), recovered locally and fully
  postprocessed (`results/rna_ds_dcl4_drb2_drb4_synthtmpl_01/all_selected_summary*.csv`,
  `dssp_summary.csv`, `overfolding_helix_stats.csv`,
  `overfolding_survivors_quicklook.tsv`).
  **RESULT: the idea works, for AlphaFold3.** AF3 DRB2 disordered tail (B
  189-434) went **66% helix / max run 24 → 4% / 4**; ~38 of 100 AF3 models
  pass a quick-look over-folding filter (1/200 in baseline). DRB4 tail only
  partially rescued (run ~13, filter edge). **OpenFold3 unchanged** (60% helix)
  — it does not act on the custom template for secondary structure. ipTM/pTM
  unchanged (template moves SS, not confidence). A few AF3 "survivors" still
  carry clash penalties (ranking_score ≈ −99).
- **synthtmpl_02 + _03: RUNNING on IFB now** — job `1776278` (`--array=0-1%2`,
  `--models ao`), submitted 2026-09-07. `1776278_0` (synthtmpl_02) RUNNING on
  gpu-node-4, `1776278_1` (synthtmpl_03) PENDING(Resources). ~6-10 h each.
  These use the 2nd/3rd-ranked Boltz templates — they test whether the AF3
  rescue is robust across templates.

## IFB inode quota — cost real debugging time 2026-09-03/07 (see reference-ifb-cluster memory)

`/shared/projects/npf_abinitio` is CephFS with `ceph.quota.max_files = 500000`
(`getfattr -n ceph.quota.max_files <dir>`), counting **all inodes**
(files+dirs+symlinks ≈ `find | wc -l`, not the smaller `ceph.dir.rfiles`).
`conda/` alone was ~544k → chronically over → **every write to the project
fails with `[Errno 122]`**, not just the job that tripped it. Fixed by purging
`conda/**/__pycache__` (~127k, `.pyc` auto-regenerate) + removing the
`redocking` conda env (~68k). **NB `redocking` was the NPF sibling's env** —
rebuildable from `ABCfold_NPF_pipeline/envs/redocking.yaml` (or it self-heals
on that pipeline's next `--use-conda` run). Project now ~379k/500k. `.pyc`
regenerate when envs run, so headroom will shrink again — a longer-term fix is
an IFB ticket to raise `max_files`.

## Notebooks

`notebooks/rna_ds_dcl4_drb2_drb4_synthtmpl_domain_analysis.ipynb` (new,
executed, committed 276082c) — the `rna_ds_dcl4_drb2_drb4_domain_analysis`
PLIP domain-contact analysis re-pointed at synthtmpl_01. Kernel
`abcfold-drbs-notebook`. Survivor section auto-picks from
`overfolding_survivors_quicklook.tsv` when `filtered_models.csv` isn't present;
`CLASH_EXCLUSIONS` starts empty (baseline's RNA(E)×PAZ clash was
baseline-specific).

## Where to pick up

1. **When `1776278_0` / `_1` finish** (check `squeue -u ereboul`; SLURM logs at
   `/shared/projects/npf_abinitio/ABCfold_ifb_drbs_dcl4_ds_rna_complexes/results/abcfold/array_manifest/logs/task_{0,1}.{log,err}`):
   rsync `results/abcfold/rna_ds_dcl4_drb2_drb4_synthtmpl_0{2,3}/` +
   `results/metadata/` back. If a run failed on the quota again, recover it the
   synthtmpl_01 way (rsync `*_model.cif` + `*_summary_confidences.json` +
   `*_confidences_aggregated.json` only, exclude the 250-300 MB per-sample
   `*_confidences.json` and `*_distogram.npz`, then
   `scripts/compress_abcfold_metadata.py` locally).
2. Local-postprocess synthtmpl_02/_03 (full chain, or light path + a ChimeraX
   `scripts/dssp_summary.py` run for just the over-folding number). Snakemake
   invocation on this Mac needs the conda-26.7.1 workaround: `env -u
   CONDA_PREFIX -u CONDA_DEFAULT_ENV CONDA_SHLVL=0 PATH="<controller bin>:/usr/local/bin:/usr/bin:/bin" snakemake ...`
   (`/usr/local/bin` for docker in the PLIP rules).
3. Compare synthtmpl_02/_03's AF3 DRB2/DRB4 disordered-tail helix to
   synthtmpl_01 — does the rescue hold across all 3 Boltz templates?
4. Optionally re-run `notebooks/rna_complexes_energy_and_overfolding_filter.ipynb`
   with the synthtmpl complexes added for the canonical (ANCHOR2-peak MoRF)
   survivor numbers vs the quick-look filter used so far.

## Older context (still true)

- RNA chains excluded from the pose-clustering feature vector; PLIP =
  receptor(anchor) vs ligand(rest) in one call, plus the dedicated
  `plip_rna_ligands:` / `plip_drb2_drb4:` passes; top-N selection by
  `ranking_score` (iptm fallback).
- `submit_abcfold.sh` GPU profile: `gpu:h200:1 / 250G / 2880min` (sized on the
  DCL4 complex). No `--prime` needed on this account — abcfold's config is
  user-scoped and the backend envs already exist from the NPF sibling.
- Two per-Mac Snakemake gotchas + the "module/micromamba not on PATH in
  non-login shells" cluster gotchas are all in `reference-ifb-cluster` memory
  and already handled in `submit_abcfold.sh`.
