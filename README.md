# ABCfold IFB DRB2/DCL4/dsRNA Complex Pipeline

Rebuild of [`../ab_initio_modelling_drbs_dcl4_ds_rna_complexes`](../ab_initio_modelling_drbs_dcl4_ds_rna_complexes)
(which submitted AF3-webserver replicas by hand, one at a time, and
downloaded results manually) using the same automated protocol as
[`../../NPF-ab-initio-modelling/ABCfold_NPF_pipeline`](../../NPF-ab-initio-modelling/ABCfold_NPF_pipeline):
[ABCfold](https://github.com/rigdenlab/ABCFold) launches AlphaFold3,
Boltz-2, Chai-1, OpenFold3, Protenix and RosettaFold3 **together** per
complex on the IFB Core Cluster, then the resulting ensemble is clustered
by rigid-anchor structural alignment, and the top predictions per cluster
are minimized and run through PLIP to detect interactions.

## Why ABCfold instead of resampling AF3 alone

The old project ran many independent AF3-webserver replicas (up to 100
models for the DCL4/DRB2 complex) of a single architecture. The sibling
`ABCfold_NPF_pipeline` project found — for the unrelated NPF transporter
family — that AF3 alone, even with 30 seeds, does not recover every
conformation a second, architecturally-different model (Boltz-2) finds for
the same sequences. This pipeline applies the same fix here: run six
independent architectures per complex instead of just resampling one.

## Complexes modelled

Two complexes only (a deliberately simplified subset of the old project's
six):

| Complex | Chains | Anchor | PLIP `dnareceptor` |
|---|---|---|---|
| `drb2_drb4` | DRB2 (A), DRB4 (B) | DRB2 (no DCL4 in this complex) | false |
| `rna_ds_dcl4_drb2_drb4` | DCL4 (A), DRB2 (B), DRB4 (C), dsRNA duplex (D, E) | DCL4 | true |

Sequences are copied verbatim from the old project's AF3-webserver
`*_job_request.json` files — see `configs/drb2_drb4.yaml` and
`configs/rna_ds_dcl4_drb2_drb4.yaml`.

---

## Pipeline overview

```text
┌───────────────────────────────────────────────────────────────┐
│  PRE-PROCESSING (local, needs internet)                        │
│  workflows/preprocessing/Snakefile                                │
│                                                                 │
│  1a. Fold input   fold_input.json per complex                    │
│                   (AlphaFold3-dialect JSON, ABCfold's own          │
│                   native input format — every chain, every seed)   │
│        v                                                        │
│  1b. MMseqs2      MSA + top-hit templates from the ColabFold        │
│      default run  webserver, via ABCfold's own `mmseqs2msa` CLI,     │
│                   once per complex, embedded into                     │
│                   fold_input.resolved.json                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │  rsync data/fold_inputs/
                           v
┌───────────────────────────────────────────────────────────────┐
│  PROCESSING (IFB cluster, no internet needed)                    │
│  workflows/processing/submit_abcfold.sh                             │
│                                                                     │
│  2. ABCfold run — one `abcfold -abcopr ...` call per complex,        │
│     launching AlphaFold3 + Boltz-2 + Chai-1 + OpenFold3 + Protenix +  │
│     RosettaFold3 TOGETHER against the same fold_input.resolved.json.   │
│     Each array task also compresses its own raw confidence sprawl       │
│     down to model_metadata.parquet before rsync.                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │  rsync results/abcfold/, results/metadata/
                           v
┌───────────────────────────────────────────────────────────────┐
│  POST-PROCESSING (local)  workflows/postprocessing/Snakefile        │
│                                                                     │
│  3a. compress_abcfold_metadata  (local fallback, if the cluster-side │
│      step above didn't run/finish)                                    │
│        v                                                              │
│  3b. pose_cluster    rigid-anchor Kabsch alignment (anchor chain's      │
│                      iteratively-refined core) + hierarchical RMSD      │
│                      clustering of the partner chain(s), pooled          │
│                      across all 6 backends x seeds                        │
│        v                                                                  │
│  3c. select_top_n_per_cluster   top 20 models/cluster by ranking_score      │
│        v                                                                    │
│  3d-3g. minimize (ChimeraX) -> fix_pdb (pdb4amber+PDBFixer, RNA complex       │
│         only) -> PLIP (Docker) -> aggregate                                    │
└───────────────────────────────────────────────────────────────┘
```

---

## Directory layout

```text
ABCfold_ifb_drbs_dcl4_ds_rna_complexes/
├── config.yaml                        ← shared defaults (tracked by git)
├── config.local.yaml.example          ← optional personal overrides
├── envs/
│   ├── pipeline.yaml                  ← controller env (install once)
│   ├── preprocessing.yaml             ← fold_input.json + mmseqs2msa resolution
│   ├── postprocessing.yaml            ← pose clustering + selection
│   ├── metadata_compress.yaml         ← compress_abcfold_metadata.py
│   ├── fix_pdb.yaml / plip.yaml / pliparser.yaml / aggregate.yaml
├── configs/
│   ├── drb2_drb4.yaml                 ← chain spec, anchor, PLIP settings
│   └── rna_ds_dcl4_drb2_drb4.yaml     ← chain spec, anchor, PLIP settings
├── workflows/
│   ├── preprocessing/Snakefile        ← stage 1 (local)
│   ├── processing/submit_abcfold.sh   ← stage 2 SLURM submission (cluster)
│   └── postprocessing/Snakefile       ← stage 3 (local)
├── scripts/
│   ├── make_multimer_af3_input.py     ← stage 1a
│   ├── fetch_mmseqs2_msa.py           ← stage 1b
│   ├── abcfold_backends.py            ← shared ABCfold output-layout knowledge
│   ├── compress_abcfold_metadata.py   ← stage 3a
│   ├── parquet_utils.py               ← self-documenting parquet helper
│   ├── pose_cluster_anchor.py         ← stage 3b
│   ├── select_top_n_per_cluster.py    ← stage 3c
│   ├── sanitize_cif.py / minimize_cif.py / fix_pdb.py / aggregate_summaries.py
└── data/, results/, logs/             ← created automatically (gitignored)
```

`scripts/abcfold_backends.py`, `parquet_utils.py` and
`compress_abcfold_metadata.py` are copied verbatim from
`ABCfold_NPF_pipeline`, where every backend's output-file layout has
already been confirmed against real completed IFB runs (not guessed from
source) — see that project's own module docstrings for the confirmation
notes. `sanitize_cif.py`, `minimize_cif.py`, `fix_pdb.py` and
`aggregate_summaries.py` are copied verbatim from the old
`ab_initio_modelling_drbs_dcl4_ds_rna_complexes` project's PLIP `Snakefile`.

---

## Quick start

### 1. Install the controller environment (once)

```bash
conda env create -f envs/pipeline.yaml
conda activate af3-ifb-drbs-pipeline
```

### 2. Pre-processing (local, needs internet)

```bash
snakemake -s workflows/preprocessing/Snakefile --cores 2 --use-conda
```

Produces `data/fold_inputs/<complex>/fold_input.resolved.json` for both
complexes.

### 3. Processing (IFB cluster)

```bash
bash workflows/processing/submit_abcfold.sh --prime     # once, on a login node
bash workflows/processing/submit_abcfold.sh --test       # QoS-safe single-task test
bash workflows/processing/submit_abcfold.sh              # full array
```

Then `rsync` `results/abcfold/` and `results/metadata/` back locally.

### 4. Post-processing (local)

```bash
snakemake -s workflows/postprocessing/Snakefile --cores 4 --use-conda
```

Produces, per complex, under `results/<complex>/`:
- `pose_clusters.csv`, `rmsf_profile.svg`, `pose_clusters_pca.svg`
- `selected/cluster_<k>/rank_<r>_*.cif` (all models/cluster)
- `minimized/`, `plip/` (per selected model)
- `all_selected_summary.csv`

---

## GPU sizing

`rna_ds_dcl4_drb2_drb4` (5 chains, ~2200 residues) is the largest job here.
`submit_abcfold.sh`'s defaults (`gpu:h200:1`, 250G, 2880min) were chosen
2026-08-21 by checking `sinfo` on IFB for free GPUs — `gpu-node-4`'s 4x
H200 (141GB VRAM each) were fully idle and are the highest-VRAM GPU on the
cluster (vs. L40S's 48GB), needed for the RNA complex's much larger
pair-representation memory footprint. Both complexes share this profile
(`submit_abcfold.sh` batches all pending complexes into one sbatch array).
Override per-run with `--gres`/`--mem`/`--time` if a future complex needs
something different.

## Simplifications relative to the old project's clustering notebooks

- Pose clustering only: rigid-anchor Kabsch + hierarchical RMSD clustering
  with silhouette k-selection. The old project's PLIP-fingerprint Jaccard
  clustering, t-SNE, UMAP and GMM-on-UMAP branches were dropped.
- RNA chains are excluded from the pose-clustering feature vector (mixing
  protein Cα and RNA P/C1′ RMSD isn't meaningful) — they still ride along
  in the minimized/PLIP structures, they just don't drive cluster
  assignment.
- PLIP runs once per selected model, receptor = anchor chain vs. ligand =
  every other chain, rather than the old project's per-chain-pair exploded
  configs (DCL4×DRB2, DCL4×DRB4 separately). Easy to add back as extra
  `configs/*.yaml` PLIP variants later if finer-grained interaction
  breakdowns are needed.
