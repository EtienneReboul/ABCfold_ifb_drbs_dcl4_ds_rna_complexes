#!/usr/bin/env bash
# =============================================================================
# submit_abcfold.sh — Stage 2: ABCfold structure prediction on the IFB cluster
# =============================================================================
#
# Adapted from ../../../NPF-ab-initio-modelling/ABCfold_NPF_pipeline's
# submit_abcfold.sh — copied nearly verbatim (same IFB cluster, same AF3
# .sif / CUDA_HOME auto-discovery, same node-exclusion list, same metadata
# compression step) — only FOLD_IN_DIR/ABCFOLD_OUT_DIR paths changed and the
# Gibberellin-importer priority-ordering logic dropped (not applicable:
# this pipeline only ever has 2 pending complexes, no apo/holo split).
#
# For every pending complex, runs ONE `abcfold` call that launches
# AlphaFold3, Boltz-2, Chai-1, OpenFold3, Protenix and RosettaFold3 together
# against the same fold_input.resolved.json
# (https://github.com/rigdenlab/ABCFold) — see configs/<complex>.yaml and
# ../../README.md for why this pipeline uses ABCfold instead of resampling
# AF3 alone (the old ../ab_initio_modelling_drbs_dcl4_ds_rna_complexes
# project's approach, run manually via the AF3 webserver).
#
# MSA + templates were already resolved locally (workflows/preprocessing/Snakefile,
# scripts/fetch_mmseqs2_msa.py — ABCfold's own `mmseqs2msa` CLI against the
# ColabFold MMseqs2 webserver) and embedded into fold_input.resolved.json, so
# this script never passes --mmseqs2 or --templates: compute nodes need no
# outbound network access for that part. What compute nodes on IFB likely
# DON'T have: Docker. ABCfold's AlphaFold3 backend shells out to `docker run`
# by default, or `singularity exec` if --af3_sif_path is set.
#
# CONFIRMED on IFB (2026-07-31, via ABCfold_NPF_pipeline): `module load
# alphafold/$AF3_MODULE_VERSION` is itself just a thin wrapper around
# Singularity — discover_af3_sif() below reads that wrapper script and
# extracts the .sif path it already uses, instead of hardcoding an absolute
# path that would go stale across module version bumps.
#
# One-time setup (needs internet — run on a login node, NOT a compute node):
#   Boltz, Chai-1, OpenFold3, Protenix and RosettaFold3 are auto-installed by
#   ABCfold into internal micromamba environments the first time each one
#   runs. Prime that install with `--dry_run` (sets up every selected
#   predictor's env + weights + a --help smoke test, no GPU, no inference):
#     bash submit_abcfold.sh --prime
#
#   Each array task also trims + standardizes its own results/abcfold/<complex>/
#   in place (raw per-sample confidence JSON/npz/npy -> results/metadata/<complex>/
#   {arrays.h5,model_metadata.parquet}, --delete-originals) right after that
#   complex's `abcfold` call finishes — see scripts/compress_abcfold_metadata.py's
#   module docstring. Needs its own conda env (regular conda, not ABCfold's
#   internal micromamba ones above), created once (also needs internet —
#   login node):
#     module load conda && conda env create -n metadata-compress -f envs/metadata_compress.yaml
#   A run whose compression fails/times out on the cluster is NOT stuck: its
#   raw files are simply left in place, and workflows/postprocessing/Snakefile's
#   compress_abcfold_metadata rule compresses it locally instead, the first
#   time that Snakefile runs after the rsync.
#
# Prerequisites:
#   - workflows/preprocessing/Snakefile completed (fold_input.resolved.json exists per complex)
#   - Run from the pipeline root directory
#   - micromamba available on $PATH (ABCfold requires it to build backend envs)
#   - `module load singularity` and `module load conda` both work (loaded automatically below)
#   - `bash submit_abcfold.sh --prime` has completed successfully at least once
#   - `module load conda && conda env create -n metadata-compress -f envs/metadata_compress.yaml`
#     has completed at least once
#
# Sized for the RNA complex (rna_ds_dcl4_drb2_drb4, 5 chains, ~2200
# residues) — the largest job here, and both complexes run under the same
# per-task resource profile since submit_abcfold.sh batches all pending
# complexes into one sbatch array. Chosen 2026-08-21 by checking `sinfo`
# for free GPUs: gpu-node-4 (4x H200, 141GB VRAM each) was fully idle,
# the highest-VRAM GPU on this cluster (vs. l40s's 48GB) — AF3's pair
# representation memory scales ~quadratically with token count, so the
# ~2.75x longer RNA complex needs much more than a linear VRAM bump.
# MEM/TIME bumped alongside it (gpu-node-4 has 1.5TB RAM to share across
# 4 GPUs; partition MaxTime is 3-00:00:00, see `scontrol show partition
# gpu` — 2880min leaves headroom under that cap). Override per-run with
# --gres/--mem/--time if a future complex needs something different.
#
# Usage:
#   bash submit_abcfold.sh --prime                     # one-time backend env warm-up (login node)
#   bash submit_abcfold.sh --dry-run                    # show plan only
#   bash submit_abcfold.sh --test                       # submit task 0 only (QoS-safe test)
#   bash submit_abcfold.sh --batch-size 1                # complexes per array task
#   bash submit_abcfold.sh --max-concurrent 2            # max parallel tasks
#   bash submit_abcfold.sh --models abcopr               # which backends (-a-b-c-o-p-r letters)
#   bash submit_abcfold.sh --gres gpu:a100:1             # bump up for the RNA complex
#   bash submit_abcfold.sh --only c1,c2                  # restrict to these complex(es)
#
# The *_synthtmpl_* re-runs (AF3 + OpenFold3 only, custom template on
# DRB2/DRB4) are submitted as their own batch:
#   bash submit_abcfold.sh \
#     --only rna_ds_dcl4_drb2_drb4_synthtmpl_01,rna_ds_dcl4_drb2_drb4_synthtmpl_02,rna_ds_dcl4_drb2_drb4_synthtmpl_03 \
#     --models ao
# They are the same 5-chain size as rna_ds_dcl4_drb2_drb4, so the default
# gpu:h200:1 / 250G / 2880min profile applies unchanged. (Per-complex model
# auto-selection from configs/<c>.yaml's `models:` block is a possible later
# improvement; for now `--models ao` is passed explicitly.)
#
# ABCfold has no documented cache-and-resume split — one `abcfold` call
# either completes or it doesn't. A retried run always passes --override
# and recomputes every backend from scratch; only complexes with a
# prediction.done sentinel are skipped entirely.
# =============================================================================

set -euo pipefail

# Neither a non-login remote shell (e.g. `ssh host 'bash submit_abcfold.sh
# ...'`) nor a SLURM batch script sources /etc/profile.d/modules.sh the way
# an interactive login shell does, so `module` is otherwise undefined here
# and inside the generated SLURM_SCRIPT heredoc below (confirmed 2026-08-21:
# job 1385070's --test run failed with "module: command not found" without
# this — sbatch does not reliably inherit the `module` shell function from
# the submitting shell across all invocation paths).
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PARAMS="/shared/bank/alphafold3/current"   # AF3 weights dir (config.yaml abcfold.model_params)
AF3_MODULE_VERSION="3.0.2"                        # config.yaml abcfold.af3_module_version
AF3_SIF_PATH=""                                   # leave empty to auto-discover (see discover_af3_sif below);
                                                   # set explicitly (or config.yaml's abcfold.af3_sif_path) to override
CUDA_MODULE_VERSION="12.9.1"                      # IFB cuda-toolkit module — Protenix needs CUDA_HOME set to
                                                   # build its CUDA extensions; see discover_cuda_home below
MODELS="abcopr"                                   # -a -b -c -o -p -r letters to run together

NUMBER_OF_MODELS=5        # abcfold --number_of_models (config.yaml abcfold.number_of_models)
NUM_RECYCLES=10            # abcfold --num_recycles     (config.yaml abcfold.num_recycles)

# ── Discover the AF3 .sif IFB's `module load alphafold` already wraps ────────
discover_af3_sif() {
    local wrapper_dir="/shared/software/singularity/wrappers/alphafold/$AF3_MODULE_VERSION"
    local wrapper="$wrapper_dir/run_alphafold.py"
    if [[ ! -f "$wrapper" ]]; then
        return 1
    fi
    local sif
    sif=$(grep -oE '[^[:space:]]+\.sif' "$wrapper" | head -n1)
    if [[ -z "$sif" ]]; then
        return 1
    fi
    if [[ "$sif" != /* ]]; then
        sif="$wrapper_dir/$sif"
    fi
    if [[ ! -f "$sif" ]]; then
        return 1
    fi
    echo "$sif"
}

if [[ -z "$AF3_SIF_PATH" ]]; then
    if DISCOVERED_SIF=$(discover_af3_sif); then
        AF3_SIF_PATH="$DISCOVERED_SIF"
        echo "[submit_abcfold] Auto-discovered AF3 .sif: $AF3_SIF_PATH"
    fi
fi

# ── Discover CUDA_HOME from IFB's cuda-toolkit module ─────────────────────────
discover_cuda_home() {
    local modulefile="/shared/software/modulefiles/cuda-toolkit/$CUDA_MODULE_VERSION"
    if [[ ! -f "$modulefile" ]]; then
        return 1
    fi
    local env_bin env_root targets_root
    env_bin=$(grep -oE '[^[:space:]]+/envs/cuda-toolkit-[^[:space:]]+/bin' "$modulefile" | head -n1)
    if [[ -z "$env_bin" ]]; then
        return 1
    fi
    env_root=$(dirname "$env_bin")
    targets_root=$(find "$env_root/targets" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -n1)
    if [[ -z "$targets_root" || ! -f "$targets_root/include/cuda_runtime_api.h" ]]; then
        echo "$env_root"
        return 0
    fi
    # See ABCfold_NPF_pipeline's submit_abcfold.sh for the full rationale
    # behind this shim (conda-forge's cuda-toolkit split, confirmed 2026-07-31).
    local shim="/shared/projects/npf_abinitio/conda/cuda_home_shim"
    mkdir -p "$shim"
    ln -sfn "$env_root/bin" "$shim/bin"
    ln -sfn "$env_root/nvvm" "$shim/nvvm"
    ln -sfn "$targets_root/include" "$shim/include"
    ln -sfn "$targets_root/lib" "$shim/lib"
    ln -sfn "$targets_root/lib" "$shim/lib64"
    echo "$shim"
}

CUDA_HOME_PATH=""
if DISCOVERED_CUDA_HOME=$(discover_cuda_home); then
    CUDA_HOME_PATH="$DISCOVERED_CUDA_HOME"
    echo "[submit_abcfold] Auto-discovered CUDA_HOME: $CUDA_HOME_PATH"
else
    echo "WARNING: could not auto-discover CUDA_HOME from cuda-toolkit/$CUDA_MODULE_VERSION —"
    echo "         Protenix (-p) will likely fail to build its CUDA extensions."
fi

# Batching — only 2 complexes total here, so default to 1/task.
BATCH_SIZE=1
MAX_CONCURRENT=2

# SLURM resources (per array task) — six backends in one task needs more
# memory/time headroom than an AF3-only pipeline's per-protein task, and
# these are sized for drb2_drb4 (2 chains); bump for the RNA complex.
PARTITION="gpu"
CPUS=8
MEM="250G"
GRES="gpu:h200:1"
TIME=2880                 # minutes per task (48h)
ACCOUNT=""

# Nodes with confirmed infra issues (via ABCfold_NPF_pipeline, 2026-08-06) —
# same IFB cluster, same exclusion list. Override with --exclude-nodes
# (empty string to stop excluding, once/if IFB confirms these are healthy).
EXCLUDE_NODES="gpu-node-7,gpu-node-9"

FOLD_IN_DIR="data/fold_inputs"
ABCFOLD_OUT_DIR="results/abcfold"
METADATA_OUT_DIR="results/metadata"           # config.yaml dirs.metadata — scripts/compress_abcfold_metadata.py output
METADATA_ENV="metadata-compress"              # envs/metadata_compress.yaml env name

# The `abcfold` CLI itself lives in this conda env (confirmed 2026-08-21:
# job 1385293's --test run failed with "abcfold: command not found" when
# invoked bare — a non-login/non-interactive submitting shell, e.g. `ssh
# host 'bash submit_abcfold.sh ...'`, never gets this env's bin/ on PATH the
# way an interactive shell with it already activated would). Resolved by
# absolute path, same reasoning as METADATA_PYTHON below, with a PATH
# fallback in case a future environment already has `abcfold` activated.
ABCFOLD_BIN="/shared/projects/npf_abinitio/conda/envs/abcfold-npf-pipeline/bin/abcfold"
[[ -x "$ABCFOLD_BIN" ]] || ABCFOLD_BIN="abcfold"

# ── Parse arguments ───────────────────────────────────────────────────────────
DRY_RUN=false
TEST_MODE=false
PRIME=false
ONLY=""                     # comma-separated complex allowlist (empty = all pending)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=true; shift ;;
        --prime)
            PRIME=true; shift ;;
        --gres)
            GRES="$2"; shift 2 ;;
        --gres=*)
            GRES="${1#--gres=}"; shift ;;
        --mem)
            MEM="$2"; shift 2 ;;
        --mem=*)
            MEM="${1#--mem=}"; shift ;;
        --time)
            TIME="$2"; shift 2 ;;
        --time=*)
            TIME="${1#--time=}"; shift ;;
        --models)
            MODELS="$2"; shift 2 ;;
        --models=*)
            MODELS="${1#--models=}"; shift ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --batch-size=*)
            BATCH_SIZE="${1#--batch-size=}"; shift ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"; shift 2 ;;
        --max-concurrent=*)
            MAX_CONCURRENT="${1#--max-concurrent=}"; shift ;;
        --exclude-nodes)
            EXCLUDE_NODES="$2"; shift 2 ;;
        --exclude-nodes=*)
            EXCLUDE_NODES="${1#--exclude-nodes=}"; shift ;;
        --only)
            ONLY="$2"; shift 2 ;;
        --only=*)
            ONLY="${1#--only=}"; shift ;;
        --test)
            TEST_MODE=true; shift ;;
        *)
            echo "ERROR: unknown argument '$1'"
            echo "Usage: bash submit_abcfold.sh [--prime] [--dry-run] [--test] [--gres <profile>]"
            echo "                              [--mem <size>] [--time <minutes>]"
            echo "                              [--models abcopr] [--batch-size N]"
            echo "                              [--max-concurrent N] [--exclude-nodes node1,node2]"
            echo "                              [--only complex1,complex2]"
            exit 1 ;;
    esac
done

MODEL_FLAG="-${MODELS}"

# ── Prime mode: warm up internal micromamba backend envs (needs internet) ────
if $PRIME; then
    echo "============================================================"
    echo " ABCfold --dry_run priming (backend env + weights setup only)"
    echo " Models  : $MODEL_FLAG"
    echo " Run this on a LOGIN NODE (or any node with outbound internet),"
    echo " never on a compute node — this is the only step that installs"
    echo " Boltz/Chai-1/OpenFold3/Protenix/RosettaFold3."
    echo "============================================================"
    ANY_JSON=$(find "$FOLD_IN_DIR" -maxdepth 2 -name 'fold_input.resolved.json' -print -quit)
    if [[ -z "$ANY_JSON" ]]; then
        echo "ERROR: no fold_input.resolved.json found under $FOLD_IN_DIR." \
             "Run workflows/preprocessing/Snakefile first." >&2
        exit 1
    fi
    module load singularity   # ABCfold shells out to `singularity exec` directly for AF3
    module load "cuda-toolkit/$CUDA_MODULE_VERSION"   # Protenix needs CUDA_HOME to build its CUDA extensions
    export CUDA_HOME="$CUDA_HOME_PATH"
    "$ABCFOLD_BIN" "$ANY_JSON" "$ABCFOLD_OUT_DIR/_prime" \
        $MODEL_FLAG \
        --model_params "$MODEL_PARAMS" \
        $( [[ -n "$AF3_SIF_PATH" ]] && echo "--af3_sif_path $AF3_SIF_PATH" ) \
        --dry_run \
        --override
    echo "Priming complete."
    exit 0
fi

# ── Validate prerequisites ────────────────────────────────────────────────────
if [[ ! -d "$FOLD_IN_DIR" ]]; then
    echo "ERROR: $FOLD_IN_DIR not found. Run workflows/preprocessing/Snakefile first."
    exit 1
fi
if [[ -z "$AF3_SIF_PATH" ]] && [[ "$MODEL_FLAG" == *a* ]]; then
    echo "WARNING: AF3_SIF_PATH is empty and AlphaFold3 (-a) is selected —"
    echo "         auto-discovery of IFB's alphafold/$AF3_MODULE_VERSION .sif failed"
    echo "         (see discover_af3_sif() above). ABCfold will fall back to"
    echo "         'docker run', which is normally unavailable on IFB compute"
    echo "         nodes. Set AF3_SIF_PATH in this script manually, or check that"
    echo "         /shared/software/singularity/wrappers/alphafold/$AF3_MODULE_VERSION/run_alphafold.py"
    echo "         still exists and still references a .sif file."
fi
if [[ ! -d "/shared/projects/npf_abinitio/conda/envs/${METADATA_ENV}" ]]; then
    echo "WARNING: conda env '$METADATA_ENV' not found — each array task's"
    echo "         post-prediction compression step (scripts/compress_abcfold_metadata.py)"
    echo "         will fail and fall back to leaving results/abcfold/<complex>/ raw"
    echo "         (harmless — workflows/postprocessing/Snakefile compresses it locally"
    echo "         later instead). Fix once, on a login node:"
    echo "           module load conda && conda env create -n $METADATA_ENV -f envs/metadata_compress.yaml"
fi

# ── Collect pending complexes ──────────────────────────────────────────────────
PENDING_JSONS=()
PENDING_DONE=()
skipped=0

only_filtered=0
for JSON in "$FOLD_IN_DIR"/*/fold_input.resolved.json; do
    [[ -f "$JSON" ]] || continue
    COMPLEX=$(basename "$(dirname "$JSON")")
    if [[ -n "$ONLY" ]] && [[ ",$ONLY," != *",$COMPLEX,"* ]]; then
        only_filtered=$((only_filtered + 1))
        continue
    fi
    DONE_FILE="$ABCFOLD_OUT_DIR/$COMPLEX/prediction.done"
    if [[ -f "$DONE_FILE" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    PENDING_JSONS+=("$JSON")
    PENDING_DONE+=("$DONE_FILE")
done

TOTAL=${#PENDING_JSONS[@]}
N_TASKS=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))   # ceiling division
LAST_TASK=$(( N_TASKS - 1 ))
ARRAY_SPEC="0-${LAST_TASK}%${MAX_CONCURRENT}"

echo "============================================================"
echo " ABCfold SLURM job array submission"
echo " Pending complexes       : $TOTAL"
echo " Already done            : $skipped"
[[ -n "$ONLY" ]] && echo " --only filter            : $ONLY  (excluded $only_filtered)"
echo " Models                  : $MODEL_FLAG"
echo " Models/recycles per run : $NUMBER_OF_MODELS / $NUM_RECYCLES"
echo " Batch size              : $BATCH_SIZE complex(es)/task"
echo " Array tasks              : $N_TASKS  (--array=${ARRAY_SPEC})"
echo " Max concurrent           : $MAX_CONCURRENT"
echo " GRES                     : $GRES"
echo " Mem/task                 : $MEM"
echo " Excluded nodes           : ${EXCLUDE_NODES:-(none)}"
echo " Time limit/task          : ${TIME} min"
if $DRY_RUN; then
    echo " Mode                     : DRY RUN (nothing submitted)"
fi
if $TEST_MODE; then
    echo " Mode                     : TEST — task 0 only ($(( BATCH_SIZE < TOTAL ? BATCH_SIZE : TOTAL )) complex(es))"
fi
echo "============================================================"
echo ""

if [[ $TOTAL -eq 0 ]]; then
    if [[ -n "$ONLY" && $skipped -eq 0 ]]; then
        echo "Nothing pending for --only '$ONLY' — no fold_input.resolved.json for"
        echo "those complex(es) yet (run workflows/preprocessing/Snakefile first)?"
    else
        echo "Nothing to do — all predictions already complete."
    fi
    exit 0
fi

# ── Write the batch manifest ──────────────────────────────────────────────────
MANIFEST_DIR="$ABCFOLD_OUT_DIR/array_manifest"
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/manifest.txt"

: > "$MANIFEST"   # truncate
for (( i=0; i<TOTAL; i++ )); do
    json="${PENDING_JSONS[$i]}"
    done_file="${PENDING_DONE[$i]}"
    complex=$(basename "$(dirname "$json")")
    out_dir="$ABCFOLD_OUT_DIR/$complex"
    mkdir -p "$(dirname "$out_dir")"
    echo "${json}|${out_dir}|${done_file}" >> "$MANIFEST"
done

echo "Manifest written: $MANIFEST ($TOTAL lines)"
echo ""

if $DRY_RUN; then
    echo "Dry-run — array tasks breakdown:"
    for (( task=0; task<N_TASKS; task++ )); do
        start=$(( task * BATCH_SIZE ))
        end=$(( start + BATCH_SIZE - 1 ))
        [[ $end -ge $TOTAL ]] && end=$(( TOTAL - 1 ))
        count=$(( end - start + 1 ))
        echo "  Task $task: $count complex(es) (lines $((start+1))-$((end+1)))"
        for (( i=start; i<=end; i++ )); do
            json="${PENDING_JSONS[$i]}"
            complex=$(basename "$(dirname "$json")")
            echo "    $complex"
        done
    done
    echo ""
    echo "Would submit: sbatch --array=${ARRAY_SPEC} ..."
    exit 0
fi

# ── Submit the job array ──────────────────────────────────────────────────────
LOG_DIR="$MANIFEST_DIR/logs"
mkdir -p "$LOG_DIR"

if $TEST_MODE; then
    ARRAY_SPEC="0"
    echo "TEST MODE: submitting task 0 only (${BATCH_SIZE} complex(es))"
    echo "           Once it completes successfully, run without --test for the full array."
    echo ""
fi

ACCOUNT_FLAG=""
[[ -n "$ACCOUNT" ]] && ACCOUNT_FLAG="--account=$ACCOUNT"

EXCLUDE_FLAG=""
[[ -n "$EXCLUDE_NODES" ]] && EXCLUDE_FLAG="--exclude=$EXCLUDE_NODES"

sbatch \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --gres="$GRES" \
    --time="$TIME" \
    --array="${ARRAY_SPEC}" \
    $ACCOUNT_FLAG \
    $EXCLUDE_FLAG \
    --job-name="abcfold_drbs_array" \
    --output="$LOG_DIR/task_%a.log" \
    --error="$LOG_DIR/task_%a.err" \
    << SLURM_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

# See the matching comment near the top of this script, above the
# Configuration section — the SLURM batch script is its own separate
# process on the compute node and needs this sourced again independently.
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh

TASK_ID=\$SLURM_ARRAY_TASK_ID
BATCH_SIZE=$BATCH_SIZE
MANIFEST="$MANIFEST"
MODEL_FLAG="$MODEL_FLAG"
MODEL_PARAMS="$MODEL_PARAMS"
AF3_SIF_PATH="$AF3_SIF_PATH"
CUDA_HOME_PATH="$CUDA_HOME_PATH"
CUDA_MODULE_VERSION="$CUDA_MODULE_VERSION"
NUMBER_OF_MODELS=$NUMBER_OF_MODELS
NUM_RECYCLES=$NUM_RECYCLES
ABCFOLD_OUT_DIR="$ABCFOLD_OUT_DIR"
METADATA_OUT_DIR="$METADATA_OUT_DIR"
METADATA_ENV="$METADATA_ENV"
ABCFOLD_BIN="$ABCFOLD_BIN"

module load singularity   # ABCfold shells out to \`singularity exec\` directly for AF3
module load "cuda-toolkit/\$CUDA_MODULE_VERSION"   # Protenix needs CUDA_HOME to build its CUDA extensions
export CUDA_HOME="\$CUDA_HOME_PATH"

echo "[\$(date)] Array task \$TASK_ID — batch size $BATCH_SIZE"
echo "  Manifest: \$MANIFEST"
echo "  abcfold: \$ABCFOLD_BIN"
echo "  singularity: \$(which singularity)"

LINE_START=\$(( TASK_ID * BATCH_SIZE + 1 ))
LINE_END=\$(( LINE_START + BATCH_SIZE - 1 ))

echo "[\$(date)] Processing manifest lines \$LINE_START-\$LINE_END"
echo ""

while IFS='|' read -r json out_dir done_file; do
    [[ -z "\$json" ]] && continue

    complex=\$(basename "\$(dirname "\$json")")

    if [[ -f "\$done_file" ]]; then
        echo "[\$(date)] SKIP: \$complex (already done)"
        continue
    fi

    echo "[\$(date)] START abcfold \$MODEL_FLAG: \$complex"
    "\$ABCFOLD_BIN" "\$json" "\$out_dir" \\
        \$MODEL_FLAG \\
        --model_params "\$MODEL_PARAMS" \\
        \$( [[ -n "\$AF3_SIF_PATH" ]] && echo "--af3_sif_path \$AF3_SIF_PATH" ) \\
        --number_of_models "\$NUMBER_OF_MODELS" \\
        --num_recycles "\$NUM_RECYCLES" \\
        --no_server \\
        --no_visuals \\
        --override

    echo "\$(date): prediction finished" > "\$done_file"
    echo "[\$(date)] DONE: \$complex"

    # Trim + standardize this run's own results/abcfold/\$complex/ in place
    # (raw per-sample confidence sprawl -> results/metadata/\$complex/
    # {arrays.h5,model_metadata.parquet}, --delete-originals) before rsync —
    # see scripts/compress_abcfold_metadata.py's module docstring. Failure
    # here is non-fatal (prediction.done is already written above): raw
    # files are left as-is and workflows/postprocessing/Snakefile's
    # compress_abcfold_metadata rule compresses this run locally instead.
    #
    # --protein is that script's flag name (inherited unmodified from
    # ABCfold_NPF_pipeline) but it's really just a run-identifier — passing
    # a complex name here is exactly what it expects.
    METADATA_PYTHON="/shared/projects/npf_abinitio/conda/envs/\$METADATA_ENV/bin/python3"
    echo "[\$(date)] COMPRESS: \$complex"
    if [[ -x "\$METADATA_PYTHON" ]] && "\$METADATA_PYTHON" scripts/compress_abcfold_metadata.py \\
            --protein "\$complex" \\
            --abcfold-output-root "\$ABCFOLD_OUT_DIR" \\
            --out-root "\$METADATA_OUT_DIR" \\
            --delete-originals \\
            --skip-merge; then
        echo "[\$(date)] COMPRESS OK: \$complex"
    else
        echo "[\$(date)] WARNING: compression failed for \$complex (missing env at \$METADATA_PYTHON, or the script itself failed) — raw results/abcfold/\$complex/ left as-is, will be compressed locally by workflows/postprocessing/Snakefile instead"
    fi
    echo ""

done < <(sed -n "\${LINE_START},\${LINE_END}p" "\$MANIFEST")

echo "[\$(date)] Array task \$TASK_ID complete."
SLURM_SCRIPT

echo "Job array submitted: --array=${ARRAY_SPEC}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $LOG_DIR/task_0.log"
