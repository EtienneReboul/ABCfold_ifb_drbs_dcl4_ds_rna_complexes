#!/bin/bash
#SBATCH --job-name=calvados_rna_prod
#SBATCH --account=npf_abinitio
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --exclude=gpu-node-7,gpu-node-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=1-3
#SBATCH --output=/shared/projects/npf_abinitio/ABCfold_ifb_drbs_dcl4_ds_rna_complexes/logs/calvados3_md/ifb_replicate_%a_%j.log

set -euo pipefail
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh

CALV=/shared/projects/npf_abinitio/conda/envs/calvados3-md-cuda
BASE=/shared/projects/npf_abinitio/ABCfold_ifb_drbs_dcl4_ds_rna_complexes
cd "$BASE"

nvidia-smi -L

REP="$SLURM_ARRAY_TASK_ID"

"$CALV/bin/python" scripts/calvados3_md/prepare_calvados_with_rna.py \
  --protein-sysname rank_415_chai1_seed4_sample3.0 \
  --protein-input-dir "$BASE/results/drb2_drb4/calvados3_md/rank_415_chai1_seed4_sample3.0/input" \
  --rna-sysname rank_01_boltz_seed5_sample0.0 \
  --rna-input-dir "$BASE/results/rna_ds_drb2_drb4/calvados3_md/rank_01_boltz_seed5_sample0.0/input" \
  --workdir "$BASE/results/rna_ds_drb2_drb4/calvados3_md/drb2_drb4_with_rna" \
  --tag "replicate_${REP}" --platform CUDA --threads 1 --steps 100000000 --wfreq 100000

RUNDIR="$BASE/results/rna_ds_drb2_drb4/calvados3_md/drb2_drb4_with_rna/replicate_${REP}"
"$CALV/bin/python" "$RUNDIR/run.py" --path "$RUNDIR"
echo "=== replicate_${REP} finished ==="
