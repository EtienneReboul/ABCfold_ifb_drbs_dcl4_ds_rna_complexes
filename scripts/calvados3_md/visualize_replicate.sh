#!/bin/bash
# Convenience wrapper: unwrap (if needed) + launch PyMOL for one CALVADOS3 replicate.
#
# Usage:
#   bash visualize_replicate.sh <replicate_num> [interactive|png] [extra pymol args...]
#
# Examples:
#   bash scripts/calvados3_md/visualize_replicate.sh 2                     # interactive GUI
#   bash scripts/calvados3_md/visualize_replicate.sh 3 png --stride 5      # PNG frames, every 5th frame
set -euo pipefail

SYSNAME=rank_415_chai1_seed4_sample3.0
BASE="results/drb2_drb4/calvados3_md/${SYSNAME}"
CHAIN_BOUNDARY=435
BOX_NM=45.0
PYMOL=/opt/homebrew/bin/pymol

REP="$1"; shift
MODE="${1:-interactive}"; [ $# -gt 0 ] && shift || true

RUNDIR="${BASE}/replicate_${REP}"
DOMAINS="${BASE}/input/domains.yaml"
UNWRAPPED="${RUNDIR}/${SYSNAME}_unwrapped.dcd"

if [ ! -f "$UNWRAPPED" ] || [ "${RUNDIR}/${SYSNAME}.dcd" -nt "$UNWRAPPED" ]; then
  echo "unwrapping replicate_${REP}..."
  micromamba run -n calvados3-md python scripts/calvados3_md/unwrap_traj.py \
    --rundir "$RUNDIR" --chain-boundary "$CHAIN_BOUNDARY" --box-nm "$BOX_NM"
fi

if [ "$MODE" = "png" ]; then
  PNGDIR="frames/replicate_${REP}"
  "$PYMOL" -cq scripts/calvados3_md/visualize_pymol.py -- \
    "$RUNDIR" "$DOMAINS" --chain-boundary "$CHAIN_BOUNDARY" \
    --png-dir "$PNGDIR" "$@"
else
  "$PYMOL" scripts/calvados3_md/visualize_pymol.py -- \
    "$RUNDIR" "$DOMAINS" --chain-boundary "$CHAIN_BOUNDARY" "$@"
fi
