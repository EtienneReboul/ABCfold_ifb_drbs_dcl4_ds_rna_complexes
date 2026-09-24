#!/bin/bash
# Convenience wrapper: unwrap (if needed) + launch PyMOL for one CALVADOS3 replicate.
#
# Usage:
#   bash visualize_replicate.sh <sysname> <replicate_num> [interactive|png] [extra pymol args...]
#
# Examples:
#   bash scripts/calvados3_md/visualize_replicate.sh rank_415_chai1_seed4_sample3.0 2
#   bash scripts/calvados3_md/visualize_replicate.sh rank_160_boltz_seed20_sample3.0 1 png --stride 5
set -euo pipefail

SYSNAME="$1"; shift
REP="$1"; shift
MODE="${1:-interactive}"; [ $# -gt 0 ] && shift || true

BASE="results/drb2_drb4/calvados3_md/${SYSNAME}"
CHAIN_BOUNDARY=435
PYMOL=/opt/homebrew/bin/pymol

RUNDIR="${BASE}/replicate_${REP}"
DOMAINS="${BASE}/input/domains.yaml"
UNWRAPPED="${RUNDIR}/${SYSNAME}_unwrapped.dcd"

# box size is per-system (depends on starting structure's extent), read it
# from this replicate's own config.yaml rather than hardcoding it
BOX_NM=$(awk '/^box:/{getline; gsub(/[- ]/,""); print; exit}' "${RUNDIR}/config.yaml")

if [ ! -f "$UNWRAPPED" ] || [ "${RUNDIR}/${SYSNAME}.dcd" -nt "$UNWRAPPED" ]; then
  echo "unwrapping replicate_${REP} (box=${BOX_NM} nm)..."
  micromamba run -n calvados3-md python scripts/calvados3_md/unwrap_traj.py \
    --rundir "$RUNDIR" --chain-boundary "$CHAIN_BOUNDARY" --box-nm "$BOX_NM"
fi

if [ "$MODE" = "png" ]; then
  PNGDIR="frames/${SYSNAME}/replicate_${REP}"
  "$PYMOL" -cq scripts/calvados3_md/visualize_pymol.py -- \
    "$RUNDIR" "$DOMAINS" --chain-boundary "$CHAIN_BOUNDARY" \
    --png-dir "$PNGDIR" "$@"
else
  "$PYMOL" scripts/calvados3_md/visualize_pymol.py -- \
    "$RUNDIR" "$DOMAINS" --chain-boundary "$CHAIN_BOUNDARY" "$@"
fi
