#!/bin/bash
# Run the 3 CALVADOS3 production replicates sequentially on OpenCL.
# Usage: bash run_triplicate.sh <workdir>   (workdir = results/.../calvados3_md/<sysname>)
set -euo pipefail

WORKDIR="$1"

for i in 1 2 3; do
  RUNDIR="${WORKDIR}/replicate_${i}"
  echo "=== [$(date)] starting replicate_${i} ==="
  micromamba run -n calvados3-md python "${RUNDIR}/run.py" --path "${RUNDIR}"
  echo "=== [$(date)] finished replicate_${i} ==="
done

echo "=== [$(date)] all replicates finished ==="
