"""
scripts/dssp_summary.py
========================
ChimeraX batch script: run ChimeraX's built-in `dssp` command (no external
DSSP binary/library needed) over a list of minimized PDBs and dump a
per-residue secondary-structure CSV.

Distinguishes two hypotheses for a compact "disordered" region: is the
region actually folding into helix/sheet where it shouldn't (a genuine
secondary-structure mis-prediction), or does it stay coil and just pack
non-specifically/compactly (physically plausible for a collapsed IDR, no
false secondary structure)?

Usage:
    chimerax --nogui --script "scripts/dssp_summary.py <manifest.txt> <output.csv>"

<manifest.txt>: one PDB path per line (blank lines/lines starting with # ignored).
<output.csv>:   fname,cluster,chain,resnum,restype,ss_type (coil/helix/strand)

Runs all PDBs in ONE ChimeraX session (open -> dssp -> record -> close, repeat)
to avoid per-file ChimeraX startup overhead -- much faster than one process
per file for a few hundred structures.
"""

import csv
import sys
from pathlib import Path

from chimerax.core.commands import run

SS_LABELS = {0: "coil", 1: "helix", 2: "strand"}


def main(session):
    if len(sys.argv) < 3:
        session.logger.error(
            "Usage: chimerax --nogui --script dssp_summary.py <manifest.txt> <output.csv>"
        )
        raise SystemExit(1)

    manifest_path = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pdb_paths = [
        Path(line.strip())
        for line in manifest_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    session.logger.info(f"[dssp_summary] {len(pdb_paths)} structure(s) to process")

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fname", "cluster", "chain", "resnum", "restype", "ss_type"])

        for i, pdb_path in enumerate(pdb_paths, start=1):
            if not pdb_path.exists():
                session.logger.warning(f"[dssp_summary] missing, skipping: {pdb_path}")
                continue

            # fname/cluster from the pipeline's own path layout:
            # results/<complex>/minimized/<cluster>/<fname>/<fname>.pdb
            fname = pdb_path.stem
            cluster = pdb_path.parent.parent.name

            run(session, f"open {str(pdb_path)!r}")
            run(session, "dssp")
            structure = session.models[-1]

            for r in structure.residues:
                writer.writerow([fname, cluster, r.chain_id, r.number, r.name,
                                  SS_LABELS.get(r.ss_type, "unknown")])

            run(session, "close")

            if i % 50 == 0:
                session.logger.info(f"[dssp_summary] {i}/{len(pdb_paths)} done")
                fh.flush()

    session.logger.info(f"[dssp_summary] wrote {out_csv}")
    run(session, "quit")


main(session)  # noqa: F821
