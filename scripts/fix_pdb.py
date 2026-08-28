#!/usr/bin/env python3
"""
fix_pdb.py – pdb4amber post-processor for the PLIP pipeline.
=============================================================
Called by Snakemake rule `fix_pdb` (step 2b) after pdb4amber has
canonicalised residue/atom names.

Root cause of the PLIP/OpenBabel segfault (exit 139)
------------------------------------------------------
OpenBabel crashes in its C++ protonation layer (addh()) when it encounters
a broken RNA/DNA backbone that has no TER record separating the fragments.
It walks the bond graph to assign hybridisation before adding H; a missing
phosphodiester linkage causes it to misassign bond orders, leading to a
null-pointer dereference.

pdb4amber correctly emits TER records at gaps, but PDBFixer rewrites the
topology from scratch and strips them.  This script therefore:

  1. Runs PDBFixer ONLY when pdb4amber actually found missing heavy atoms
     (skips it entirely otherwise – no point paying the cost and losing TER).
  2. After any topology rewrite, re-inserts TER records at backbone gaps
     (Cα–Cα > 4 Å for protein; P–P > 8 Å for RNA/DNA).
  3. Does NOT add hydrogens – PLIP re-protonates the structure itself.

Usage
-----
    python scripts/fix_pdb.py \\
        --input   model_amber.pdb \\
        --output  model_fixed.pdb \\
        [--ca-gap-threshold   4.0]   # Å, protein chain-break detection
        [--p-gap-threshold    8.0]   # Å, nucleic-acid chain-break detection

Exit codes
----------
    0 – success
    1 – input file not found or fatal error
"""

import argparse
import math
import sys
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a pdb4amber PDB: optionally run PDBFixer for missing "
            "heavy atoms, then reinsert TER records at backbone gaps so that "
            "OpenBabel does not segfault during PLIP protonation."
        )
    )
    parser.add_argument("--input",  required=True,  help="Input PDB (pdb4amber output)")
    parser.add_argument("--output", required=True,  help="Output PDB (fixed, no hydrogens)")
    parser.add_argument(
        "--ca-gap-threshold", type=float, default=4.0,
        help="Cα-Cα distance (A) above which a protein chain break is inserted (default 4.0)",
    )
    parser.add_argument(
        "--p-gap-threshold", type=float, default=8.0,
        help="P-P distance (A) above which an RNA/DNA chain break is inserted (default 8.0)",
    )
    return parser.parse_args()


# ── PDB parsing helpers ───────────────────────────────────────────────────────

def parse_pdb_lines(path: Path):
    """Read a PDB file and return a list of raw line strings (newlines preserved)."""
    return path.read_text().splitlines(keepends=True)


def _xyz(line: str):
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _atom_name(line: str) -> str:
    return line[12:16].strip()


def _chain_id(line: str) -> str:
    return line[21]


def _resseq(line: str) -> int:
    return int(line[22:26])


def _resname(line: str) -> str:
    return line[17:20].strip()


def _is_nucleic(resname: str) -> bool:
    """Return True if the residue looks like an RNA or DNA nucleotide."""
    return resname.upper() in {
        "A", "G", "C", "U",           # RNA (pdb4amber canonical)
        "RA", "RG", "RC", "RU",        # RNA (alternate notation)
        "DA", "DG", "DC", "DT", "DU",  # DNA
        "ADE", "GUA", "CYT", "URA", "THY",
    }


# ── TER insertion ─────────────────────────────────────────────────────────────

def insert_ter_at_gaps(lines: list, ca_threshold: float, p_threshold: float) -> list:
    """
    Scan ATOM/HETATM records for backbone gaps and insert TER lines.

    Strategy
    --------
    - For protein residues  : watch Ca atoms; gap if Ca-Ca > ca_threshold
    - For nucleic residues  : watch P  atoms; gap if P-P   > p_threshold
    - A chain-ID change always triggers TER (belt-and-suspenders).
    - Existing TER records are preserved; we never insert a duplicate.

    Returns a new list of lines with TER inserted where needed.
    """
    out         = []
    last_ca     = None   # (chain_id, resseq, xyz) of last Ca seen
    last_p      = None   # (chain_id, resseq, xyz) of last P  seen
    last_chain  = None
    prev_record = None

    def make_ter(chain_id: str) -> str:
        return f"TER       chain {chain_id}\n"

    for line in lines:
        record = line[:6].strip()

        if record in ("ATOM", "HETATM"):
            chain  = _chain_id(line)
            aname  = _atom_name(line)
            rname  = _resname(line)
            resseq = _resseq(line)
            xyz    = _xyz(line)
            is_nuc = _is_nucleic(rname)

            # Chain-ID change → insert TER before this atom
            if last_chain is not None and chain != last_chain:
                if prev_record != "TER":
                    out.append(make_ter(last_chain))
                    print(f"[fix_pdb]   TER inserted: chain change {last_chain} -> {chain}")
                last_ca = last_p = None

            # Protein backbone gap
            if aname == "CA" and not is_nuc:
                if last_ca is not None and last_ca[0] == chain:
                    d = _dist(last_ca[2], xyz)
                    if d > ca_threshold:
                        out.append(make_ter(chain))
                        print(f"[fix_pdb]   TER inserted: Ca gap {d:.2f} A at chain {chain} res {resseq}")
                        last_ca = last_p = None
                last_ca = (chain, resseq, xyz)

            # Nucleic-acid backbone gap
            if aname == "P" and is_nuc:
                if last_p is not None and last_p[0] == chain:
                    d = _dist(last_p[2], xyz)
                    if d > p_threshold:
                        out.append(make_ter(chain))
                        print(f"[fix_pdb]   TER inserted: P gap {d:.2f} A at chain {chain} res {resseq}")
                        last_ca = last_p = None
                last_p = (chain, resseq, xyz)

            last_chain  = chain
            prev_record = record

        elif record == "TER":
            last_ca = last_p = None
            prev_record = "TER"

        else:
            prev_record = record

        out.append(line)

    return out


# ── PDBFixer stage (conditional) ──────────────────────────────────────────────

def count_missing_heavy_atoms(pdb_path: Path) -> int:
    """Query PDBFixer for missing heavy atom count without modifying anything."""
    try:
        from pdbfixer import PDBFixer
    except ImportError:
        return 0  # can't check -> assume none, skip PDBFixer

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    return sum(len(v) for v in fixer.missingAtoms.values())


def run_pdbfixer(input_path: Path, output_path: Path) -> None:
    """Add missing heavy atoms with PDBFixer and write to output_path."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    print(f"[fix_pdb] PDBFixer: loading {input_path}")
    fixer = PDBFixer(filename=str(input_path))

    # Drop terminal missing residues (disordered tails -> worse than absent)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    for key in list(fixer.missingResidues.keys()):
        chain_idx, res_idx = key
        n_res = sum(1 for _ in chains[chain_idx].residues())
        if res_idx == 0 or res_idx >= n_res:
            del fixer.missingResidues[key]

    fixer.findMissingAtoms()
    n = sum(len(v) for v in fixer.missingAtoms.values())
    print(f"[fix_pdb] PDBFixer: adding {n} missing heavy atom(s)")
    fixer.addMissingAtoms()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)
    print(f"[fix_pdb] PDBFixer: written {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    inp  = Path(args.input)
    out  = Path(args.output)

    if not inp.exists():
        print(f"[fix_pdb] ERROR: input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: run PDBFixer only if there are missing heavy atoms ──────────
    print("[fix_pdb] Checking for missing heavy atoms ...")
    n_missing = count_missing_heavy_atoms(inp)

    if n_missing == 0:
        print(
            "[fix_pdb] No missing heavy atoms -> skipping PDBFixer "
            "(preserves TER records that prevent OpenBabel segfault)"
        )
        working_path = inp
    else:
        print(f"[fix_pdb] {n_missing} missing heavy atom(s) -> running PDBFixer")
        pdbfixer_tmp = out.with_suffix(".pdbfixer_tmp.pdb")
        try:
            run_pdbfixer(inp, pdbfixer_tmp)
        except Exception as exc:
            print(f"[fix_pdb] ERROR: PDBFixer failed – {exc}", file=sys.stderr)
            sys.exit(1)
        working_path = pdbfixer_tmp

    # ── Stage 2: reinsert TER records at backbone gaps ───────────────────────
    # This is always run, even when PDBFixer was skipped, because pdb4amber
    # may have missed gaps or the source PDB may lack TER entirely.
    print("[fix_pdb] Scanning backbone for gaps and reinserting TER records ...")
    lines = parse_pdb_lines(working_path)
    fixed = insert_ter_at_gaps(
        lines,
        ca_threshold=args.ca_gap_threshold,
        p_threshold=args.p_gap_threshold,
    )

    out.write_text("".join(fixed))
    print(f"[fix_pdb] Written  : {out}")

    # Clean up PDBFixer temp file
    if n_missing > 0:
        working_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()