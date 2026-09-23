"""
Extract a clean, CALVADOS-ready PDB (+ per-chain FASTA) from a minimized
ABCfold model.

CALVADOS reads a component's structure with MDAnalysis (calvados.sequence.
seq_from_pdb / calvados.build.geometry_from_pdb), treating each PDB chain as
an MDAnalysis "segment". It infers chain breaks from those segments, so the
only requirement here is a plain protein-only PDB with clean, standard
residue names and one chain ID per polypeptide -- which the minimized ABCfold
output already is. This script just copies the ATOM records for the
requested chains into the CALVADOS input folder and writes one FASTA file
per chain (used later by predict_disorder.py).

Usage:
    python extract_chains.py --input <minimized.pdb> --outdir <input_dir> --name <sysname>
"""

import argparse
from pathlib import Path

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="minimized PDB with chains to extract")
    p.add_argument("--outdir", required=True, help="CALVADOS input/ directory")
    p.add_argument("--name", required=True, help="CALVADOS component/system name")
    p.add_argument("--chains", default="A,B", help="comma-separated chain IDs to keep, in order")
    return p.parse_args()


def main():
    args = parse_args()
    chains = args.chains.split(",")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lines = Path(args.input).read_text().splitlines()

    kept_atom_lines = []
    seqs = {c: [] for c in chains}
    seen_res = {c: set() for c in chains}

    for line in lines:
        if not line.startswith("ATOM"):
            continue
        chain_id = line[21]
        if chain_id not in chains:
            continue
        kept_atom_lines.append(line)

        resname = line[17:20].strip()
        resseq = line[22:26].strip()
        key = (chain_id, resseq)
        if key not in seen_res[chain_id]:
            seen_res[chain_id].add(key)
            seqs[chain_id].append(AA3TO1.get(resname, "X"))

    out_pdb = outdir / f"{args.name}.pdb"
    with open(out_pdb, "w") as f:
        for chain_id in chains:
            for line in kept_atom_lines:
                if line[21] == chain_id:
                    f.write(line + "\n")
            f.write("TER\n")
        f.write("END\n")

    fasta_path = outdir / f"{args.name}_chains.fasta"
    with open(fasta_path, "w") as f:
        for chain_id in chains:
            seq = "".join(seqs[chain_id])
            f.write(f">{args.name}_{chain_id}\n{seq}\n")
            print(f"chain {chain_id}: {len(seq)} residues")

    print(f"wrote {out_pdb}")
    print(f"wrote {fasta_path}")


if __name__ == "__main__":
    main()
