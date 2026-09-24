"""
Extract the dsRNA duplex (chains C+D) from a minimized ABCfold RNA-bound
model into its own CALVADOS-ready PDB, and verify the base-pairing register
between the two strands so the fdomains ranges (paired vs single-stranded
overhang) are correct rather than assumed.

Mirrors extract_chains.py's approach (protein), but RNA resnames are already
single-letter (A/C/G/U) so no 3-to-1 conversion is needed.

Usage:
    python extract_rna_chains.py --input <minimized.pdb> --outdir <input_dir> --name <sysname> --chains C,D
"""

import argparse
from pathlib import Path

COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--chains", default="C,D", help="comma-separated RNA chain IDs, strand1 then strand2")
    return p.parse_args()


def check_register(seq1, seq2):
    """ Reverse-complement seq2 and find the best-aligned offset against seq1.
    Returns (offset, n_match, n_total) for offset=0 (direct 5' alignment) and
    reports the best of a small search window, to catch overhangs at either end. """
    revcomp2 = "".join(COMPLEMENT[b] for b in reversed(seq2))
    results = []
    for offset in range(-10, 11):
        matches = total = 0
        for i, b in enumerate(revcomp2):
            j = i + offset
            if 0 <= j < len(seq1):
                total += 1
                if seq1[j] == b:
                    matches += 1
        if total > 0:
            results.append((offset, matches, total))
    best = max(results, key=lambda r: r[1] / r[2])
    return best, revcomp2


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
            seqs[chain_id].append(resname)

    out_pdb = outdir / f"{args.name}.pdb"
    with open(out_pdb, "w") as f:
        # calvados.build.geometry_from_pdb_rna (unlike the protein version)
        # crashes on a PDB with no CRYST1 record (u.dimensions is None, no
        # guard) -- add a placeholder box so MDAnalysis has dimensions to
        # report; the real simulation box comes from config.yaml, not this.
        f.write("CRYST1  999.000  999.000  999.000  90.00  90.00  90.00 P 1           1\n")
        for chain_id in chains:
            for line in kept_atom_lines:
                if line[21] == chain_id:
                    f.write(line + "\n")
            f.write("TER\n")
        f.write("END\n")

    seq1 = "".join(seqs[chains[0]])
    seq2 = "".join(seqs[chains[1]])
    print(f"strand1 (chain {chains[0]}): {len(seq1)} nt: {seq1}")
    print(f"strand2 (chain {chains[1]}): {len(seq2)} nt: {seq2}")

    (offset, matches, total), revcomp2 = check_register(seq1, seq2)
    print(f"best alignment: offset={offset}, {matches}/{total} bp match "
          f"(revcomp(strand2)={revcomp2})")
    if matches != total or offset != 0:
        print("WARNING: imperfect or offset base-pairing register -- "
              "verify fdomains ranges manually before trusting the automatic ones below")

    n1, n2 = len(seq1), len(seq2)
    paired1_start = max(0, offset) + 1
    paired1_end = min(n1, len(revcomp2) + offset)
    global_offset2 = n1
    print(f"\nsuggested fdomains (global 1-based, strand1=1-{n1}, strand2={n1+1}-{n1+n2}):")
    print(f"  strand1 paired: [{paired1_start}, {paired1_end}]")
    print(f"  strand2 paired: [{global_offset2+1}, {global_offset2+n2}]")
    if paired1_start > 1:
        print(f"  strand1 5' overhang (unpaired): [1, {paired1_start-1}]")
    if paired1_end < n1:
        print(f"  strand1 3' overhang (unpaired): [{paired1_end+1}, {n1}]")

    print(f"\nwrote {out_pdb}")


if __name__ == "__main__":
    main()
