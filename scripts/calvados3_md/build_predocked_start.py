"""
Build a pre-docked CG starting configuration for the protein+RNA CALVADOS3
system, preserving the REAL relative protein-RNA arrangement from a single
ABCfold prediction (as opposed to the earlier run, which started the two
components unbound via topol='random' to test spontaneous binding).

CALVADOS's topol= placement options ('random', 'grid', 'center', ...)
independently re-position each component and explicitly forbid 'center' for
>1 component (calvados/sim.py), so there is no built-in way to start two
different-molecule-type components in a specific joint pose. Instead this
uses calvados.sim's other documented path: restart='pdb' reads positions
directly from a user-supplied PDB (calvados/sim.py's `simulate()`: if
restart=='pdb' and the frestart file exists, `pdb = app.pdbfile.PDBFile(
fcheck_in)` and `simulation.context.setPositions(pdb.positions)` -- purely
ORDER-based, no name/topology matching against calvados's own topology), so
it completely bypasses topol-based placement when the file is present.

This script computes CG bead positions directly (replicating calvados.build.
geometry_from_pdb's per-residue COM for protein and geometry_from_pdb_rna's
backbone/base COM split for RNA) from ALL FOUR chains of one real predicted
structure loaded together in a single coordinate frame (no independent
per-molecule recentering, which is what geometry_from_pdb/_rna do on their
own and would destroy the real relative arrangement) -- then writes a single
1013-bead PDB (789 protein beads + 224 RNA beads, in the same atom order
CALVADOS's own component order produces: protein first, then RNA) shifted so
the combined complex sits at the box center.

Usage:
    python build_predocked_start.py --input <4-chain complex pdb> \
        --protein-chains A,B --rna-chains C,D --box-nm 46.5 \
        --out <predocked_start.pdb>
"""
import argparse
import warnings

import numpy as np
from MDAnalysis import Universe

BACKBONE_ATOMS_RNA = ["1H2'", "1H5'", "2H5'", "2HO'", "C1'", "C2'", "C3'", "C4'", "C5'",
                       "H1'", "H3'", "H4'", "O2'", "O3'", "O5'", "O4'", "OP1", "OP2", "P"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--protein-chains", default="A,B")
    p.add_argument("--rna-chains", default="C,D")
    p.add_argument("--box-nm", type=float, required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    protein_chains = args.protein_chains.split(",")
    rna_chains = args.rna_chains.split(",")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = Universe(args.input)

    protein_pos = []
    for ch in protein_chains:
        sel = u.select_atoms(f"segid {ch} or chainID {ch}")
        for res in sel.residues:
            protein_pos.append(res.atoms.center_of_mass())

    rna_pos = []
    for ch in rna_chains:
        sel = u.select_atoms(f"segid {ch} or chainID {ch}")
        for res in sel.residues:
            backbone = res.atoms.select_atoms("name " + " ".join(BACKBONE_ATOMS_RNA))
            non_backbone = res.atoms.difference(backbone)
            rna_pos.append(backbone.center_of_mass())
            rna_pos.append(non_backbone.center_of_mass())

    protein_pos = np.array(protein_pos)
    rna_pos = np.array(rna_pos)
    print(f"protein beads: {len(protein_pos)}, RNA beads: {len(rna_pos)}")

    combined = np.vstack([protein_pos, rna_pos])
    combined -= combined.mean(0)  # single shared recentering -- preserves relative offset
    box_A = args.box_nm * 10.0
    combined += box_A / 2.0  # shift to box center

    with open(args.out, "w") as f:
        f.write(f"CRYST1{box_A:9.3f}{box_A:9.3f}{box_A:9.3f}{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} P 1           1\n")
        serial = 1
        for i, pos in enumerate(combined):
            name = "CA" if i < len(protein_pos) else ("sP" if (i - len(protein_pos)) % 2 == 0 else "sN")
            resname = "GLY" if i < len(protein_pos) else "s"
            chain = "A" if i < len(protein_pos) else "B"
            resi = (i + 1) if i < len(protein_pos) else (i - len(protein_pos)) // 2 + 1
            f.write(f"ATOM  {serial:5d}  {name:<3s} {resname:<3s} {chain}{resi:4d}    "
                     f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00\n")
            serial += 1
        f.write("END\n")

    print(f"wrote {args.out}: {len(combined)} beads, box={args.box_nm} nm, "
          f"combined COM shifted to box center")


if __name__ == "__main__":
    main()
