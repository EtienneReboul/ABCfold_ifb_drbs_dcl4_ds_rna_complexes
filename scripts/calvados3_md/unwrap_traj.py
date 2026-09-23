"""
Undo periodic-boundary wrapping in a CALVADOS3 trajectory for visualization.

OpenMM's DCD reporter wraps each particle independently back into the
primary unit cell every frame. For a single-molecule system in a big box
(our case: one component, box=45 nm), that means once part of a long
disordered tail drifts past the box edge, it gets teleported to the opposite
face while the rest of the chain doesn't move -- the physics is unaffected
(CALVADOS itself uses minimum-image nonbonded distances), but naive
rendering of raw frame coordinates draws a bogus straight line all the way
across the box between consecutive residues that are actually adjacent in
real space.

This script "unwraps" each frame in place: walking each of the two chains
(split at --chain-boundary) in bonded order, whenever consecutive residues
are more than half a box length apart in any dimension, the rest of that
chain is shifted by one box length in that dimension. Purely a
visualization/analysis fix -- does not touch the original .dcd.

Usage:
    python unwrap_traj.py --rundir <replicate_dir> --chain-boundary 435 --box-nm 45.0
"""

import argparse
import glob

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDWriter


def unwrap_chain(positions, box_A, start, end):
    """ In-place sequential unwrap of positions[start:end] (0-based, exclusive end). """
    for i in range(start + 1, end):
        delta = positions[i] - positions[i - 1]
        shift = np.round(delta / box_A) * box_A
        if np.any(shift != 0):
            positions[i:end] -= shift


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", required=True)
    p.add_argument("--chain-boundary", type=int, default=None,
                    help="global 1-based resi where chain B starts; omit for a single-chain system")
    p.add_argument("--box-nm", type=float, required=True, help="cubic box side length in nm (from config.yaml)")
    p.add_argument("--out-suffix", default="_unwrapped")
    return p.parse_args()


def main():
    args = parse_args()
    box_A = args.box_nm * 10.0  # nm -> Angstrom (MDAnalysis/DCD convention)

    top = f"{args.rundir}/top.pdb"
    dcds = sorted(glob.glob(f"{args.rundir}/*.dcd"))
    dcd = [d for d in dcds if args.out_suffix not in d][0]
    out_dcd = dcd.replace(".dcd", f"{args.out_suffix}.dcd")

    u = mda.Universe(top, dcd)
    n_atoms = len(u.atoms)
    boundary = args.chain_boundary - 1 if args.chain_boundary else n_atoms  # 0-based

    with DCDWriter(out_dcd, n_atoms=n_atoms) as writer:
        for ts in u.trajectory:
            pos = ts.positions
            unwrap_chain(pos, box_A, 0, boundary)
            if boundary < n_atoms:
                unwrap_chain(pos, box_A, boundary, n_atoms)
            writer.write(u.atoms)

    print(f"wrote {out_dcd} ({len(u.trajectory)} frames, box={args.box_nm} nm, chain break at resi {args.chain_boundary})")


if __name__ == "__main__":
    main()
