"""
PyMOL visualization of the protein+RNA 2-component CALVADOS3 replicate.

top.pdb layout for this system (fixed, from prepare_calvados_with_rna.py):
  chain A = protein complex, resi 1-789 (chain A sub-range 1-434, chain B
            sub-range 435-789, both currently lumped under PDB chain "A")
  chain B = RNA duplex, resi 1-112 local (strand1 = 1-57, strand2 = 58-112,
            2 beads/residue: atom name "sP" backbone, "sN" base)

Re-splits into 4 real PyMOL chains (protein A/B, RNA strand C/D) so explicit
bonds don't cross molecule/strand boundaries (same rationale as
visualize_pymol.py for the protein-only case -- top.pdb's single-chain-per-
component layout doesn't reflect real connectivity, cartoon/bond-guessing
would draw bogus links across chain breaks otherwise).

Usage:
    pymol scripts/calvados3_md/visualize_pymol_with_rna.py -- \\
        results/rna_ds_drb2_drb4/calvados3_md/drb2_drb4_with_rna/replicate_1

    pymol -cq scripts/calvados3_md/visualize_pymol_with_rna.py -- \\
        results/rna_ds_drb2_drb4/calvados3_md/drb2_drb4_with_rna/replicate_1 \\
        --png-dir frames/with_rna/replicate_1 --stride 10
"""

import argparse
import glob
import sys

from pymol import cmd

PROTEIN_DOMAINS = [(8, 132, "marine"), (137, 197, "orange"), (435, 545, "teal"), (740, 789, "purpleblue")]
DISORDER_COLOR = "grey80"
RNA_BACKBONE_COLOR = "grey40"
RNA_BASE_COLOR = "firebrick"


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("rundir")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--png-dir", default=None)
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=900)
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    rundir = args.rundir

    top = f"{rundir}/top.pdb"
    dcds = sorted(glob.glob(f"{rundir}/*.dcd"))
    dcd = dcds[0]
    print(f"loading {top} + {dcd}")

    obj = "sys"
    cmd.load(top, obj)

    # re-split the two lumped PDB chains into 4 real chains: do RNA first
    # (chain B) while it's unambiguous, then protein (chain A)
    cmd.alter(f"{obj} and chain B and resi 58-999", "chain='D'")
    cmd.alter(f"{obj} and chain B and resi 1-57", "chain='C'")
    cmd.alter(f"{obj} and chain A and resi 435-999", "chain='B'")
    cmd.rebuild(obj)

    # protein: bond consecutive CA atoms within each chain (already split, no cross-chain risk)
    def protein_bond(ch, n):
        for i in range(1, n):
            cmd.bond(f"{obj} and chain {ch} and resi {i}", f"{obj} and chain {ch} and resi {i+1}")

    protein_bond("A", 434)
    protein_bond("B", 789 - 434)

    # RNA: bond backbone-backbone consecutive (sP-sP) and each residue's own sP-sN
    def rna_bond(ch, n):
        for i in range(1, n + 1):
            cmd.bond(f"{obj} and chain {ch} and resi {i} and name sP",
                      f"{obj} and chain {ch} and resi {i} and name sN")
            if i < n:
                cmd.bond(f"{obj} and chain {ch} and resi {i} and name sP",
                          f"{obj} and chain {ch} and resi {i+1} and name sP")

    rna_bond("C", 57)
    rna_bond("D", 112 - 57)

    cmd.load_traj(dcd, obj, state=1, interval=args.stride)
    n_states = cmd.count_states(obj)
    print(f"loaded {n_states} states")

    cmd.hide("everything", obj)
    cmd.show("sticks", obj)
    cmd.set("stick_radius", 0.25, obj)
    cmd.show("spheres", obj)
    cmd.set("sphere_scale", 0.35, obj)

    cmd.color(DISORDER_COLOR, f"{obj} and chain A+B")
    for s, e, color in PROTEIN_DOMAINS:
        cmd.color(color, f"{obj} and resi {s}-{e} and chain A+B")
    cmd.color(RNA_BACKBONE_COLOR, f"{obj} and chain C+D and name sP")
    cmd.color(RNA_BASE_COLOR, f"{obj} and chain C+D and name sN")

    if args.smooth_window and n_states > args.smooth_window:
        cmd.smooth(obj, window=args.smooth_window)

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.orient(obj)
    cmd.zoom(obj, buffer=10)

    if args.png_dir:
        import os
        os.makedirs(args.png_dir, exist_ok=True)
        for state in range(1, n_states + 1):
            cmd.frame(state)
            cmd.ray(args.width, args.height)
            cmd.png(f"{args.png_dir}/frame_{state:05d}.png", dpi=150)
            if state % 20 == 0 or state == n_states:
                print(f"rendered {state}/{n_states}")
    else:
        cmd.mset(f"1-{n_states}")
        session = f"{rundir}/sys_with_rna_pymol.pse"
        cmd.save(session)
        print(f"saved session -> {session}")
        print("scrub frames with the slider, or run: mplay")


main()
