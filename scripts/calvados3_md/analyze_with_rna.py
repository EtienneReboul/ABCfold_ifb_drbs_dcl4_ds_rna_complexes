"""
Post-hoc analysis for the protein+RNA 2-component CALVADOS3 system:
RNA duplex integrity (per-base-pair distance), protein domain RMSD, IDR Rg,
and protein-RNA / inter-protein-chain distance as a rough bound-fraction
proxy. Bead layout (fixed for this system): protein beads global 0-788
(same numbering as the protein-only runs), RNA beads global 789-1012
(112 nt x 2 beads; nucleotide g (1-based) -> backbone bead 788+2*(g-1)+...
see base_idx below).

Usage:
    python analyze_with_rna.py --rundir <replicate_dir> --outprefix <prefix>
"""
import argparse
import glob
import os
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.align import rotation_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROTEIN_DOMAINS = [(8, 132), (137, 197), (435, 545), (740, 789)]
IDR_RANGES = [(1, 7), (133, 136), (198, 434), (546, 739)]
BP_PAIRS = [(1, 112), (14, 99), (28, 85), (42, 71), (55, 58)]
PROTEIN_N = 789


def base_idx(g):
    return 788 + 2 * g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", required=True)
    p.add_argument("--outprefix", required=True)
    args = p.parse_args()

    top = f"{args.rundir}/top.pdb"
    if not os.path.isfile(top):
        # restart='pdb' runs (pre-docked starts) don't write top.pdb; the
        # end-of-run checkpoint.pdb has the same atom count/order and works
        # fine as a topology reference for MDAnalysis
        top = f"{args.rundir}/checkpoint.pdb"
    dcd = glob.glob(f"{args.rundir}/*.dcd")[0]
    u = Universe(top, dcd)
    n = len(u.trajectory)
    print(f"{args.rundir}: {n} frames, {len(u.atoms)} beads")

    protein_ag = u.atoms[0:PROTEIN_N]
    rna_ag = u.atoms[PROTEIN_N:]

    # --- RNA base-pair distances ---
    u.trajectory[0]
    pos0 = u.atoms.positions.copy()
    bp_series = {p_: [] for p_ in BP_PAIRS}
    for ts in u.trajectory:
        pos = u.atoms.positions
        for g1, g2 in BP_PAIRS:
            i, j = base_idx(g1), base_idx(g2)
            bp_series[(g1, g2)].append(np.linalg.norm(pos[i] - pos[j]))
    print("\nRNA base-pair distances (A):")
    for pair, arr in bp_series.items():
        arr = np.array(arr)
        print(f"  nt {pair[0]}-{pair[1]}: mean={arr.mean():.2f} std={arr.std():.2f} max={arr.max():.2f}")

    # --- protein domain RMSD ---
    print("\nprotein domain RMSD (A):")
    for s, e in PROTEIN_DOMAINS:
        ag = u.atoms[s - 1:e]
        u.trajectory[0]
        ref = ag.positions.copy()
        rr = []
        for ts in u.trajectory:
            _, r = rotation_matrix(ag.positions - ag.positions.mean(0), ref - ref.mean(0))
            rr.append(r)
        print(f"  domain {s}-{e}: mean={np.mean(rr):.2f} last={rr[-1]:.2f} max={max(rr):.2f}")

    # --- IDR Rg ---
    print("\nIDR Rg (A):")
    idr_rg = {}
    for s, e in IDR_RANGES:
        if e - s + 1 < 3:
            continue
        ag = u.atoms[s - 1:e]
        rgs = [ag.radius_of_gyration() for ts in u.trajectory]
        idr_rg[(s, e)] = np.array(rgs)
        print(f"  IDR {s}-{e} ({e - s + 1} aa): mean={np.mean(rgs):.1f} std={np.std(rgs):.1f}")

    # --- protein-RNA and inter-chain-protein distances ---
    chainA_ag = u.atoms[0:434]
    chainB_ag = u.atoms[434:789]
    prot_rna_dist = []
    interchain_dist = []
    for ts in u.trajectory:
        prot_rna_dist.append(np.linalg.norm(protein_ag.center_of_mass() - rna_ag.center_of_mass()))
        interchain_dist.append(np.linalg.norm(chainA_ag.center_of_mass() - chainB_ag.center_of_mass()))
    prot_rna_dist = np.array(prot_rna_dist)
    interchain_dist = np.array(interchain_dist)
    for cutoff in [80, 100, 150]:
        frac = (prot_rna_dist < cutoff).mean()
        print(f"protein-RNA COM distance < {cutoff} A: {100*frac:.1f}% of frames")
    print(f"protein-RNA COM distance: mean={prot_rna_dist.mean():.1f} min={prot_rna_dist.min():.1f} max={prot_rna_dist.max():.1f}")
    print(f"chainA-chainB COM distance: mean={interchain_dist.mean():.1f} min={interchain_dist.min():.1f} max={interchain_dist.max():.1f}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for pair, arr in bp_series.items():
        axes[0].plot(arr, label=f"nt {pair[0]}-{pair[1]}", alpha=0.7)
    axes[0].set_ylabel("base-pair distance (A)")
    axes[0].set_title("RNA duplex integrity")
    axes[0].legend(fontsize=7)

    for (s, e), rgs in idr_rg.items():
        if e - s > 50:
            axes[1].plot(rgs, label=f"IDR {s}-{e}")
    axes[1].set_ylabel("Rg (A)")
    axes[1].set_title("IDR Rg")
    axes[1].legend(fontsize=8)

    axes[2].plot(prot_rna_dist, label="protein-RNA COM dist")
    axes[2].plot(interchain_dist, label="chainA-chainB COM dist")
    axes[2].set_ylabel("distance (A)")
    axes[2].set_xlabel("frame")
    axes[2].set_title("binding distances")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{args.outprefix}_check.png", dpi=150)
    print(f"\nwrote {args.outprefix}_check.png")


if __name__ == "__main__":
    main()
