"""
Quick post-hoc check of a CALVADOS3 replicate: did the AIUPred-restrained
folded domains keep their fold, and did the disordered regions actually move?

For each domain listed in domains.yaml, computes per-frame CA (COM-bead) RMSD
to frame 0 after optimal superposition on that domain alone (folded domains
are restrained independently, not to each other -- see prepare_calvados.py
docstring -- so each domain must be checked on its own, not as one rigid
group). For the disordered stretches (everything outside domains.yaml),
reports radius of gyration over time as a measure of conformational sampling.

Usage:
    python analyze_run.py --rundir <results/.../calvados3_md/<name>/replicate_1> \
        --domains <results/.../input/domains.yaml> --nres 789 --outprefix <name>_replicate_1
"""

import argparse
import glob

import yaml
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.align import rotation_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", required=True)
    p.add_argument("--domains", required=True, help="domains.yaml")
    p.add_argument("--nres", type=int, required=True, help="total residues/beads in the system")
    p.add_argument("--outprefix", required=True)
    return p.parse_args()


def disordered_ranges(domains, nres):
    """ 1-based inclusive ranges NOT covered by any folded domain. """
    covered = np.zeros(nres, dtype=bool)
    for s, e in domains:
        covered[s - 1:e] = True
    ranges = []
    i = 0
    while i < nres:
        if not covered[i]:
            j = i
            while j < nres and not covered[j]:
                j += 1
            ranges.append((i + 1, j))
            i = j
        else:
            i += 1
    return ranges


def main():
    args = parse_args()

    with open(args.domains) as f:
        domainbib = yaml.safe_load(f)
    sysname = list(domainbib.keys())[0]
    domains = domainbib[sysname]

    top = f"{args.rundir}/top.pdb"
    dcd = glob.glob(f"{args.rundir}/*.dcd")[0]
    u = mda.Universe(top, dcd)
    print(f"{args.rundir}: {len(u.trajectory)} frames, {len(u.atoms)} beads")

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    for didx, (s, e) in enumerate(domains):
        ag = u.atoms[s - 1:e]
        u.trajectory[0]
        ref = ag.positions.copy()
        rmsds = []
        for ts in u.trajectory:
            _, r = rotation_matrix(ag.positions - ag.positions.mean(0), ref - ref.mean(0))
            rmsds.append(r)
        print(f"domain {s}-{e}: RMSD first={rmsds[0]:.2f} last={rmsds[-1]:.2f} "
              f"mean={np.mean(rmsds):.2f} max={max(rmsds):.2f} A")
        axes[0].plot(rmsds, label=f"domain {s}-{e}")

    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("Cα RMSD to frame 0 (Å)")
    axes[0].set_title(f"{sysname}: folded-domain RMSD (restraint check)")
    axes[0].legend(fontsize=8)

    idr_ranges = disordered_ranges(domains, args.nres)
    for s, e in idr_ranges:
        if e - s + 1 < 3:
            continue
        ag = u.atoms[s - 1:e]
        rgs = [ag.radius_of_gyration() for ts in u.trajectory]
        print(f"disordered {s}-{e} ({e - s + 1} aa): Rg first={rgs[0]:.1f} last={rgs[-1]:.1f} "
              f"mean={np.mean(rgs):.1f} std={np.std(rgs):.1f} A")
        axes[1].plot(rgs, label=f"IDR {s}-{e}")

    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("Rg (Å)")
    axes[1].set_title(f"{sysname}: disordered-region Rg over time")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_png = f"{args.outprefix}_check.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
