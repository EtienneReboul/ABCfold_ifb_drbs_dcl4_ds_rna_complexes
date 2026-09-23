"""
Residue-residue contact-frequency map aggregated across CALVADOS3 replicates,
with a per-domain breakdown to test whether the DRB2-DRB4 interface is
mediated mainly by the folded (AIUPred-restrained) domains or by the
disordered regions.

Uses the RAW (periodic-wrapped) .dcd files, not the *_unwrapped.dcd used for
PyMOL visualization: unwrap_traj.py only makes each chain internally
contiguous for rendering, it does not preserve the correct minimum-image
distance BETWEEN the two chains. Contact distances here are computed with
MDAnalysis's minimum-image convention (`box=`) directly on the raw
coordinates, which is the physically correct way to measure who's touching
whom under periodic boundaries.

A "contact" is defined per-pair from each residue's own CALVADOS3 sigma
(residues_CALVADOS3.csv, 4.5-6.8 A depending on residue type): distance(i,j)
< --scale * 0.5*(sigma_i + sigma_j). scale=1.0 means the two beads' surfaces
are exactly touching (the repulsive-wall distance for that specific pair) --
the natural, physically-grounded "contact" definition for a 1-bead-per-
residue CG model, rather than one flat cutoff applied to every residue type
regardless of size. Calibrated empirically on this system: scale=0.8 ->
~150 non-bonded contacts/frame, 0.9 -> ~350, 1.0 -> ~750, 1.2 -> ~2200; a flat
3 A cutoff (smaller than any bead's own radius) gives essentially zero.

Usage:
    python contact_map.py --rundirs results/.../replicate_1 results/.../replicate_2 results/.../replicate_3 \\
        --domains results/.../input/domains.yaml --nres 789 --chain-boundary 435 \\
        --box-nm 45.0 --outprefix results/.../analysis/rank_415_..._contacts
"""

import argparse
import glob
import re

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_domains_yaml(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    name = lines[0].rstrip(":").strip()
    domains = []
    for line in lines[1:]:
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", line)
        if m:
            domains.append([int(m.group(1)), int(m.group(2))])
    return name, domains


AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def per_residue_sigma_A(top_pdb, fresidues):
    """ Per-bead sigma (Angstrom), in residue order, from CALVADOS3 params. """
    sigma_map = pd.read_csv(fresidues).set_index("one")["sigmas"].to_dict()
    u = mda.Universe(top_pdb)
    sigmas = np.array([sigma_map[AA3TO1[r]] for r in u.residues.resnames])
    return sigmas * 10.0  # nm -> Angstrom


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rundirs", nargs="+", required=True)
    p.add_argument("--domains", required=True)
    p.add_argument("--nres", type=int, required=True)
    p.add_argument("--chain-boundary", type=int, required=True,
                    help="global 1-based resi where chain B starts")
    p.add_argument("--box-nm", type=float, required=True)
    p.add_argument("--fresidues", required=True, help="residues_CALVADOS3.csv (per-residue sigmas)")
    p.add_argument("--scale", type=float, default=1.0,
                    help="contact if distance < scale * 0.5*(sigma_i+sigma_j); 1.0 = beads just touching")
    p.add_argument("--outprefix", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    sysname, domains = load_domains_yaml(args.domains)
    nres = args.nres
    box_A = args.box_nm * 10.0
    box = np.array([box_A, box_A, box_A, 90.0, 90.0, 90.0])

    sigmas = per_residue_sigma_A(f"{args.rundirs[0]}/top.pdb", args.fresidues)
    assert len(sigmas) == nres
    cutoff_matrix = args.scale * 0.5 * (sigmas[:, None] + sigmas[None, :])
    print(f"per-pair sigma-scaled cutoff (scale={args.scale}): "
          f"range {cutoff_matrix.min():.1f}-{cutoff_matrix.max():.1f} A, mean {cutoff_matrix.mean():.1f} A")

    contact_sum = np.zeros((nres, nres), dtype=np.int64)
    total_frames = 0

    for rd in args.rundirs:
        top = f"{rd}/top.pdb"
        dcds = sorted(glob.glob(f"{rd}/*.dcd"))
        raw = [d for d in dcds if "_unwrapped" not in d][0]
        u = mda.Universe(top, raw)
        n = 0
        for ts in u.trajectory:
            d = distance_array(ts.positions, ts.positions, box=box)
            contact_sum += (d < cutoff_matrix)
            n += 1
        total_frames += n
        print(f"{rd}: {n} frames processed (raw={raw})")

    freq = contact_sum / total_frames
    np.fill_diagonal(freq, 0.0)
    np.save(f"{args.outprefix}_full_contact_freq.npy", freq)
    print(f"total frames pooled across replicates: {total_frames}")

    is_folded = np.zeros(nres, dtype=bool)
    for s, e in domains:
        is_folded[s - 1:e] = True

    boundary0 = args.chain_boundary - 1  # 0-based index where chain B starts
    idxA = np.arange(0, boundary0)
    idxB = np.arange(boundary0, nres)

    # ---- inter-chain heatmap (chain A residues x chain B residues) ----
    sub = freq[np.ix_(idxA, idxB)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sub, origin="lower", aspect="auto", cmap="viridis",
                    extent=[idxB[0] + 1, idxB[-1] + 1, idxA[0] + 1, idxA[-1] + 1])
    for s, e in domains:
        if s - 1 in idxA or e - 1 in idxA and s <= boundary0:
            if e <= boundary0:
                ax.axhspan(s, e, color="none", ec="red", lw=1, fill=False)
        if s - 1 >= boundary0:
            ax.axvspan(s, e, color="none", ec="red", lw=1, fill=False)
    ax.set_xlabel("chain B global resi")
    ax.set_ylabel("chain A global resi")
    ax.set_title(f"{sysname}: inter-chain contact frequency "
                 f"(scale={args.scale} x sigma, pooled {len(args.rundirs)} replicates, {total_frames} frames)")
    fig.colorbar(im, ax=ax, label="contact frequency")
    fig.tight_layout()
    fig.savefig(f"{args.outprefix}_interchain_heatmap.png", dpi=150)
    print(f"wrote {args.outprefix}_interchain_heatmap.png")

    # ---- per-domain categorical breakdown (inter-chain only) ----
    def cat_ranges(idx_chain, chain_label):
        chain_domains = [(s, e) for s, e in domains if (s - 1) in idx_chain]
        cats = []
        covered = np.zeros(len(idx_chain), dtype=bool)
        offset = idx_chain[0]
        for i, (s, e) in enumerate(chain_domains, start=1):
            cats.append((f"{chain_label}_dom{i}({s}-{e})", np.arange(s - 1, e) - offset))
            covered[np.arange(s - 1, e) - offset] = True
        cats.append((f"{chain_label}_disordered", np.where(~covered)[0]))
        return cats

    catsA = cat_ranges(idxA, "A")
    catsB = cat_ranges(idxB, "B")

    table = pd.DataFrame(0.0, index=[c[0] for c in catsA], columns=[c[0] for c in catsB])
    for nameA, rowsA in catsA:
        for nameB, rowsB in catsB:
            table.loc[nameA, nameB] = sub[np.ix_(rowsA, rowsB)].sum()

    total_mass = table.values.sum()
    pct_table = 100 * table / total_mass
    print("\ninter-chain contact-frequency MASS by domain pair (sum of per-pair frequencies):")
    print(table.round(2))
    print("\nsame, as % of total inter-chain contact mass:")
    print(pct_table.round(1))

    folded_rows = [c[0] for c in catsA if "disordered" not in c[0]]
    folded_cols = [c[0] for c in catsB if "disordered" not in c[0]]
    disordered_rows = [c[0] for c in catsA if "disordered" in c[0]]
    disordered_cols = [c[0] for c in catsB if "disordered" in c[0]]

    ff = table.loc[folded_rows, folded_cols].values.sum()
    fd = table.loc[folded_rows, disordered_cols].values.sum()
    df_ = table.loc[disordered_rows, folded_cols].values.sum()
    dd = table.loc[disordered_rows, disordered_cols].values.sum()

    print("\n=== summary: what mediates the inter-chain interface? ===")
    print(f"folded-folded     : {ff:8.2f}  ({100*ff/total_mass:5.1f}%)")
    print(f"folded-disordered : {fd:8.2f}  ({100*fd/total_mass:5.1f}%)")
    print(f"disordered-folded : {df_:8.2f}  ({100*df_/total_mass:5.1f}%)")
    print(f"disordered-disord.: {dd:8.2f}  ({100*dd/total_mass:5.1f}%)")
    print(f"-> any folded domain involved : {100*(ff+fd+df_)/total_mass:5.1f}%")
    print(f"-> disordered-disordered only : {100*dd/total_mass:5.1f}%")

    table.to_csv(f"{args.outprefix}_interchain_domain_table.csv")
    pct_table.to_csv(f"{args.outprefix}_interchain_domain_table_pct.csv")
    print(f"\nwrote {args.outprefix}_interchain_domain_table.csv / _pct.csv")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(pct_table.values, cmap="magma")
    ax2.set_xticks(range(len(pct_table.columns)))
    ax2.set_xticklabels(pct_table.columns, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(pct_table.index)))
    ax2.set_yticklabels(pct_table.index, fontsize=8)
    for i in range(pct_table.shape[0]):
        for j in range(pct_table.shape[1]):
            ax2.text(j, i, f"{pct_table.values[i, j]:.1f}", ha="center", va="center",
                      color="white" if pct_table.values[i, j] < pct_table.values.max() / 2 else "black",
                      fontsize=8)
    ax2.set_title(f"{sysname}: inter-chain contact mass by domain pair (%)")
    fig2.colorbar(im2, ax=ax2, label="% of total inter-chain contact mass")
    fig2.tight_layout()
    fig2.savefig(f"{args.outprefix}_interchain_domain_table.png", dpi=150)
    print(f"wrote {args.outprefix}_interchain_domain_table.png")


if __name__ == "__main__":
    main()
