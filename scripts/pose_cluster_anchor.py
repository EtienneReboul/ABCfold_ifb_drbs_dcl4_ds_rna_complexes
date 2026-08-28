#!/usr/bin/env python3
"""
scripts/pose_cluster_anchor.py
================================
Stage 3a of the ABCfold IFB DRB2/DCL4/dsRNA pipeline: rigid-anchor pose
clustering, pooled across every backend ABCfold ran (AlphaFold3, Boltz-2,
Chai-1, OpenFold3, Protenix, RosettaFold3) and every seed.

Forks ../../../NPF-ab-initio-modelling/ABCfold_NPF_pipeline/scripts/
tm_helix_alignment.py's structure (same abcfold_backends discovery, same
iterative Kabsch/Procrustes machinery) but replaces "TM helix Cα of a
single chain" with "rigid core of the complex's anchor chain" (DCL4, or
DRB2 for the anchor-less binary complex — see configs/<complex>.yaml's
anchor_chain), and instead of aligning the whole chain onto itself, applies
the anchor's per-frame rigid transform to the PARTNER chain(s)' Cα and
hierarchically clusters those aligned coordinates — the pose-clustering
approach from the old project's notebooks (../ab_initio_modelling_drbs_dcl4_ds_rna_complexes/
notebooks/*_domain_analysis.ipynb, cells 38-58), generalized here to an
arbitrary anchor/partner chain split and to ABCfold's 6-backend ensemble
instead of just AF3 webserver replicas.

Algorithm:
  1. Extract the anchor chain's Cα from every model.
  2. Iteratively refine a rigid "core" subset of the anchor (Kabsch-align
     each frame's anchor to a running reference, drop the most mobile
     residues each iteration) until the core's median RMSF drops below
     --core-rmsf-target or the core would shrink past --min-core-frac of
     the anchor's own length.
  3. Apply each frame's core-derived (R, t) to that frame's partner
     chain(s) Cα.
  4. Hierarchical clustering (average linkage) on pairwise partner-chain Cα
     RMSD; silhouette-based k selection.

RNA chains are intentionally excluded from the partner pose vector (mixing
protein Cα and RNA P/C1' RMSD isn't meaningful) — they still ride along in
the minimized/PLIP structures downstream, they just don't drive cluster
assignment.

Reads results/metadata/<complex>/model_metadata.parquet (produced by
scripts/compress_abcfold_metadata.py) for the per-model ptm/iptm/
ranking_score used by scripts/select_top_n_per_cluster.py, joined against
each row's own cif_path for the actual Cα coordinates.

Outputs, written to <out-root>/<complex>/:
  - pose_clusters.csv        — backend, seed, sample_index, cif_path, cluster,
                                core_rmsd_to_ref, ptm, iptm, ranking_score
  - rmsf_profile.svg         — per-residue RMSF of the anchor chain after
                                core superposition, core highlighted
  - pose_clusters_pca.svg    — 2D PCA of the aligned partner-chain pose
                                vectors, colored by cluster, marker by backend

Usage:
    python scripts/pose_cluster_anchor.py \\
        --complex             rna_ds_dcl4_drb2_drb4 \\
        --complex-config      configs/rna_ds_dcl4_drb2_drb4.yaml \\
        --abcfold-output-root results/abcfold \\
        --metadata-root       results/metadata \\
        --out-root            results/rna_ds_dcl4_drb2_drb4 \\
        --n-iter 8 --core-rmsf-target 1.5 --min-core-frac 0.4 --max-k 12
"""

import argparse
import json
from pathlib import Path

import gemmi  # pyright: ignore[reportMissingImports]
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import yaml
from scipy.cluster.hierarchy import fcluster, linkage  # pyright: ignore[reportMissingImports]
from scipy.spatial.distance import pdist, squareform  # pyright: ignore[reportMissingImports]
from sklearn.decomposition import PCA  # pyright: ignore[reportMissingImports]
from sklearn.metrics import silhouette_score  # pyright: ignore[reportMissingImports]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--complex", required=True)
    p.add_argument("--complex-config", required=True)
    p.add_argument("--abcfold-output-root", default="results/abcfold")
    p.add_argument("--metadata-root", default="results/metadata")
    p.add_argument("--out-root", required=True)
    p.add_argument("--n-iter", type=int, default=8)
    p.add_argument("--core-rmsf-target", type=float, default=1.5)
    p.add_argument("--min-core-frac", type=float, default=0.4)
    p.add_argument("--partner-rmsf-target", type=float, default=5.0,
                    help="Partner-chain Ca RMSF (A, after anchor-frame alignment) above "
                         "which an atom is dropped from the clustering feature vector -- "
                         "trims disordered/linker regions of the partner chain(s) that "
                         "would otherwise dilute pose-cluster signal (default: 5.0)")
    p.add_argument("--partner-min-frac", type=float, default=0.4,
                    help="Floor on kept partner Ca as a fraction of the partner chain(s)' "
                         "total length, even if more atoms exceed --partner-rmsf-target "
                         "(default: 0.4)")
    p.add_argument("--max-k", type=int, default=12)
    return p.parse_args()


# ── Chain extraction ─────────────────────────────────────────────────────────

def extract_chain_ca(cif_path: Path, chain_id: str):
    """Return (ca_coords [N,3] float32, resids [N] int32) for one named chain."""
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    chain = next((c for c in model if c.name == chain_id), None)
    if chain is None:
        raise KeyError(f"{cif_path}: no chain {chain_id!r} (chains present: "
                        f"{[c.name for c in model]})")
    coords, resids = [], []
    for residue in chain:
        for atom in residue:
            if atom.name == "CA":
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                resids.append(residue.seqid.num)
    return np.array(coords, dtype=np.float32), np.array(resids, dtype=np.int32)


def extract_partner_ca(cif_path: Path, partner_chains: list[str]):
    """Concatenated Cα coordinates for every partner chain, in the given
    fixed order (same order for every model, so the resulting flattened
    vector is comparable across frames), plus a parallel (chain_id, resid)
    label per atom for the RMSF profile plot / trim reporting."""
    per_chain = [extract_chain_ca(cif_path, cid) for cid in partner_chains]
    coords = np.concatenate([c for c, _ in per_chain], axis=0)
    labels = [(cid, int(r)) for cid, (_, resids) in zip(partner_chains, per_chain) for r in resids]
    return coords, labels


# ── Kabsch ─────────────────────────────────────────────────────────────────────

def kabsch(P, Q):
    """Rotation R (3,3) and translation t (3,) such that (R @ P.T).T + t ~= Q."""
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - p_mean, Q - q_mean
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_mean - R @ p_mean
    return R, t


# ── Iterative rigid-core refinement (anchor chain) ──────────────────────────────

def refine_core(anchor_frames: np.ndarray, n_iter: int, rmsf_target: float,
                 min_core_size: int):
    """anchor_frames: (n_models, n_ca, 3). Returns (core_idx, rmsf) where
    rmsf is per-residue RMSF (full anchor length) after the final core
    superposition."""
    reference = anchor_frames[0]
    core_idx = np.arange(anchor_frames.shape[1])
    rmsf = None
    print("[pose_cluster] Iterative anchor core refinement:")
    for it in range(n_iter):
        aligned = np.empty_like(anchor_frames)
        for m in range(len(anchor_frames)):
            R, t = kabsch(anchor_frames[m][core_idx], reference[core_idx])
            aligned[m] = (R @ anchor_frames[m].T).T + t
        mean_struct = aligned.mean(axis=0)
        rmsf = np.sqrt(np.mean(np.sum((aligned - mean_struct) ** 2, axis=2), axis=0))
        core_rmsf = rmsf[core_idx]
        median_rmsf = float(np.median(core_rmsf))
        print(f"  iter {it}: core = {len(core_idx)} residues, median RMSF = {median_rmsf:.2f} A")
        if median_rmsf <= rmsf_target or len(core_idx) <= min_core_size:
            break
        keep = core_rmsf <= np.percentile(core_rmsf, 60)
        core_idx = core_idx[keep]
    return core_idx, rmsf


# ── Partner-chain RMSF trim (disordered linkers, not the core) ─────────────────

def select_stable_partner_atoms(partner_aligned: np.ndarray, rmsf_target: float,
                                 min_frac: float):
    """partner_aligned: (n_models, n_partner_ca, 3), already carrying each
    frame's anchor-derived (R, t) -- i.e. positioned in the anchor's rigid
    frame, which IS the pose signal. Unlike refine_core, this does NOT
    re-superpose on the selected subset: partner chains (DRB2/DRB4 here)
    can carry large disordered/linker stretches that move independently of
    the folded domain actually docking against the anchor, and their high
    RMSF would otherwise dominate the pairwise-RMSD feature vector driving
    clustering/PCA, drowning out genuine pose differences. Single-pass RMSF
    threshold: keep atoms with RMSF <= rmsf_target; if that keeps fewer
    than min_frac of atoms (e.g. a partner that's mostly disordered), fall
    back to keeping the min_frac lowest-RMSF atoms instead, so there's
    always an informative footprint left to cluster on."""
    mean_struct = partner_aligned.mean(axis=0)
    rmsf = np.sqrt(np.mean(np.sum((partner_aligned - mean_struct) ** 2, axis=2), axis=0))
    n_atoms = len(rmsf)
    min_keep = max(1, int(min_frac * n_atoms))
    keep_idx = np.where(rmsf <= rmsf_target)[0]
    if len(keep_idx) < min_keep:
        keep_idx = np.argsort(rmsf)[:min_keep]
    return np.sort(keep_idx), rmsf


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    spec = yaml.safe_load(Path(args.complex_config).read_text())
    anchor_chain = spec["anchor_chain"]
    partner_chains = spec["partner_chains"]

    abcfold_output_root = Path(args.abcfold_output_root)
    metadata_path = Path(args.metadata_root) / args.complex / "model_metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found — run scripts/compress_abcfold_metadata.py "
            f"for {args.complex} first (see worflows/postprocessing/Snakefile)."
        )
    meta = pd.read_parquet(metadata_path)
    print(f"[pose_cluster] {args.complex}: {len(meta)} models in metadata "
          f"(anchor={anchor_chain}, partners={partner_chains})")

    anchor_frames, partner_frames, rows = [], [], []
    ref_anchor_resids = None
    ref_partner_labels = None
    for _, row in meta.iterrows():
        cif_path = abcfold_output_root / row["cif_path"]
        try:
            a_coords, a_resids = extract_chain_ca(cif_path, anchor_chain)
            p_coords, p_labels = extract_partner_ca(cif_path, partner_chains)
        except (KeyError, FileNotFoundError) as e:
            print(f"[pose_cluster] WARNING: skipping {cif_path} ({e})")
            continue

        if ref_anchor_resids is None:
            ref_anchor_resids = a_resids
        elif len(a_coords) != len(ref_anchor_resids):
            print(f"[pose_cluster] WARNING: {cif_path} has {len(a_coords)} anchor Ca "
                  f"(expected {len(ref_anchor_resids)}) - skipping")
            continue

        if ref_partner_labels is None:
            ref_partner_labels = p_labels

        anchor_frames.append(a_coords)
        partner_frames.append(p_coords)
        rows.append({
            "backend": row["backend"], "seed": row["seed"], "sample_index": row["sample_index"],
            "cif_path": row["cif_path"], "ptm": row["ptm"], "iptm": row["iptm"],
            "ranking_score": row["ranking_score"],
        })

    n_used = len(anchor_frames)
    if n_used == 0:
        raise RuntimeError(f"All models were skipped for {args.complex}")
    print(f"[pose_cluster] {args.complex}: {n_used}/{len(meta)} models usable")

    anchor_frames = np.stack(anchor_frames)
    n_partner_ca = partner_frames[0].shape[0]
    if any(p.shape[0] != n_partner_ca for p in partner_frames):
        raise RuntimeError(
            f"Partner chain Ca counts differ across models for {args.complex} — "
            "expected identical sequences/chain composition for every model."
        )

    min_core_size = max(1, int(args.min_core_frac * anchor_frames.shape[1]))
    core_idx, rmsf = refine_core(anchor_frames, args.n_iter, args.core_rmsf_target, min_core_size)

    reference = anchor_frames[0]
    core_rmsds, partner_aligned = [], []
    for m in range(n_used):
        R, t = kabsch(anchor_frames[m][core_idx], reference[core_idx])
        aligned_core = (R @ anchor_frames[m][core_idx].T).T + t
        core_rmsds.append(float(np.sqrt(np.mean(np.sum((aligned_core - reference[core_idx]) ** 2, axis=1)))))
        partner_aligned.append((R @ partner_frames[m].T).T + t)
    partner_aligned = np.stack(partner_aligned)

    partner_core_idx, partner_rmsf = select_stable_partner_atoms(
        partner_aligned, args.partner_rmsf_target, args.partner_min_frac
    )
    print(
        f"[pose_cluster] partner trim: kept {len(partner_core_idx)}/{n_partner_ca} Ca "
        f"(RMSF<={args.partner_rmsf_target}A, floor={args.partner_min_frac:.0%}) "
        f"-- dropped disordered/linker atoms from the clustering feature vector"
    )
    X_pose = partner_aligned[:, partner_core_idx, :].reshape(n_used, -1)

    # ── Hierarchical clustering + silhouette k-selection ────────────────────────
    def ca_rmsd_pdist(coord_matrix, n_atoms):
        return pdist(coord_matrix) / np.sqrt(n_atoms)

    pose_dist_condensed = ca_rmsd_pdist(X_pose, len(partner_core_idx))
    pose_dist_matrix = squareform(pose_dist_condensed)
    Z = linkage(pose_dist_condensed, method="average")

    max_k = min(args.max_k, n_used // 2)
    sil_scores = {}
    for k in range(2, max_k + 1):
        labels_k = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels_k)) < 2:
            continue
        sil_scores[k] = silhouette_score(pose_dist_matrix, labels_k, metric="precomputed")

    if sil_scores:
        best_k = max(sil_scores, key=sil_scores.get)
        cluster_labels = fcluster(Z, t=best_k, criterion="maxclust")
        print(f"[pose_cluster] silhouette per k: {({k: round(v, 3) for k, v in sil_scores.items()})}")
        print(f"[pose_cluster] selected k={best_k} (silhouette={sil_scores[best_k]:.3f})")
    else:
        best_k = 1
        cluster_labels = np.ones(n_used, dtype=int)
        print("[pose_cluster] WARNING: silhouette selection found no valid k>=2 "
              "(too few models?) — assigning everything to cluster 1")

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PCA fit computed here (not down by the plotting code) so pc1/pc2 can be
    # persisted into pose_clusters.csv -- lets notebooks/ re-explore this
    # exact embedding (re-cluster with GMM/HDBSCAN, filter, etc.) without
    # recomputing the anchor-Kabsch + partner-trim pipeline from raw CIFs.
    n_comp = min(2, X_pose.shape[0] - 1, X_pose.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pcs = pca.fit_transform(X_pose)
    var_ratio = pca.explained_variance_ratio_
    pc1_var = var_ratio[0]
    pc2_var = var_ratio[1] if n_comp > 1 else 0.0
    print(
        f"[pose_cluster] {args.complex}: PCA explained variance -- "
        f"PC1={pc1_var:.1%} PC2={pc2_var:.1%} (sum={pc1_var + pc2_var:.1%})"
    )

    df = pd.DataFrame(rows)
    df["cluster"] = cluster_labels
    df["core_rmsd_to_ref"] = core_rmsds
    df["pc1"] = pcs[:, 0]
    df["pc2"] = pcs[:, 1] if n_comp > 1 else 0.0
    df.attrs["pc1_explained_variance"] = pc1_var
    df.attrs["pc2_explained_variance"] = pc2_var
    df.to_csv(out_dir / "pose_clusters.csv", index=False)
    (out_dir / "pose_clusters_pca_variance.json").write_text(
        json.dumps({"pc1_explained_variance": float(pc1_var),
                    "pc2_explained_variance": float(pc2_var)}, indent=2) + "\n"
    )
    print(f"[pose_cluster] {args.complex}: {n_used} models -> {out_dir / 'pose_clusters.csv'}")
    print(df["cluster"].value_counts().sort_index().rename("n_models"))

    # ── RMSF profile SVG (anchor core + partner trim, one panel each) ──────────
    fig, (ax_anchor, ax_partner) = plt.subplots(2, 1, figsize=(9, 6.0))

    ax_anchor.plot(ref_anchor_resids, rmsf, color="#888888", lw=1, label="anchor RMSF")
    core_mask = np.isin(np.arange(anchor_frames.shape[1]), core_idx)
    ax_anchor.scatter(ref_anchor_resids[core_mask], rmsf[core_mask], s=6, color="#C44E52", label="final rigid core")
    ax_anchor.set_xlabel(f"{anchor_chain} residue")
    ax_anchor.set_ylabel("RMSF after core\nsuperposition (A)")
    ax_anchor.set_title(f"anchor ({anchor_chain}) conservation across {n_used} models")
    ax_anchor.legend(frameon=False, fontsize=8)

    partner_x = np.arange(n_partner_ca)
    partner_kept_mask = np.isin(partner_x, partner_core_idx)
    ax_partner.plot(partner_x, partner_rmsf, color="#888888", lw=1, label="partner RMSF (anchor-frame)")
    ax_partner.scatter(partner_x[partner_kept_mask], partner_rmsf[partner_kept_mask],
                        s=6, color="#4C72B0", label="kept (clustering feature vector)")
    ax_partner.axhline(args.partner_rmsf_target, color="#C44E52", lw=0.8, ls="--",
                        label=f"target ({args.partner_rmsf_target}A)")
    # mark where the concatenated partner chain identity changes
    boundaries = [i for i in range(1, n_partner_ca) if ref_partner_labels[i][0] != ref_partner_labels[i - 1][0]]
    for b in boundaries:
        ax_partner.axvline(b, color="black", lw=0.6, ls=":")
    tick_pos, tick_lab = [], []
    start = 0
    for b in boundaries + [n_partner_ca]:
        mid = (start + b) // 2
        tick_pos.append(mid)
        tick_lab.append(ref_partner_labels[mid][0])
        start = b
    ax_partner.set_xticks(tick_pos)
    ax_partner.set_xticklabels(tick_lab)
    ax_partner.set_xlabel("partner chain (concatenated)")
    ax_partner.set_ylabel("RMSF in anchor\nframe (A)")
    ax_partner.set_title(
        f"partner ({', '.join(partner_chains)}) mobility -- "
        f"{len(partner_core_idx)}/{n_partner_ca} Ca kept for clustering"
    )
    ax_partner.legend(frameon=False, fontsize=8)

    fig.suptitle(f"{args.complex}: RMSF analysis ({n_used} models)")
    fig.tight_layout()
    fig.savefig(out_dir / "rmsf_profile.svg")
    plt.close(fig)

    # ── Pose-cluster PCA scatter SVG (colored by cluster, marker by backend) ────
    # pcs/pc1_var/pc2_var already computed above (needed there to persist
    # pc1/pc2 into pose_clusters.csv before writing it).
    fig, ax = plt.subplots(figsize=(6, 5.5))
    backends = sorted(df["backend"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_of = {b: markers[i % len(markers)] for i, b in enumerate(backends)}
    cmap = plt.get_cmap("tab10")
    for cluster in sorted(df["cluster"].unique()):
        for backend in backends:
            sel = (df["cluster"] == cluster) & (df["backend"] == backend)
            if not sel.any():
                continue
            ax.scatter(pcs[sel.to_numpy(), 0], pcs[sel.to_numpy(), 1] if n_comp > 1 else np.zeros(sel.sum()),
                       color=cmap(int(cluster) % 10), marker=marker_of[backend],
                       s=30, edgecolor="white", linewidth=0.4,
                       label=f"cluster {cluster} / {backend}")
    ax.set_xlabel(f"PC1 ({pc1_var:.1%} var)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1%} var)" if n_comp > 1 else "")
    ax.set_title(
        f"{args.complex}: pose clusters (partner chains, anchor={anchor_chain})\n"
        f"PC1+PC2 explain {pc1_var + pc2_var:.1%} of variance"
    )
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "pose_clusters_pca.svg")
    plt.close(fig)

    print(f"[pose_cluster] {args.complex}: SVGs written to {out_dir}")


if __name__ == "__main__":
    main()
