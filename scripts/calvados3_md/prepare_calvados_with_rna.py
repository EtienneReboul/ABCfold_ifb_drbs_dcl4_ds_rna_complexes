"""
Build CALVADOS3 config.yaml / components.yaml / run.py for a 3-component
system: the DRB2/DRB4 protein heterodimer (as a single restrained component,
see prepare_calvados.py) plus the dsRNA duplex they bind (a second component,
molecule_type='rna'), to compare disordered-region and inter-chain contact
behavior against the existing protein-only triplicate.

dsRNA mechanism (traced from calvados/components.py's Rna class): giving the
RNA component restraint=True, restraint_type='harmonic' and an fdomains file
listing the base-paired nucleotide ranges does TWO things at once, driven by
the same domain list: (1) turns on the elastic-network harmonic restraint
between paired beads within cutoff_restr, holding the duplex geometry, and
(2) automatically assigns the low-lambda 'SRN' base bead type (reduced
"stickiness", representing a buried/paired base) to every nucleotide inside
those ranges, vs. the high-lambda 'RNA' base bead type outside them (exposed/
single-stranded). No custom code needed beyond a correct fdomains file --
see extract_rna_chains.py for how the paired/overhang ranges were determined
from the actual reverse-complement alignment of the two strands (not
assumed).

use_com=True (not the CALVADOS example's use_com=False) is required here:
with use_com=False, calvados.build.geometry_from_pdb_rna centers the backbone
bead directly on the "P" (phosphate) atom, which does not exist on the 5'-
terminal residue of each strand (free 5'-OH, standard RNA chemistry) --
confirmed this crashes on our real structure. use_com=True instead takes the
center of mass of whichever backbone/base atoms are present per residue,
which handles the missing terminal phosphate correctly.

Placement: proteins and RNA start UNBOUND (topol='random', matching the
CALVADOS two_IDR example), not pre-docked, so any association during the
simulation is spontaneous -- per the handoff's explicit requirement. This is
also why 'center'/'shift_ref_bead' aren't options here: sim.py raises for
those topologies whenever there is more than one component.

Usage:
    python prepare_calvados_with_rna.py \
        --protein-sysname rank_415_chai1_seed4_sample3.0 \
        --protein-input-dir results/drb2_drb4/calvados3_md/rank_415_chai1_seed4_sample3.0/input \
        --rna-sysname rank_01_boltz_seed5_sample0.0 \
        --rna-input-dir results/rna_ds_drb2_drb4/calvados3_md/complex/input \
        --workdir results/rna_ds_drb2_drb4/calvados3_md/complex \
        --tag smoketest --platform OpenCL --threads 1 --steps 10000 --wfreq 1000
"""

import argparse
import os

import numpy as np
import pandas as pd
from calvados.cfg import Config, Components


def flory_ree_nm(n_residues, prefactor=0.55, nu=0.588):
    return prefactor * (n_residues ** nu)


def longest_disordered_run(aiupred_csv):
    df = pd.read_csv(aiupred_csv)
    longest = cur = 0
    for d in df["disordered"]:
        cur = cur + 1 if d else 0
        longest = max(longest, cur)
    return longest


def pdb_extent_nm(pdb_path):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    coords = np.array(coords) / 10.0
    return coords.max(0) - coords.min(0)


def compute_box_length(protein_pdb, aiupred_csv, rna_pdb, min_L=25.0, max_L=50.0, padding=3.0):
    protein_extent = pdb_extent_nm(protein_pdb)
    rna_extent = pdb_extent_nm(rna_pdb)
    n_longest_idr = longest_disordered_run(aiupred_csv)
    ree = flory_ree_nm(n_longest_idr)

    L_protein = float(protein_extent.max()) + 2 * ree + padding
    L_rna = float(rna_extent.max()) + padding
    L = max(L_protein, L_rna, min_L)
    L = min(L, max_L)
    print(f"box sizing: protein_extent_max={protein_extent.max():.1f} nm, "
          f"rna_extent_max={rna_extent.max():.1f} nm, longest IDR={n_longest_idr} aa -> "
          f"Ree~{ree:.1f} nm, L_protein={L_protein:.1f}, L_rna={L_rna:.1f}, chosen L={L:.1f} nm")
    return L


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protein-sysname", required=True)
    p.add_argument("--protein-input-dir", required=True)
    p.add_argument("--rna-sysname", required=True)
    p.add_argument("--rna-input-dir", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--platform", required=True, choices=["CPU", "OpenCL", "CUDA", "Reference"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--wfreq", type=int, required=True)
    p.add_argument("--temp", type=float, default=293.0)
    p.add_argument("--ionic", type=float, default=0.15,
                    help="0.15 M to match the protein-only runs -- NOTE: CALVADOS's own "
                         "single_dsRNA example uses 0.20 M; kept at 0.15 M deliberately "
                         "here for a fair comparison, flagged per the plan")
    p.add_argument("--pH", type=float, default=7.0)
    p.add_argument("--box-nm", type=float, default=None,
                    help="override the auto-computed box length (nm); needed to match a "
                         "pre-built --predocked-pdb's own box")
    p.add_argument("--predocked-pdb", default=None,
                    help="if set, start from this pre-built full-system PDB (see "
                         "build_predocked_start.py) instead of topol-based random placement "
                         "-- sets restart='pdb' and copies this file in as frestart")
    return p.parse_args()


def main():
    args = parse_args()
    workdir = os.path.abspath(args.workdir)
    run_dir = f"{workdir}/{args.tag}"
    os.makedirs(run_dir, exist_ok=True)

    protein_pdb = f"{args.protein_input_dir}/{args.protein_sysname}.pdb"
    aiupred_csv = f"{args.protein_input_dir}/{args.protein_sysname}_aiupred.csv"
    protein_fdomains = f"{args.protein_input_dir}/domains.yaml"
    protein_fresidues = f"{args.protein_input_dir}/residues_CALVADOS3.csv"

    rna_pdb = f"{args.rna_input_dir}/{args.rna_sysname}.pdb"
    rna_fdomains = f"{args.rna_input_dir}/domains.yaml"
    rna_fresidues = f"{args.rna_input_dir}/residues_C2RNA.csv"

    L = args.box_nm if args.box_nm is not None else compute_box_length(protein_pdb, aiupred_csv, rna_pdb)

    if args.predocked_pdb:
        import shutil
        frestart = "predocked_start.pdb"
        shutil.copy(args.predocked_pdb, f"{run_dir}/{frestart}")
        restart_mode = "pdb"
    else:
        frestart = "restart.chk"
        restart_mode = "checkpoint"

    config = Config(
        sysname="drb2_drb4_with_rna",
        box=[L, L, L],
        temp=args.temp,
        ionic=args.ionic,
        pH=args.pH,
        topol="random",
        wfreq=args.wfreq,
        steps=args.steps,
        runtime=0,
        platform=args.platform,
        threads=args.threads,
        restart=restart_mode,
        frestart=frestart,
        verbose=True,
    )
    config.write(run_dir, name="config.yaml")

    components = Components(molecule_type="protein")

    components.add(
        name=args.protein_sysname,
        molecule_type="protein",
        nmol=1,
        restraint=True,
        charge_termini="both",
        fresidues=protein_fresidues,
        pdb_folder=args.protein_input_dir,
        restraint_type="harmonic",
        use_com=True,
        k_harmonic=700.0,
        fdomains=protein_fdomains,
    )

    components.add(
        name=args.rna_sysname,
        molecule_type="rna",
        nmol=1,
        restraint=True,
        fresidues=rna_fresidues,
        pdb_folder=args.rna_input_dir,
        restraint_type="harmonic",
        use_com=True,
        k_harmonic=10.0,
        cutoff_restr=1.5,
        ext_restraint=True,
        fdomains=rna_fdomains,
        rna_kb1=1400.0,
        rna_kb2=2200.0,
        rna_ka=4.20,
        rna_pa=3.14,
        rna_nb_sigma=0.4,
        rna_nb_scale=136,
        rna_nb_cutoff=2.0,
    )

    components.write(run_dir, name="components.yaml")

    print(f"wrote {run_dir}/config.yaml, components.yaml, run.py")


if __name__ == "__main__":
    main()
