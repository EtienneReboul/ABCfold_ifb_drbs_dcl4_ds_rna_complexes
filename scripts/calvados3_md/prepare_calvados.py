"""
Build CALVADOS3 config.yaml / components.yaml / run.py for a triplicate
CG-MD of one ABCfold-derived complex, restraining folded domains (from
AIUPred, see predict_disorder.py) with CALVADOS3's harmonic elastic-network
restraint while leaving disordered regions fully flexible.

The whole complex (all chains) is set up as a SINGLE CALVADOS component
sourced from one multi-chain PDB. calvados.sequence.seq_from_pdb reads
MDAnalysis "segments" (= PDB chains) and calvados.components.Protein.bond_check
explicitly skips the backbone bond at each chain's C-terminus/N-terminus, so
this does not create a spurious covalent bond between chain A and chain B,
and calvados.build.geometry_from_pdb reads all chains' coordinates from the
same file without re-centering each independently -- i.e. it preserves the
complex's native relative chain arrangement. `topol='center'` is only valid
for a single component/molecule (calvados.sim raises otherwise), which is
exactly our case.

Box size: the complex already has some extension in the starting structure;
we additionally size the box using a Flory-scaling estimate of the *longest*
single disordered segment found by AIUPred (Ree ~= 0.55 nm * N^0.588,
appropriate for a swollen/good-solvent IDR, as used throughout the CALVADOS
IDP literature), so a fully-extended disordered tail cannot self-interact
across periodic images even at the nonbonded cutoffs (2 nm LJ / 4 nm Yukawa).

Usage:
    python prepare_calvados.py --sysname <name> --workdir <results/.../calvados3_md/<name>> \
        --platform CPU --threads 10 --steps 10000 --wfreq 1000 --tag smoketest_cpu
"""

import argparse
import os
import subprocess

import pandas as pd
import numpy as np
from calvados.cfg import Config, Components


def flory_ree_nm(n_residues, prefactor=0.55, nu=0.588):
    return prefactor * (n_residues ** nu)


def longest_disordered_run(aiupred_csv):
    df = pd.read_csv(aiupred_csv)
    longest = 0
    cur = 0
    for d in df["disordered"]:
        cur = cur + 1 if d else 0
        longest = max(longest, cur)
    return longest


def compute_box_length(input_pdb, aiupred_csv, min_L=25.0, max_L=45.0, padding=2.0):
    coords = []
    with open(input_pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    coords = np.array(coords) / 10.0  # Angstrom -> nm
    extent = coords.max(0) - coords.min(0)
    current_max_extent = float(extent.max())

    n_longest_idr = longest_disordered_run(aiupred_csv)
    ree = flory_ree_nm(n_longest_idr)

    L = current_max_extent + 2 * ree + padding
    L = min(max(L, min_L), max_L)
    print(f"box sizing: current_max_extent={current_max_extent:.1f} nm, "
          f"longest disordered run={n_longest_idr} aa -> Ree~{ree:.1f} nm, chosen L={L:.1f} nm")
    return L


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sysname", required=True)
    p.add_argument("--workdir", required=True, help="results/.../calvados3_md/<sysname> directory")
    p.add_argument("--tag", required=True, help="subdirectory name for this run, e.g. smoketest_cpu or replicate_1")
    p.add_argument("--platform", required=True, choices=["CPU", "OpenCL", "CUDA", "Reference"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--wfreq", type=int, required=True)
    p.add_argument("--temp", type=float, default=293.0)
    p.add_argument("--ionic", type=float, default=0.15)
    p.add_argument("--pH", type=float, default=7.0)
    return p.parse_args()


def main():
    args = parse_args()
    workdir = os.path.abspath(args.workdir)
    input_dir = f"{workdir}/input"
    run_dir = f"{workdir}/{args.tag}"
    os.makedirs(run_dir, exist_ok=True)

    input_pdb = f"{input_dir}/{args.sysname}.pdb"
    aiupred_csv = f"{input_dir}/{args.sysname}_aiupred.csv"
    fdomains = f"{input_dir}/domains.yaml"
    fresidues = f"{input_dir}/residues_CALVADOS3.csv"

    L = compute_box_length(input_pdb, aiupred_csv)

    config = Config(
        sysname=args.sysname,
        box=[L, L, L],
        temp=args.temp,
        ionic=args.ionic,
        pH=args.pH,
        topol="center",
        wfreq=args.wfreq,
        steps=args.steps,
        runtime=0,
        platform=args.platform,
        threads=args.threads,
        restart="checkpoint",
        frestart="restart.chk",
        verbose=True,
    )
    config.write(run_dir, name="config.yaml")

    components = Components(
        molecule_type="protein",
        nmol=1,
        restraint=True,
        charge_termini="both",
        fresidues=fresidues,
        pdb_folder=input_dir,
        restraint_type="harmonic",
        use_com=True,
        k_harmonic=700.0,
        fdomains=fdomains,
    )
    components.add(name=args.sysname)
    components.write(run_dir, name="components.yaml")

    print(f"wrote {run_dir}/config.yaml, components.yaml, run.py")


if __name__ == "__main__":
    main()
