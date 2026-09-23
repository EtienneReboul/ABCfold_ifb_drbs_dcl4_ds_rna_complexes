"""
Run AIUPred on each chain of the complex and turn the per-residue disorder
propensity into folded-domain ranges for CALVADOS3's harmonic (elastic
network) restraints.

CALVADOS reads folded-domain boundaries from a `fdomains` YAML file, keyed by
component name, as a list of [start, end] residue ranges in 1-based,
inclusive, *global* numbering across all chains of that component
(calvados.build.get_ssdomains). For this complex (chain A = 434 aa, chain B =
355 aa, concatenated as one CALVADOS component -- see extract_chains.py),
global numbering is: chain A residues 1-434, chain B residues 435-789.

Folded-domain calling from the AIUPred disorder track:
  1. threshold at 0.5 (the published AIUPred/IUPred default) -> per-residue
     binary disordered/ordered call
  2. close disordered gaps of length <= GAP_MERGE inside an ordered run
     (AIUPred is a smooth per-residue score; short dips at a domain's
     surface loops should not fragment a real folded domain)
  3. drop ordered islands shorter than MIN_DOMAIN_LEN (noise, not a real
     restrainable domain -- CALVADOS's own multidomain examples use domains
     of tens of residues, e.g. the dsRBD fold here)

Usage:
    python predict_disorder.py --fasta <chains.fasta> --outdir <input_dir> --name <sysname>
"""

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# aiupred (torch) must be imported before pandas/yaml: importing pandas
# first pulls in a numpy/BLAS runtime that segfaults once torch initializes
# its own bundled OpenMP/MKL later (observed on this conda env: exit 139).
from aiupred import AIUPred, multifasta_reader

import yaml
import pandas as pd

THRESHOLD = 0.5
GAP_MERGE = 3
MIN_DOMAIN_LEN = 15


def calls_to_ranges(disordered_calls):
    """ disordered_calls: list[bool], True = disordered. Returns 1-based
    inclusive [start,end] ranges of the *ordered* (folded) stretches. """
    ordered = [not d for d in disordered_calls]

    # merge short disordered gaps inside ordered runs
    n = len(ordered)
    i = 0
    while i < n:
        if not ordered[i]:
            j = i
            while j < n and not ordered[j]:
                j += 1
            gap_len = j - i
            flanked = i > 0 and j < n and ordered[i - 1] and ordered[j]
            if flanked and gap_len <= GAP_MERGE:
                for k in range(i, j):
                    ordered[k] = True
            i = j
        else:
            i += 1

    # extract ordered runs, drop short ones
    ranges = []
    i = 0
    while i < n:
        if ordered[i]:
            j = i
            while j < n and ordered[j]:
                j += 1
            if (j - i) >= MIN_DOMAIN_LEN:
                ranges.append([i + 1, j])  # 1-based inclusive
            i = j
        else:
            i += 1
    return ranges


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True, help="multi-FASTA with one entry per chain, in chain order")
    p.add_argument("--outdir", required=True, help="CALVADOS input/ directory")
    p.add_argument("--name", required=True, help="CALVADOS component/system name")
    return p.parse_args()


def main():
    args = parse_args()
    records = multifasta_reader(args.fasta)

    predictor = AIUPred(force_cpu=True)

    rows = []
    domains_global = []
    offset = 0
    for header, seq in records.items():
        disorder = predictor.predict_disorder(seq)
        disordered_calls = [s > THRESHOLD for s in disorder]
        chain_ranges = calls_to_ranges(disordered_calls)

        print(f"{header}: {len(seq)} aa, {sum(disordered_calls)} disordered "
              f"({100*sum(disordered_calls)/len(seq):.1f}%), "
              f"folded domains (chain-local, 1-based): {chain_ranges}")

        for local_i, (res, score) in enumerate(zip(seq, disorder), start=1):
            rows.append({
                "chain_header": header,
                "resi_local": local_i,
                "resi_global": local_i + offset,
                "resn": res,
                "aiupred_disorder": float(score),
                "disordered": bool(score > THRESHOLD),
            })

        for start, end in chain_ranges:
            domains_global.append([start + offset, end + offset])

        offset += len(seq)

    df = pd.DataFrame(rows)
    csv_path = f"{args.outdir}/{args.name}_aiupred.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    domains_yaml_path = f"{args.outdir}/domains.yaml"
    with open(domains_yaml_path, "w") as f:
        yaml.dump({args.name: domains_global}, f, default_flow_style=None, sort_keys=False)
    print(f"wrote {domains_yaml_path}")
    print(f"global folded domains: {domains_global}")


if __name__ == "__main__":
    main()
