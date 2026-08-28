#!/usr/bin/env python3
"""
scripts/make_multimer_af3_input.py
====================================
Stage 1a of the ABCfold IFB DRB2/DCL4/dsRNA pipeline:
Generate ONE base fold_input.json per complex, covering every replica
(seed) at once. AlphaFold3-dialect JSON — ABCfold accepts exactly this
format as its own input (https://github.com/rigdenlab/ABCFold): the same
"name", "sequences", "modelSeeds", "dialect", "version" JSON drives
AlphaFold3, Boltz-2, Chai-1, OpenFold3, Protenix and RosettaFold3 together.

Generalizes ../../../NPF-ab-initio-modelling/ABCfold_NPF_pipeline/scripts/
make_af3_input.py (single protein + optional ligand) to an arbitrary
multi-chain complex (N protein chains + N RNA chains), read from a
configs/<complex>.yaml chain spec (see configs/drb2_drb4.yaml and
configs/rna_ds_dcl4_drb2_drb4.yaml).

This file has no MSA or templates embedded yet — scripts/fetch_mmseqs2_msa.py
(stage 1b) adds those from the ColabFold MMseqs2 webserver, once per
complex, producing fold_input.resolved.json.

Usage (called by Snakemake rule `prepare_af3_input`):
    python scripts/make_multimer_af3_input.py \\
        --complex-config    configs/rna_ds_dcl4_drb2_drb4.yaml \\
        --output            data/fold_inputs/rna_ds_dcl4_drb2_drb4/fold_input.json \\
        --n-replicas        20 \\
        --seed-strategy      sequential \\
        --seed-base          1 \\
        --dialect            alphafold3 \\
        --json-version       1
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--complex-config",     required=True,
                    help="configs/<complex>.yaml chain spec")
    p.add_argument("--output",             required=True)
    p.add_argument("--n-replicas",         type=int, required=True,
                    help="j — number of independent seeds (modelSeeds entries)")
    p.add_argument("--seed-strategy",      choices=["sequential", "random"],
                    default="sequential")
    p.add_argument("--seed-base",          type=int, default=1,
                    help="sequential: seeds = seed_base .. seed_base+n_replicas-1")
    p.add_argument("--random-master-seed", type=int, default=0,
                    help="random: RNG seed, combined with a hash of the complex name "
                         "so every complex gets its own draw")
    p.add_argument("--dialect",            default="alphafold3")
    p.add_argument("--json-version",       type=int, default=1)
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_complex_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text())
    if "sequences" not in spec or not spec["sequences"]:
        raise ValueError(f"{path}: no 'sequences' entries found")
    for seq in spec["sequences"]:
        if seq.get("type") not in ("protein", "rna"):
            raise ValueError(
                f"{path}: unsupported sequence type {seq.get('type')!r} "
                "(expected 'protein' or 'rna')"
            )
    return spec


def generate_seeds(strategy: str, n_replicas: int, seed_base: int,
                    random_master_seed: int, complex_name: str) -> list[int]:
    """Return `n_replicas` distinct, explicit model seeds."""
    if strategy == "sequential":
        return [seed_base + i for i in range(n_replicas)]

    # "random": deterministic per complex, but not a simple arithmetic
    # sequence — combine the master seed with a stable hash of the complex
    # name so re-running preprocessing reproduces the same seeds.
    digest = hashlib.sha256(complex_name.encode()).hexdigest()
    complex_hash = int(digest[:8], 16)
    rng = random.Random(random_master_seed + complex_hash)
    return rng.sample(range(1, 2**31 - 1), n_replicas)


def build_fold_input(spec: dict, seeds: list[int], dialect: str, json_version: int) -> dict:
    sequences = []
    for seq in spec["sequences"]:
        key = "protein" if seq["type"] == "protein" else "rna"
        sequences.append({
            key: {
                "id": [seq["id"]],
                "sequence": seq["sequence"],
            }
        })
    return {
        "name": spec["name"],
        "sequences": sequences,
        "modelSeeds": seeds,
        "dialect": dialect,
        "version": json_version,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    spec = load_complex_spec(Path(args.complex_config))
    seeds = generate_seeds(
        args.seed_strategy, args.n_replicas, args.seed_base,
        args.random_master_seed, spec["name"],
    )
    doc = build_fold_input(spec, seeds, args.dialect, args.json_version)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(doc, indent=2) + "\n")

    chains = ", ".join(f"{s['id']}={s['name']}({s['type']})" for s in spec["sequences"])
    print(
        f"[af3_input] {spec['name']}: {len(seeds)} seeds "
        f"({args.seed_strategy}, first={seeds[0]}, last={seeds[-1]}), "
        f"chains: {chains} -> {output_path}"
    )


if __name__ == "__main__":
    main()
