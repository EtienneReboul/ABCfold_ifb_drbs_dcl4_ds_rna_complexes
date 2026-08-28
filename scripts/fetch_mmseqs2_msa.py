#!/usr/bin/env python3
"""
scripts/fetch_mmseqs2_msa.py
=============================
Stage 1b of the ABCfold IFB DRB2/DCL4/dsRNA pipeline — the "default run"
MSA/template resolution: MSA + top-hit templates from the ColabFold
MMseqs2 webserver, no manual curation, no pocket restraint. Thin wrapper
around ABCfold's own `mmseqs2msa` CLI utility (ships with the abcfold PyPI
package) — the same tool ../../../NPF-ab-initio-modelling/ABCfold_NPF_pipeline's
scripts/fetch_mmseqs2_msa.py wraps.

Simplified from that version: this repo has no sibling pipeline with
already-fetched DRB2/DCL4/dsRNA MSAs to reuse locally, so there is no
local-reuse branch — every complex goes straight to the webserver call.
`mmseqs2msa` itself is multimer-aware (it walks every `protein` entry in
fold_input.json's `sequences` list and fills in each chain's own
unpairedMsa/pairedMsa/templates — confirmed against the ABCFold source,
abcfold.scripts.add_mmseqs_msa), so no chain-by-chain looping is needed
here for the 2- and 5-chain complexes this pipeline runs. RNA chains are
passed through untouched (no MSA to search for).

Run ONCE per complex (there is no apo/holo split here, unlike
ABCfold_NPF_pipeline — every complex in this pipeline is holo by
definition, the RNA/DRB2/DRB4/DCL4 sequences ARE the complex).

Usage (called by Snakemake rule `fetch_mmseqs2_msa`):
    python scripts/fetch_mmseqs2_msa.py \\
        --input-json   data/fold_inputs/drb2_drb4/fold_input.json \\
        --output-json  data/fold_inputs/drb2_drb4/fold_input.resolved.json \\
        --num-templates 20 \\
        --retries 3 \\
        --delay 8
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json",    required=True)
    p.add_argument("--output-json",   required=True)
    p.add_argument("--num-templates", type=int, default=20)
    p.add_argument("--retries",       type=int, default=3)
    p.add_argument("--delay",         type=float, default=8,
                   help="Seconds to sleep after a successful webserver call "
                        "(politeness towards the shared ColabFold webserver)")
    return p.parse_args()


def patch_rna_empty_msa(output_json: Path) -> None:
    """AF3's own data pipeline only skips its internal RNA MSA search when
    an RNA chain's `unpairedMsa` key is present at all (even `""`) —
    `alphafold3/data/pipeline.py:process_rna_chain` branches on `chain.
    unpaired_msa is not None`, not on whether it's non-empty. Left absent
    (mmseqs2msa never touches RNA chains — no MSA to search for), AF3
    instead calls its own `_get_rna_msa()` against local nt_rna/rfam/
    rnacentral databases via `nhmmer`, which segfaulted on IFB (confirmed
    2026-08-22, job 1396814_0's af3_error.log) — so setting the key to
    `""` here isn't just cosmetic, it's what makes AF3 skip that crashing
    path entirely."""
    doc = json.loads(output_json.read_text())
    patched = 0
    for seq in doc.get("sequences", []):
        rna = seq.get("rna")
        if rna is not None and "unpairedMsa" not in rna:
            rna["unpairedMsa"] = ""
            patched += 1
    if patched:
        output_json.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"[mmseqs2_msa] patched {patched} RNA chain(s) with empty unpairedMsa "
              f"(skips AF3's own crash-prone internal RNA MSA search)")


def fetch_from_webserver(input_json: Path, output_json: Path, num_templates: int,
                          retries: int, delay: float) -> None:
    mmseqs2msa = shutil.which("mmseqs2msa")
    if mmseqs2msa is None:
        raise RuntimeError(
            "mmseqs2msa not found on PATH — install the `abcfold` package "
            "(see envs/preprocessing.yaml) to get this CLI entry point."
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        mmseqs2msa,
        "--input_json", str(input_json),
        "--output_json", str(output_json),
        "--templates",
        "--num_templates", str(num_templates),
    ]

    last_err = None
    for attempt in range(1, retries + 1):
        print(f"[mmseqs2_msa] {input_json.parent.name}: attempt {attempt}/{retries} "
              f"— {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True)
            if not output_json.exists():
                raise RuntimeError(f"mmseqs2msa exited 0 but {output_json} was not written")
            patch_rna_empty_msa(output_json)
            print(f"[mmseqs2_msa] done → {output_json}")
            time.sleep(delay)
            return
        except (subprocess.CalledProcessError, RuntimeError) as e:
            last_err = e
            if attempt < retries:
                wait = 30 * attempt
                print(f"[mmseqs2_msa] attempt {attempt} failed ({e}); "
                      f"waiting {wait}s before retry ...", flush=True)
                time.sleep(wait)

    raise RuntimeError(
        f"mmseqs2msa failed after {retries} attempts for {input_json}: {last_err}"
    ) from last_err


def main():
    args = parse_args()
    fetch_from_webserver(
        Path(args.input_json), Path(args.output_json),
        args.num_templates, args.retries, args.delay,
    )


if __name__ == "__main__":
    sys.exit(main())
