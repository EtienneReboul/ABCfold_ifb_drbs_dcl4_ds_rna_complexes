#!/usr/bin/env python3
"""
scripts/inject_synthetic_template.py
======================================
Preprocessing stage for the ``*_synthtmpl_*`` re-runs: build a
``fold_input.resolved.json`` for the full DCL4/DRB2/DRB4/dsRNA complex that
carries a Chai-1/Boltz DRB2+DRB4 conformation as an AlphaFold3-dialect custom
template on the DRB2 and DRB4 chains.

The synthtmpl complexes have byte-for-byte the same sequences as
``rna_ds_dcl4_drb2_drb4``, so their MSA / template resolution is identical --
this script does NOT re-hit the ColabFold webserver. It starts from a frozen
snapshot of that complex's resolved ``fold_input.resolved.json`` (the *MSA
donor*, copied by ``scripts/select_synthetic_templates.py``), then:

  1. retitles it (``name`` + ``modelSeeds`` from the synthtmpl's own
     ``fold_input.json``);
  2. drops the ColabFold hit-templates from the requested chains
     (``--strip-target-templates``, default DRB2+DRB4) -- their MSAs are
     kept, and DCL4's templates are left untouched;
  3. inserts one ``{mmcif, queryIndices, templateIndices}`` entry at index 0
     of each target chain's ``templates`` list. Both AlphaFold3 and ABCfold's
     OpenFold3 wrapper (``abcfold/openfold3/af3_to_openfold3.py``) consume
     that same field, so one injection feeds both backends.

We do the injection here rather than via ABCfold's own ``add_custom_template``
because that path aligns the query against a crude first-letter-of-resname
pseudo-sequence with ``Bio.Align.PairwiseAligner`` defaults, which are
BioPython-version-sensitive and produced a shredded, mostly-gap mapping under
the pinned env (biopython 1.84). Here the template chain IS the exact same
DRB2 / DRB4 construct as the query (verified reside-by-residue below), so the
map is a provable identity and no alignment is needed. The mmCIF itself is
still built the ABCfold way (single chain via BioPython ``MMCIFIO`` + a
synthetic ``revision_date``), which is the shape AlphaFold3 is tested against.

Usage (called by workflows/preprocessing/Snakefile rule `resolve_synthtmpl`;
runs under envs/preprocessing.yaml -- needs biopython + gemmi):
    python scripts/inject_synthetic_template.py \\
        --donor-resolved-json  data/synthetic_templates/rna_ds_dcl4_drb2_drb4_synthtmpl/donor_resolved.json \\
        --target-fold-json     data/fold_inputs/rna_ds_dcl4_drb2_drb4_synthtmpl_01/fold_input.json \\
        --template-pdb         data/synthetic_templates/rna_ds_dcl4_drb2_drb4_synthtmpl/tmpl_01.pdb \\
        --target-ids           B C \\
        --template-chains      A B \\
        --strip-target-templates B C \\
        --output-json          data/fold_inputs/rna_ds_dcl4_drb2_drb4_synthtmpl_01/fold_input.resolved.json
"""

import argparse
import json
import sys
import time
from io import StringIO
from pathlib import Path

import gemmi
from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--donor-resolved-json", required=True,
                   help="frozen fold_input.resolved.json of the MSA donor complex "
                        "(same sequences as the target)")
    p.add_argument("--target-fold-json", required=True,
                   help="the synthtmpl complex's own fold_input.json "
                        "(source of `name` + `modelSeeds`)")
    p.add_argument("--template-pdb", required=True)
    p.add_argument("--target-ids", nargs="+", required=True,
                   help="chain ids IN THE TARGET COMPLEX that get the custom template")
    p.add_argument("--template-chains", nargs="+", required=True,
                   help="chain ids IN THE TEMPLATE PDB, parallel to --target-ids")
    p.add_argument("--strip-target-templates", nargs="*", default=[],
                   help="chain ids whose existing (ColabFold) `templates` list is "
                        "cleared before injection (MSAs kept)")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _ids(seq_protein) -> list:
    v = seq_protein.get("id")
    return v if isinstance(v, list) else [v]


def chain_one_letter(pdb_path: Path, chain_id: str) -> str:
    """Ordered 1-letter sequence of the polymer residues in `chain_id`."""
    st = gemmi.read_structure(str(pdb_path))
    if len(st) == 0 or chain_id not in {c.name for c in st[0]}:
        raise SystemExit(f"{pdb_path}: no chain {chain_id!r}")
    chain = st[0][chain_id]
    return gemmi.one_letter_code([r.name for r in chain]).upper()


def build_template_mmcif(pdb_path: Path, chain_id: str) -> tuple[str, int]:
    """Single-chain mmCIF string for `chain_id`, residues renumbered 1..N,
    plus a synthetic revision_date (AlphaFold3 wants a release date on every
    template). Mirrors abcfold.scripts.abc_script_utils.get_mmcif's shape."""
    structure = PDBParser(QUIET=True).get_structure("custom", str(pdb_path))
    model = next(iter(structure))

    for ch_id in [c.id for c in model if c.id != chain_id]:
        model.detach_child(ch_id)
    chain = model[chain_id]
    for res in [r for r in chain if r.id[0] != " "]:      # drop any HETATM
        chain.detach_child(res.id)
    for new_num, res in enumerate(list(chain), start=1):  # 1..N, gapless
        res.id = (" ", new_num, " ")
    n_res = len(list(chain))

    io = MMCIFIO()
    io.set_structure(structure)
    buf = StringIO()
    io.save(buf)

    # re-parse -> inject revision_date -> re-serialise (as get_mmcif does)
    reparsed = MMCIFParser(QUIET=True)
    reparsed.get_structure("custom", StringIO(buf.getvalue()))
    mmcif_dict = reparsed._mmcif_dict
    mmcif_dict.setdefault("_pdbx_audit_revision_history.revision_date",
                          time.strftime("%Y-%m-%d"))
    out = StringIO()
    io.set_dict(mmcif_dict)
    io.save(out)
    return out.getvalue(), n_res


def main():
    args = parse_args()
    if len(args.target_ids) != len(args.template_chains):
        raise SystemExit("--target-ids and --template-chains must have equal length")

    template_pdb = Path(args.template_pdb).resolve()
    if not template_pdb.exists():
        raise SystemExit(f"template PDB not found: {template_pdb}")

    donor = json.loads(Path(args.donor_resolved_json).read_text())
    target_meta = json.loads(Path(args.target_fold_json).read_text())
    donor["name"] = target_meta["name"]
    donor["modelSeeds"] = target_meta["modelSeeds"]

    prot_by_id = {}
    for seq in donor.get("sequences", []):
        prot = seq.get("protein")
        if prot:
            for cid in _ids(prot):
                prot_by_id[cid] = prot

    # 1. strip ColabFold templates from the mapped chains (keep MSAs)
    stripped = []
    for cid in args.strip_target_templates:
        prot = prot_by_id.get(cid)
        if prot is not None and prot.pop("templates", None) is not None:
            stripped.append(cid)

    # 2. inject one identity-mapped custom template per (target, template chain)
    injected = []
    for tid, tchain in zip(args.target_ids, args.template_chains):
        prot = prot_by_id.get(tid)
        if prot is None:
            raise SystemExit(f"target chain {tid!r} is not a protein in the donor JSON")
        query_seq = prot["sequence"]
        tmpl_seq = chain_one_letter(template_pdb, tchain)
        if tmpl_seq != query_seq:
            raise SystemExit(
                f"chain {tid}: template PDB chain {tchain} sequence "
                f"({len(tmpl_seq)} aa) does not match the query ({len(query_seq)} aa) "
                f"exactly -- identity mapping is only valid for the same construct.\n"
                f"  first mismatch at {next((i for i,(a,b) in enumerate(zip(tmpl_seq, query_seq)) if a != b), min(len(tmpl_seq), len(query_seq)))}"
            )
        mmcif_str, n_res = build_template_mmcif(template_pdb, tchain)
        if n_res != len(query_seq):
            raise SystemExit(f"chain {tid}: mmCIF kept {n_res} residues, expected {len(query_seq)}")
        idx = list(range(n_res))
        prot.setdefault("templates", [])
        prot["templates"].insert(0, {
            "mmcif": mmcif_str, "queryIndices": idx, "templateIndices": idx,
        })
        injected.append(f"{tid}<-{tchain}({n_res})")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(donor, indent=2) + "\n")

    # 3. verify
    problems = []
    for tid in args.target_ids:
        n = len(prot_by_id[tid].get("templates", []))
        if n != 1:
            problems.append(f"chain {tid}: expected 1 template, found {n}")
    for seq in donor["sequences"]:
        rna = seq.get("rna")
        if rna is not None and "unpairedMsa" not in rna:
            problems.append("an RNA chain lost its `unpairedMsa` key (AF3 RNA-MSA crash guard)")
    if problems:
        for pmsg in problems:
            print(f"ERROR: {pmsg}", file=sys.stderr)
        sys.exit(1)

    print(f"[inject] {donor['name']}: stripped ColabFold templates {stripped}, "
          f"injected {', '.join(injected)} from {template_pdb.name} -> {out}")


if __name__ == "__main__":
    main()
