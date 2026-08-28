"""
ChimeraX Script: CIF/PDB Structure Sanitation
==============================================
Prepares AlphaFold3 (and similar predictor) CIF structures for
ChimeraX AMBER energy minimisation.

Standalone usage:
    chimerax --nogui --script "sanitize_cif.py <input.cif> <output.cif>"

Can also be loaded as a module by other ChimeraX scripts (main() is guarded):
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("sanitize_cif", <path_to_this_file>)
    _mod  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    _mod.sanitize(session)

Sanitation steps (applied in order)
------------------------------------
1. Strip 5'-phosphate from RNA/DNA terminal residues
   AlphaFold3 places a 5'-phosphate (P, OP1, OP2, OP3) on the first nucleotide,
   giving it the non-standard AMBER residue type RU5PP / RA5PP etc.
   ChimeraX's addh then protonates OP3, and add_charge crashes:
       KeyError ('RU5PP', 'op3')
       ChargeError: Hydrogen HP3 bonded to atom that should not have
                    hydrogens (OP3)
   Removing the phosphate converts the terminus to AMBER's standard 5'-OH
   form (RU5, RA5 ...) which has proper charge parameters.

2. Delete solvent molecules (HOH, WAT, crystallographic solvents)

3. Delete non-current alternate conformations

4. Delete protein residues with an incomplete backbone (missing N, CA, C or O)
   Such residues crash AMBER's bond-topology builder during minimisation.

5. Warn about non-standard residue types that may lack AMBER parameters
   These are logged as warnings; nothing is deleted automatically so the
   user can decide how to handle them.
"""

import sys
from pathlib import Path

from chimerax.core.commands import run  # pyright: ignore[reportMissingImports]
from chimerax.atomic import Residue, Structure  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROTEIN_BACKBONE = frozenset({"N", "CA", "C", "O"})

_5PRIME_PHOSPHATE_ATOMS = frozenset({"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"})

_STANDARD_AMBER_AA = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX",
    "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
})

_STANDARD_AMBER_RNA = frozenset({
    "A",   "C",   "G",   "U",             # raw names as written in AF3 CIF
    "RA",  "RC",  "RG",  "RU",            # AMBER internal
    "RA5", "RC5", "RG5", "RU5",           # AMBER 5'-OH terminal (after sanitisation)
    "RA3", "RC3", "RG3", "RU3",           # AMBER 3'-terminal
})

_STANDARD_AMBER_DNA = frozenset({
    "DA",  "DC",  "DG",  "DT",
    "DA5", "DC5", "DG5", "DT5",
    "DA3", "DC3", "DG3", "DT3",
})

_STANDARD_RESIDUES = _STANDARD_AMBER_AA | _STANDARD_AMBER_RNA | _STANDARD_AMBER_DNA


# ---------------------------------------------------------------------------
# Step 1 — Strip 5'-terminal phosphate from nucleic-acid chains
# ---------------------------------------------------------------------------

def strip_5prime_phosphate(session) -> int:
    """
    Remove P, OP1, OP2, OP3 atoms from the first residue of every RNA/DNA chain.
    Returns the number of chain termini modified.
    """
    count = 0
    for m in session.models:
        if not isinstance(m, Structure):
            continue
        for chain in m.chains:
            residues = chain.existing_residues
            if not residues:
                continue
            first = residues[0]
            if first.polymer_type != Residue.PT_NUCLEIC:
                continue
            to_del = [a for a in first.atoms if a.name in _5PRIME_PHOSPHATE_ATOMS]
            if to_del:
                removed = [a.name for a in to_del]
                for atom in to_del:
                    atom.delete()
                count += 1
                session.logger.info(
                    f"  [step 1] chain {chain.chain_id}: stripped "
                    f"{{{', '.join(removed)}}} from {first.name}{first.number}"
                )
    return count


# ---------------------------------------------------------------------------
# Step 2 — Delete solvent
# ---------------------------------------------------------------------------

def delete_solvent(session) -> None:
    """Delete water and other solvent molecules using ChimeraX's built-in classification."""
    run(session, "delete solvent")


# ---------------------------------------------------------------------------
# Step 3 — Delete non-current alternate conformations
# ---------------------------------------------------------------------------

def delete_alt_locs(session) -> None:
    """Keep only the primary alternate conformation for each atom."""
    for m in session.models:
        if isinstance(m, Structure):
            m.delete_alt_locs()


# ---------------------------------------------------------------------------
# Step 4 — Delete protein residues with an incomplete backbone
# ---------------------------------------------------------------------------

def delete_incomplete_backbone(session) -> list:
    """
    Delete any protein residue missing one or more of N, CA, C, O.
    Such residues crash AMBER's bond-topology builder.
    Returns a list of human-readable descriptions of deleted residues.
    """
    deleted = []
    for m in session.models:
        if not isinstance(m, Structure):
            continue
        for res in list(m.residues):
            if res.polymer_type != Residue.PT_AMINO:
                continue
            missing = _PROTEIN_BACKBONE - {a.name for a in res.atoms}
            if missing:
                desc = (
                    f"/{res.chain_id} {res.name}{res.number} "
                    f"(missing: {', '.join(sorted(missing))})"
                )
                session.logger.warning(
                    f"  [step 4] deleting incomplete residue {desc}"
                )
                deleted.append(desc)
                res.delete()
    return deleted


# ---------------------------------------------------------------------------
# Step 5 — Warn about non-standard residues
# ---------------------------------------------------------------------------

def warn_nonstandard(session) -> list:
    """
    Log a warning for every polymer residue type not in the standard AMBER
    parameter set.  Residues with PT_NONE (ligands, cofactors) are skipped
    since dock_prep handles those separately.
    Returns a list of warning strings.
    """
    warnings = []
    seen: set = set()
    for m in session.models:
        if not isinstance(m, Structure):
            continue
        for res in m.residues:
            name = res.name
            if name in seen or name in _STANDARD_RESIDUES:
                continue
            if res.polymer_type == Residue.PT_NONE:
                continue
            seen.add(name)
            msg = (
                f"  [step 5] non-standard residue '{name}' "
                f"(chain {res.chain_id}, res {res.number}) — "
                f"may lack AMBER parameters"
            )
            session.logger.warning(msg)
            warnings.append(msg)
    return warnings


# ---------------------------------------------------------------------------
# Step 6 — Rename RNA/DNA terminal residues for OpenMM template matching
#           (called AFTER addcharge, not part of the general sanitize flow)
# ---------------------------------------------------------------------------

def rename_rna_for_openmm(session) -> int:
    """
    Rename RNA/DNA residue .name to match OpenMM's RNA.OL3.xml template names.

    After addcharge runs, ChimeraX stores the AMBER residue classification in
    residue.amber_name (e.g. RU5, RU, RU3).  The residue .name stays as the
    original CIF name (e.g. U for all positions).  OpenMM's RNA templates use
    unprefixed names (U5, U, U3), so terminal residues that don't get the R-
    prefix stripped fall through to a GAFF auto-template.  The auto-template
    assigns atom type 'P' to RNA phosphorus, which is not in GAFF's classNameForType
    and causes:

        KeyError: 'P'
        in openmm/app/forcefield.py setAtomClasses

    Fix: for every nucleic-acid residue whose amber_name starts with 'R',
    rename .name to amber_name[1:] so OpenMM finds the correct template.

        amber_name  →  .name (OpenMM template)
        RU5         →  U5    (5'-OH terminal)
        RU          →  U     (internal, already correct)
        RU3         →  U3    (3'-terminal)
        RA5         →  A5
        ...

    Must be called AFTER addcharge (which sets amber_name) and BEFORE
    minimize dockPrep false (which builds the OpenMM topology).
    """
    count = 0
    for m in session.models:
        if not isinstance(m, Structure):
            continue
        for res in m.residues:
            if res.polymer_type != Residue.PT_NUCLEIC:
                continue
            amber = getattr(res, "amber_name", None)
            if not amber or not amber.startswith("R"):
                continue
            new_name = amber[1:]  # "RU5" → "U5", "RA3" → "A3", etc.
            if res.name != new_name:
                session.logger.info(
                    f"  [openmm rename] /{res.chain_id} {res.name}{res.number}: "
                    f"'{res.name}' (amber: {amber}) → '{new_name}'"
                )
                res.name = new_name
                count += 1
    if count:
        session.logger.info(
            f"Renamed {count} RNA/DNA residue(s) to match OpenMM templates"
        )
    return count


# ---------------------------------------------------------------------------
# Step 7 — Strip terminal suffixes from RNA/DNA residue names for PLIP
#           (called AFTER minimize, not part of the general sanitize flow)
# ---------------------------------------------------------------------------

# Map from AMBER/OpenMM terminal names → canonical one-letter RNA names.
# PLIP only recognises A, G, C, U (and DA, DG, DC, DT for DNA).  The
# terminal variants produced by rename_rna_for_openmm (U5, U3, A5, A3 …)
# are not in PLIP's nucleotide set, so PLIP classifies them as SMALLMOLECULE
# and folds the remaining (recognised) residues into the receptor when
# --dnareceptor is used.  Stripping the suffix here gives all 21 positions
# standard names before the PDB is written for PLIP.
_RNA_TERMINAL_TO_BASE = {
    "A5": "A", "A3": "A",
    "G5": "G", "G3": "G",
    "C5": "C", "C3": "C",
    "U5": "U", "U3": "U",
}

_DNA_TERMINAL_TO_BASE = {
    "DA5": "DA", "DA3": "DA",
    "DG5": "DG", "DG3": "DG",
    "DC5": "DC", "DC3": "DC",
    "DT5": "DT", "DT3": "DT",
}

_ALL_TERMINAL_MAP = {**_RNA_TERMINAL_TO_BASE, **_DNA_TERMINAL_TO_BASE}


def rename_rna_terminals_for_plip(session) -> int:
    """
    Strip AMBER/OpenMM terminal-position suffixes from nucleic-acid residue
    names so that PLIP recognises every nucleotide in the chain.

    rename_rna_for_openmm() converts AMBER names RU5/RU3 → U5/U3 to satisfy
    OpenMM's RNA template matching.  After minimisation those names are no
    longer needed and must be reversed before the PDB is handed to PLIP:
    PLIP's RNA set is {A, G, C, U} — it does not contain U5, A3, etc.  When
    --dnareceptor is active, unrecognised residues fall out of the receptor
    and their interactions are not reported, causing false negatives for every
    nucleotide that happens to carry a terminal label.

    Mapping applied:
        U5 / U3  →  U      A5 / A3  →  A
        G5 / G3  →  G      C5 / C3  →  C
        DA5/DA3  →  DA     DG5/DG3  →  DG
        DC5/DC3  →  DC     DT5/DT3  →  DT

    Must be called AFTER minimize (the terminal names are required by OpenMM)
    and BEFORE saving the PDB that will be passed to PLIP.
    """
    count = 0
    for m in session.models:
        if not isinstance(m, Structure):
            continue
        for res in m.residues:
            if res.polymer_type != Residue.PT_NUCLEIC:
                continue
            new_name = _ALL_TERMINAL_MAP.get(res.name)
            if new_name is None:
                continue
            session.logger.info(
                f"  [plip rename] /{res.chain_id} {res.name}{res.number}: "
                f"'{res.name}' → '{new_name}'"
            )
            res.name = new_name
            count += 1
    if count:
        session.logger.info(
            f"Stripped terminal suffixes from {count} RNA/DNA residue(s) for PLIP"
        )
    return count


# ---------------------------------------------------------------------------
# Combined entry point (used by minimize_cif.py via importlib)
# ---------------------------------------------------------------------------

def sanitize(session) -> dict:
    """
    Run all five sanitation steps in order and return a summary dict:
        5prime_chains        — int, number of nucleic-acid termini modified
        incomplete_backbone  — list[str], deleted residue descriptions
        nonstandard_warnings — list[str], warning messages logged
    """
    results: dict = {}
    results["5prime_chains"]        = strip_5prime_phosphate(session)
    delete_solvent(session)
    delete_alt_locs(session)
    results["incomplete_backbone"]  = delete_incomplete_backbone(session)
    results["nonstandard_warnings"] = warn_nonstandard(session)
    return results


# ---------------------------------------------------------------------------
# Standalone script entry point
# Guard: when loaded as a module via importlib, __name__ == 'sanitize_cif'
#        so main() is skipped.  When run by ChimeraX as a script, __name__
#        is the file path and main() executes normally.
# ---------------------------------------------------------------------------

def main(session):
    if len(sys.argv) < 3:
        session.logger.error(
            'Usage: chimerax --nogui --script '
            '"sanitize_cif.py <input.cif|pdb> <output.cif|pdb>"'
        )
        raise SystemExit(1)

    in_path  = Path(sys.argv[1]).expanduser().resolve()
    out_path = Path(sys.argv[2]).expanduser().resolve()

    if not in_path.exists():
        session.logger.error(f"Input not found: {in_path}")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    session.logger.info(f"Opening: {in_path.name}")
    run(session, f"open {str(in_path)!r}")

    session.logger.info("Running sanitation steps...")
    results = sanitize(session)

    n_warn = len(results["nonstandard_warnings"])
    n_inc  = len(results["incomplete_backbone"])
    session.logger.info(
        f"Sanitation complete — "
        f"5'-phosphate stripped: {results['5prime_chains']} chain(s) | "
        f"incomplete backbone removed: {n_inc} residue(s) | "
        f"non-standard warnings: {n_warn}"
    )

    fmt = out_path.suffix.lstrip(".").lower()
    if fmt not in ("cif", "pdb"):
        fmt = "cif"

    session.logger.info(f"Saving sanitized structure: {out_path.name}")
    run(session, f"save {str(out_path)!r} models #1 format {fmt}")

    run(session, "close #1")
    run(session, "quit")


if __name__ != "sanitize_cif":
    main(session)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
