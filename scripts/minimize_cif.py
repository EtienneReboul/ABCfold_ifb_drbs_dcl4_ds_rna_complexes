"""
ChimeraX Script: CIF Minimization & PDB Export
===============================================
Usage (called by Snakemake, but also works standalone):

    chimerax --nogui --script "minimize_cif.py /path/to/input.cif /path/to/output.pdb"

What it does:
    1. Opens the CIF file
    2. Runs energy minimization (skipped if the output PDB already exists)
    3. Optionally logs the energy trajectory to <output_stem>_energy.csv
    4. Saves the minimized structure as PDB
"""

import sys
import csv
import re
import importlib.util
from pathlib import Path

# ChimeraX imports (available at runtime, but not recognized by static analysis)
from chimerax.core.commands import run # pyright: ignore[reportMissingImports]
from chimerax.core.logger import StringPlainTextLog # pyright: ignore[reportMissingImports]

# Load sanitize() and rename_rna_for_openmm() from sibling sanitize_cif.py
# without triggering its main() (guarded by __name__ check inside that file).
def _load_sanitize_mod():
    _scripts_dir = Path(sys.argv[0]).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "sanitize_cif", _scripts_dir / "sanitize_cif.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod

_san = _load_sanitize_mod()
sanitize                    = _san.sanitize
rename_rna_for_openmm       = _san.rename_rna_for_openmm
rename_rna_terminals_for_plip = _san.rename_rna_terminals_for_plip


# ---------------------------------------------------------------------------
# Energy log helpers
# ---------------------------------------------------------------------------

def parse_energy_log(log_text: str) -> list[dict]:
    """
    Parse energy entries from the minimize logEnergy output.

    ChimeraX logs lines like:
        'Step 100, energy: -12345.67 kJ/mol'

    Returns a list of dicts: [{step: int, energy_kJ_mol: float}, ...]
    """
    entries = []

    # Primary pattern: "Step N, energy: -XXXXX.XX"
    for step, energy in re.findall(
        r"[Ss]tep\s+(\d+)[,:\s]+energy[:\s=]+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        log_text,
    ):
        entries.append({"step": int(step), "energy_kJ_mol": float(energy)})

    if entries:
        return entries

    # Fallback: any pair of (integer, float) on the same line
    for step, energy in re.findall(
        r"(\d+)\D+([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)",
        log_text,
    ):
        entries.append({"step": int(step), "energy_kJ_mol": float(energy)})

    return entries


def save_energy_csv(entries: list[dict], path: Path, session) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "energy_kJ_mol"])
        writer.writeheader()
        writer.writerows(entries)
    session.logger.info(
        f"  Energy trajectory saved ({len(entries)} steps): {path.name}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(session):
    if len(sys.argv) < 3:
        session.logger.error(
            "Usage: chimerax --nogui --script minimize_cif.py <input.cif> <output.pdb>"
        )
        raise SystemExit(1)

    cif_path = Path(sys.argv[1]).expanduser().resolve()
    pdb_path = Path(sys.argv[2]).expanduser().resolve()

    if not cif_path.exists():
        session.logger.error(f"Input CIF not found: {cif_path}")
        raise SystemExit(1)

    # If the output PDB already exists, skip minimization entirely
    if pdb_path.exists():
        session.logger.info(
            f"Output PDB already exists — skipping minimization: {pdb_path.name}"
        )
        run(session, "quit")
        return

    pdb_path.parent.mkdir(parents=True, exist_ok=True)

    session.logger.info(f"Opening: {cif_path.name}")
    run(session, f"open {str(cif_path)!r}")

    session.logger.info("Sanitizing structure before minimization...")
    results = sanitize(session)
    n_warn = len(results["nonstandard_warnings"])
    n_inc  = len(results["incomplete_backbone"])
    if n_warn or n_inc or results["5prime_chains"]:
        session.logger.info(
            f"Sanitation: {results['5prime_chains']} 5'-phosphate(s) stripped, "
            f"{n_inc} incomplete-backbone residue(s) removed, "
            f"{n_warn} non-standard residue warning(s)"
        )

    # Run addh and addcharge manually so we can rename residues in between.
    # Then call minimize with dockPrep false (skips dock_prep since we
    # already handled it), which avoids the KeyError:'P' crash caused by
    # ChimeraX's AMBER residue naming (RU5/RU3) not matching OpenMM's RNA
    # template names (U5/U3).
    session.logger.info("Adding hydrogens...")
    run(session, "addh #1")

    session.logger.info("Adding AMBER charges...")
    run(session, "addcharge #1")

    session.logger.info("Renaming RNA/DNA residues for OpenMM template matching...")
    rename_rna_for_openmm(session)

    # A handful of RosettaFold3-derived structures have been observed to
    # diverge (energy -> nan) and then never terminate -- ChimeraX's minimize
    # has no built-in nan/convergence bailout, so without a cap those loop
    # forever (one ran 4M+ steps over 7+ hours before being killed manually).
    # Rather than one long capped run, screen cheaply first: a short PROBE_STEPS
    # run is enough for a genuinely diverging structure to already show nan
    # (observed cases were nan within the first few hundred steps), so a
    # structure that fails the probe is rejected after ~seconds, not after
    # burning through a large step budget. A structure that probes clean
    # continues for as long as it needs (up to FULL_MAX_STEPS, generous
    # headroom past the ~5-6k steps healthy runs on this system size actually
    # converge in) -- normal runs are not truncated by the probe, only
    # diverging ones are cut short.
    PROBE_STEPS = 2000
    FULL_MAX_STEPS = 50000

    def run_minimize(max_steps):
        with StringPlainTextLog(session.logger) as log:
            run(session, f"minimize #1 dockPrep false liveUpdates false logEnergy true maxSteps {max_steps}")
            return log.getvalue()

    def diverged(log_text):
        return re.search(r"\bnan\b", log_text, re.IGNORECASE) is not None

    session.logger.info(f"Running minimization probe (maxSteps {PROBE_STEPS}) to screen for divergence...")
    probe_log = run_minimize(PROBE_STEPS)
    session.logger.info(f"[minimize probe log]:\n{probe_log.strip()}")

    if diverged(probe_log):
        raise RuntimeError(
            f"Minimization diverged (energy -> nan) within the first {PROBE_STEPS} probe "
            "steps -- refusing to continue or save a garbage structure. Observed so far "
            "only on a handful of RosettaFold3-derived inputs; not fixable by more steps."
        )

    session.logger.info(f"Probe energy looks sane -- continuing to convergence (cap {FULL_MAX_STEPS} more steps)...")
    continue_log = run_minimize(FULL_MAX_STEPS)
    session.logger.info(f"[minimize continuation log]:\n{continue_log.strip()}")

    # Defensive: re-check after the long continuation too, in case divergence
    # only sets in later despite a clean probe.
    if diverged(continue_log):
        raise RuntimeError(
            "Minimization diverged (energy -> nan) during the continuation run, after "
            "passing the initial probe -- refusing to save a garbage structure."
        )

    minimize_log = probe_log + continue_log

    # Save energy trajectory alongside the PDB (optional, does not affect pipeline)
    energy_entries = parse_energy_log(minimize_log)
    if energy_entries:
        energy_csv = pdb_path.with_name(pdb_path.stem + "_energy.csv")
        save_energy_csv(energy_entries, energy_csv, session)
    else:
        session.logger.warning(
            "Could not parse energy values from minimize log.\n"
            f"Raw log:\n{minimize_log.strip()}"
        )

    session.logger.info("Stripping terminal suffixes from RNA/DNA residues for PLIP...")
    rename_rna_terminals_for_plip(session)

    session.logger.info(f"Saving minimized PDB: {pdb_path.name}")
    run(session, f"save {str(pdb_path)!r} models #1 format pdb")

    run(session, "close #1")
    run(session, "quit")


# ChimeraX injects `session` at module scope at runtime
main(session)  # pyright: ignore[reportUndefinedVariable] # noqa: F821