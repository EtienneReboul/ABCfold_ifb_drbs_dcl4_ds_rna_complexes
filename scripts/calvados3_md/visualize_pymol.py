"""
PyMOL visualization of a CALVADOS3 CG-MD replicate.

CALVADOS writes the whole complex as ONE chain ("A") with continuous global
1-based residue numbering across all original chains (top.pdb: chain A
residues 1-434, chain B residues 435-789 for this system -- see
prepare_calvados.py). This script:
  - loads the CG topology + DCD trajectory
  - draws the single-bead-per-residue chain as explicit bonds (sticks) between
    consecutive residues + small spheres for the beads. NOTE: an earlier
    version used `cartoon_trace_atoms` + tube, but PyMOL's cartoon spline
    fitter mishandles a tightly-coiled ~200-bead CG trace and collapses it
    into a misleading simplified "fan of straight lines" -- verified against
    a plain-spheres render and the raw coordinates (bond lengths, contour
    length vs end-to-end distance) that this was a rendering artifact, not a
    real conformation. Explicit sticks draw exactly the bonded path per
    frame with no spline smoothing, so this is the faithful representation.
  - colors each folded domain from domains.yaml a distinct solid color, and
    every disordered residue (everything else) light grey, so you can see
    at a glance which regions are restrained vs freely moving
  - optionally shades the two original chains differently (--chain-boundary)
  - either leaves an interactive PyMOL session open (scrub frames yourself
    with the frame slider / `mplay`) or, headless, ray-traces a PNG per frame
    for you to assemble into a movie (this PyMOL build has no ffmpeg, so no
    direct .mp4 export)

Usage (interactive, opens the PyMOL GUI):
    pymol scripts/calvados3_md/visualize_pymol.py -- \\
        results/drb2_drb4/calvados3_md/<name>/replicate_1 \\
        results/drb2_drb4/calvados3_md/<name>/input/domains.yaml \\
        --chain-boundary 435

Usage (headless PNG frames for a movie):
    pymol -cq scripts/calvados3_md/visualize_pymol.py -- \\
        results/drb2_drb4/calvados3_md/<name>/replicate_1 \\
        results/drb2_drb4/calvados3_md/<name>/input/domains.yaml \\
        --chain-boundary 435 --png-dir frames/replicate_1 --stride 5
"""

import argparse
import glob
import re
import sys

from pymol import cmd

DOMAIN_COLORS = ["marine", "orange", "teal", "purpleblue", "yellow", "firebrick"]
DISORDER_COLOR = "grey80"


def load_domains_yaml(path):
    """ Minimal parser for calvados's domains.yaml (no PyYAML in this PyMOL's
    bundled python): '<name>:' header followed by '- [start, end]' lines. """
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    name = lines[0].rstrip(":").strip()
    domains = []
    for line in lines[1:]:
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", line)
        if m:
            domains.append([int(m.group(1)), int(m.group(2))])
    return name, domains


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("rundir", help="replicate directory with top.pdb + *.dcd")
    p.add_argument("domains_yaml", help="domains.yaml (global 1-based residue ranges)")
    p.add_argument("--chain-boundary", type=int, default=None,
                    help="global resi where chain B starts (e.g. 435), just for a visual chain split cue")
    p.add_argument("--stride", type=int, default=1, help="load every Nth frame")
    p.add_argument("--smooth-window", type=int, default=3,
                    help="cmd.smooth window (CG beads are jittery at 1 frame/100k steps); 0 disables")
    p.add_argument("--png-dir", default=None,
                    help="if set, ray-trace every loaded frame to this directory instead of staying interactive")
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=900)
    return p.parse_args(argv)


def main():
    # pymol passes script args after "--" through sys.argv
    argv = sys.argv[1:]
    args = parse_args(argv)

    sysname, domains = load_domains_yaml(args.domains_yaml)

    top = f"{args.rundir}/top.pdb"
    dcds = sorted(glob.glob(f"{args.rundir}/*.dcd"))
    if not dcds:
        raise SystemExit(f"no .dcd found in {args.rundir}")
    unwrapped = [d for d in dcds if "_unwrapped" in d]
    dcd = unwrapped[0] if unwrapped else dcds[0]
    if not unwrapped:
        print("WARNING: no *_unwrapped.dcd found -- rendering raw (periodic-wrapped) "
              "coordinates; run unwrap_traj.py first to avoid box-edge teleport artifacts")
    else:
        print(f"using unwrapped trajectory: {dcd}")

    obj = sysname.replace(".", "_")
    cmd.load(top, obj)

    # top.pdb writes the WHOLE complex under a single PDB chain ID with
    # continuous residue numbering (see prepare_calvados.py docstring), so
    # cartoon_trace_atoms would otherwise draw an illegitimate tube straight
    # from the last residue of chain A to the first residue of chain B --
    # those two are different, unbonded molecules. Re-splitting into real
    # PyMOL chains at the known boundary makes the cartoon break there too.
    n_res = cmd.count_atoms(obj)
    boundary = args.chain_boundary if args.chain_boundary is not None else n_res + 1
    if args.chain_boundary is not None:
        cmd.alter(f"{obj} and resi {args.chain_boundary}-999999", "chain='B'")
        cmd.rebuild(obj)

    # explicit consecutive-residue bonds (1-based PDB atom index == global
    # residue number here, one bead per residue), skipping the chain break
    for i in range(1, n_res):
        if i + 1 == boundary:
            continue
        cmd.bond(f"{obj} and index {i}", f"{obj} and index {i + 1}")

    cmd.load_traj(dcd, obj, state=1, interval=args.stride)
    n_states = cmd.count_states(obj)
    print(f"loaded {obj}: {n_states} states from {dcd}")

    # representation: CG bead chain as explicit sticks (bonded path, no
    # spline smoothing) + small spheres for the beads themselves
    cmd.hide("everything", obj)
    cmd.show("sticks", obj)
    cmd.set("stick_radius", 0.25, obj)
    cmd.show("spheres", obj)
    cmd.set("sphere_scale", 0.35, obj)

    # base color = disordered
    cmd.color(DISORDER_COLOR, obj)

    # color each folded domain distinctly
    for i, (s, e) in enumerate(domains):
        color = DOMAIN_COLORS[i % len(DOMAIN_COLORS)]
        sel = f"{obj} and resi {s}-{e}"
        cmd.color(color, sel)
        print(f"domain {s}-{e}: colored {color}")

    # optional visual cue for the original chain split (dim chain B slightly)
    if args.chain_boundary is not None:
        cmd.set("stick_transparency", 0.15, f"{obj} and resi {args.chain_boundary}-9999")
        cmd.set("sphere_transparency", 0.15, f"{obj} and resi {args.chain_boundary}-9999")

    if args.smooth_window and n_states > args.smooth_window:
        cmd.smooth(obj, window=args.smooth_window)

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.orient(obj)
    cmd.zoom(obj, buffer=10)

    if args.png_dir:
        import os
        os.makedirs(args.png_dir, exist_ok=True)
        for state in range(1, n_states + 1):
            cmd.frame(state)
            out = f"{args.png_dir}/frame_{state:05d}.png"
            cmd.ray(args.width, args.height)
            cmd.png(out, dpi=150)
            if state % 20 == 0 or state == n_states:
                print(f"rendered {state}/{n_states} -> {out}")
        print(f"done. Assemble with e.g.:\n"
              f"  ffmpeg -framerate 20 -i {args.png_dir}/frame_%05d.png "
              f"-pix_fmt yuv420p {args.png_dir}/movie.mp4")
    else:
        session = f"{args.rundir}/{obj}_pymol.pse"
        cmd.mset(f"1-{n_states}")
        cmd.save(session)
        print(f"saved interactive session -> {session}")
        print("scrub frames with the frame slider, or run: mplay")


main()
