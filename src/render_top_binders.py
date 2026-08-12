"""Render the top-N switch binders in both conformations, coloured by pLDDT.

Runs under PyMOL (uses the Boltz-predicted complexes s6a_boltz_holo_location /
s6b_boltz_apo_location from final_all_ranked.csv). Saves one PNG per binder per
state into <run>/structure_snapshots/ and prints the metrics per hit.

    ~/.conda/envs/mdplot/bin/pymol -cq src/render_top_binders.py -- \
        --run outputs/<run> --n 5 --metric af2_switch_plddt

Uses only the stdlib csv module (no pandas) so it works in the PyMOL env.
"""
import os
import sys
import csv
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Path to the run output dir")
    p.add_argument("--arm", choices=["dynamicmpnn", "msd"], default="dynamicmpnn")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--metric", default="af2_switch_plddt")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=850)
    # PyMOL usually strips the "--" separator, leaving the flags in sys.argv[1:];
    # fall back to that, and ignore any extras PyMOL may inject.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args, _ = p.parse_known_args(argv)
    return args


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    args = parse_args()
    ARM = {
        "dynamicmpnn": dict(ranked="final_all_ranked.csv", out="structure_snapshots", tag="DynamicMPNN",
            states=[("holo", "s6a_boltz_holo_location", "PD-L1", "s6a_boltz_holo_iptm", "s6a_boltz_holo_plddt_mean"),
                    ("apo", "s6b_boltz_apo_location", "PCNA", "s6b_boltz_apo_iptm", "s6b_boltz_apo_plddt_mean")]),
        "msd": dict(ranked="msd_final_all_ranked.csv", out="structure_snapshots_msd", tag="ProteinMPNN-MSD",
            states=[("holo", "s7a_msd_boltz_holo_location", "PD-L1", "s7a_msd_boltz_holo_iptm", "s7a_msd_boltz_holo_plddt_mean"),
                    ("apo", "s7b_msd_boltz_apo_location", "PCNA", "s7b_msd_boltz_apo_iptm", "s7b_msd_boltz_apo_plddt_mean")]),
    }[args.arm]
    ranked = os.path.join(args.run, ARM["ranked"])
    if not os.path.isfile(ranked):
        print(f"ERROR: no {ARM['ranked']} in {args.run}"); return
    with open(ranked) as fh:
        rows = list(csv.DictReader(fh))
    if not rows or args.metric not in rows[0]:
        print("ERROR: metric", args.metric, "not in", ranked); return
    rows = [r for r in rows if r.get(args.metric) not in (None, "", "nan")]
    rows.sort(key=lambda r: fnum(r[args.metric]), reverse=True)
    top = rows[:args.n]

    out_dir = os.path.join(args.run, ARM["out"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"arm: {ARM['tag']}")

    from pymol import cmd
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("ray_shadows", 0)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("antialias", 2)

    states = ARM["states"]

    manifest = []
    for rank, r in enumerate(top, 1):
        desc = r.get("poses_description", f"rank{rank}")
        print("#%d  %s  %s=%.3f" % (rank, desc, args.metric, fnum(r[args.metric])))
        for name, loc_col, target, iptm_col, plddt_col in states:
            path = r.get(loc_col, "")
            if not path or not os.path.isfile(str(path)):
                print("     WARNING: missing %s structure: %s" % (name, path)); continue
            cmd.delete("all")
            cmd.load(path, "cplx")
            cmd.hide("everything", "cplx")
            cmd.show("cartoon", "cplx")
            # Distinguish the designed BINDER from the TARGET it binds. The target
            # is the longer chain (PD-L1 ~110, PCNA ~260 res vs binder ~80). Target
            # -> flat grey (context); binder -> coloured by pLDDT (red=low..blue=high),
            # so the designed protein is the coloured foreground.
            chains = cmd.get_chains("cplx")
            if len(chains) >= 2:
                lens = {c: cmd.count_atoms("cplx and chain %s and name CA" % c) for c in chains}
                target_chain = max(lens, key=lens.get)
                target_sel = "cplx and chain %s" % target_chain
                binder_sel = "cplx and not chain %s" % target_chain
                cmd.color("grey75", target_sel)
                cmd.set("cartoon_transparency", 0.30, target_sel)
                cmd.spectrum("b", "red_orange_yellow_green_blue", binder_sel, minimum=0, maximum=100)
            else:
                cmd.spectrum("b", "red_orange_yellow_green_blue", "cplx", minimum=0, maximum=100)
            cmd.orient("cplx")
            cmd.zoom("cplx", 3)
            png = os.path.join(out_dir, "rank%d_%s.png" % (rank, name))
            cmd.ray(args.width, args.height)
            cmd.png(png, dpi=150)
            print("     wrote %s   ipTM=%.3f  pLDDT=%.1f  (%s)" %
                  (os.path.basename(png), fnum(r.get(iptm_col)), fnum(r.get(plddt_col)), target))
            manifest.append((rank, name, desc, png, fnum(r[args.metric]),
                             fnum(r.get(iptm_col)), fnum(r.get(plddt_col))))

    with open(os.path.join(out_dir, "snapshots_manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "state", "poses_description", "png", "af2_switch_plddt", "boltz_iptm", "boltz_plddt_mean"])
        w.writerows(manifest)
    print("\nSnapshots in:", out_dir)


main()
