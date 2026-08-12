"""
Generate PyMOL visualization sessions for top-ranked switch candidates.

Loads holo and apo Boltz predictions side-by-side, colors by pLDDT,
and highlights the binder interface.

Usage:
    pymol -cq src/visualize_top_hits.py -- [--n 5] [--outputs outputs/] [--run smoke2]
    # or interactively:
    pymol src/visualize_top_hits.py

Runs live in subfolders under outputs/ (e.g. outputs/smoke2/final_all_ranked.csv).
--run picks one explicitly; omit it to use the most recently modified run.
"""
import os
import sys
import argparse

import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="Number of top candidates to visualize")
    p.add_argument("--outputs", default=None, help="Path to outputs/ directory")
    p.add_argument("--run", default=None, help="Run subfolder under outputs/ (e.g. smoke2). Defaults to the most recently modified run.")
    p.add_argument("--metric", default="switch_score", help="Column to rank by")
    if "--" in sys.argv:
        args = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    else:
        args = p.parse_args([])
    return args


def find_outputs_dir():
    for candidate in [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"),
        os.path.join(os.getcwd(), "outputs"),
    ]:
        if os.path.isdir(candidate):
            return candidate
    return None


def find_run_dir(outputs_root, run=None):
    """Resolve a run subfolder (e.g. outputs/smoke2) containing final_all_ranked.csv."""
    if run:
        run_dir = os.path.join(outputs_root, run)
        if not os.path.isfile(os.path.join(run_dir, "final_all_ranked.csv")):
            print(f"ERROR: no final_all_ranked.csv in {run_dir}")
        return run_dir

    candidates = []
    for name in os.listdir(outputs_root):
        run_dir = os.path.join(outputs_root, name)
        ranked_csv = os.path.join(run_dir, "final_all_ranked.csv")
        if os.path.isdir(run_dir) and os.path.isfile(ranked_csv):
            candidates.append((os.path.getmtime(ranked_csv), run_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def main():
    args = parse_args()
    outputs_root = args.outputs or find_outputs_dir()
    if not outputs_root or not os.path.isdir(outputs_root):
        print("ERROR: could not locate outputs/ directory. Pass --outputs explicitly.")
        return

    outputs = find_run_dir(outputs_root, args.run)
    if not outputs:
        print(f"ERROR: no run subfolders with final_all_ranked.csv found under {outputs_root}")
        return
    print(f"Using run: {outputs}")

    ranked_csv = os.path.join(outputs, "final_all_ranked.csv")
    if not os.path.isfile(ranked_csv):
        print(f"ERROR: {ranked_csv} not found. Run the pipeline first.")
        return

    df = pd.read_csv(ranked_csv)
    if args.metric not in df.columns:
        print(f"ERROR: column '{args.metric}' not in {ranked_csv}")
        print(f"Available: {list(df.columns)}")
        return

    top = df.nlargest(args.n, args.metric)

    print(f"\n{'='*60}")
    print(f"Top {len(top)} candidates by {args.metric}")
    print(f"{'='*60}\n")

    plddt_colors = """
set_color plddt_vlow,  [1.000, 0.502, 0.000]
set_color plddt_low,   [1.000, 0.843, 0.000]
set_color plddt_mid,   [0.510, 0.839, 1.000]
set_color plddt_high,  [0.000, 0.337, 0.804]
"""

    plddt_color_cmd = """
color plddt_vlow, {obj} and b < 50
color plddt_low,  {obj} and b > 50 and b < 70
color plddt_mid,  {obj} and b > 70 and b < 90
color plddt_high, {obj} and b > 90
"""

    # Boltz states to load, in display order. s6c/s6d are the four-state
    # off-diagonal controls (binder + wrong target/ligand combo) and are
    # only present in runs with evaluation.four_state_control enabled.
    states = [
        ("holo", "s6a_boltz_holo_location", "s6a_boltz_holo_iptm", "holo ON"),
        ("apo", "s6b_boltz_apo_location", "s6b_boltz_apo_iptm", "apo ON"),
        ("ctrl_holo_nolig", "s6c_boltz_holo_nolig_location", "s6c_boltz_holo_nolig_iptm", "holo OFF ctrl"),
        ("ctrl_apo_lig", "s6d_boltz_apo_lig_location", "s6d_boltz_apo_lig_iptm", "apo OFF ctrl"),
    ]
    states = [s for s in states if s[1] in df.columns]

    script_lines = [
        "# Auto-generated PyMOL visualization script",
        "# Top switch candidates colored by pLDDT",
        "",
        "bg_color white",
        "set ray_shadow, 0",
        "set cartoon_fancy_helices, 1",
        "set cartoon_side_chain_helper, 1",
        plddt_colors,
    ]

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        desc = row["poses_description"]
        score = row.get(args.metric, float("nan"))

        summary = "  ".join(
            f"{label}_ipTM={row.get(iptm_col, float('nan')):.3f}"
            for _, _, iptm_col, label in states
        )
        print(f"  #{rank}: {desc}")
        print(f"       switch_score={score:.3f}  {summary}")
        if "delta_iptm" in row and pd.notna(row.get("delta_iptm")):
            print(f"       delta_iptm={row['delta_iptm']:.3f}  selectivity_score={row.get('selectivity_score', float('nan')):.3f}")

        for state_name, loc_col, _iptm_col, label in states:
            pdb_path = row.get(loc_col, "")
            obj_name = f"rank{rank}_{state_name}"

            if pdb_path and os.path.isfile(str(pdb_path)):
                script_lines.append(f'load {pdb_path}, {obj_name}')
                script_lines.append(f'show cartoon, {obj_name}')
                script_lines.append(f'hide lines, {obj_name}')
                script_lines.append(plddt_color_cmd.format(obj=obj_name))
            else:
                print(f"       WARNING: {label} structure not found: {pdb_path}")

        script_lines.append("")

    script_lines += [
        "# Group by rank",
    ]
    for rank in range(1, len(top) + 1):
        members = " ".join(f"rank{rank}_{state_name}" for state_name, _, _, _ in states)
        script_lines.append(f"group rank{rank}, {members}")

    script_lines += [
        "",
        "# Show only rank 1 initially",
    ]
    for rank in range(2, len(top) + 1):
        script_lines.append(f"disable rank{rank}")

    script_lines += [
        "",
        "orient",
        "zoom",
        "",
        f'# Save session: save {outputs}/top_hits.pse',
    ]

    pml_path = os.path.join(outputs, "visualize_top_hits.pml")
    with open(pml_path, "w") as f:
        f.write("\n".join(script_lines))
    print(f"\nPyMOL script written to: {pml_path}")
    print(f"Run with:  pymol {pml_path}")

    try:
        from pymol import cmd
        print("\nPyMOL detected — executing script...")
        for line in script_lines:
            line = line.strip()
            if line and not line.startswith("#"):
                cmd.do(line)
        print("Done. Use 'enable rank2' etc. to toggle candidates.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
