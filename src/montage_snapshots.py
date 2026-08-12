"""Assemble the per-binder PyMOL snapshots into one labelled montage
(rows = top binders, columns = holo / apo). Runs in the protflow env.

    python src/montage_snapshots.py outputs/<run>
"""
import os
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main(run_dir, snap_subdir="structure_snapshots", tag="DynamicMPNN"):
    snap = os.path.join(run_dir, snap_subdir)
    man = os.path.join(snap, "snapshots_manifest.csv")
    if not os.path.isfile(man):
        print("no manifest at", man); return
    rows = list(csv.DictReader(open(man)))
    ranks = sorted({int(r["rank"]) for r in rows})
    by = {(int(r["rank"]), r["state"]): r for r in rows}
    states = ["holo", "apo"]
    state_target = {"holo": "PD-L1 (state A)", "apo": "PCNA (state B)"}

    fig, axes = plt.subplots(len(ranks), 2, figsize=(8.4, 3.6 * len(ranks)))
    if len(ranks) == 1:
        axes = axes.reshape(1, 2)
    for i, rank in enumerate(ranks):
        for j, st in enumerate(states):
            ax = axes[i][j]; ax.axis("off")
            rec = by.get((rank, st))
            if rec and os.path.isfile(rec["png"]):
                ax.imshow(mpimg.imread(rec["png"]))
                iptm = float(rec["boltz_iptm"]); pl = float(rec["boltz_plddt_mean"])
                ax.set_title(f"{state_target[st]}    Boltz ipTM={iptm:.2f}  pLDDT={pl:.2f}",
                             fontsize=9)
            if j == 0:
                sp = float(rec["af2_switch_plddt"]) if rec else float("nan")
                ax.text(-0.04, 0.5, f"#{rank}\nswitch-pLDDT\n{sp:.2f}", transform=ax.transAxes,
                        ha="right", va="center", fontsize=9, fontweight="bold")
    fig.suptitle(f"Top {tag} switch binders — each de novo sequence in both bound conformations\n"
                 "designed BINDER coloured by pLDDT (blue = confident, red = low);  "
                 "TARGET (PD-L1 / PCNA) in grey",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0.05, 0, 1, 0.97])
    out = os.path.join(snap, "top_binders_montage.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "."
    sub = sys.argv[2] if len(sys.argv) > 2 else "structure_snapshots"
    tag = sys.argv[3] if len(sys.argv) > 3 else "DynamicMPNN"
    main(run, sub, tag)
