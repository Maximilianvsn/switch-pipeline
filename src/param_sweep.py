"""Reduced-scale parameter sweeps for selecting partial_t and related settings.

Several settings admit a range of values: `apo_batch`, `dmpnn_nseq`, the state-2
designability thresholds, `pre_af2_top_k`, `num_recycles` and
`apo_target.partial_t`. This harness sweeps them systematically in place of
manual configuration edits.

Three operations are provided:

  plan     write one derived config per grid point (base config plus overrides)
  submit   launch a reduced-scale run per config, named after its grid point
  collect  assemble the decision metrics of every run into one table and figure

The pipeline itself is unchanged: derived configs are ordinary YAML files and the
runs are ordinary runs, so a sweep cannot affect a production run.

## Two sweep modes

`--mode geometry` (cheap, minutes) runs only Steps 1-2 via the pipeline's existing
`--geometry-calibration` path. This is the right mode for `apo_target.partial_t`,
because partial_t's effect is entirely upstream — it decides how far the binder moves
between states, which is measurable from geometry alone.

`--mode pipeline` (expensive, hours) runs the full pipeline at smoke or pilot scale.
Required for any setting whose effect is visible only after scoring: the designability
thresholds, sequence budgets, recycles.

## Metrics harvested

Chosen because each was shown to be decision-relevant on
`production_proteins_pdl1_pcna_nogate_20260720_v1`:

  geometry_pass_rate          fraction of state-2 candidates clearing the geometry gate
  state2_designable_rate      per-candidate designability rate (baseline 5.0% -- the bottleneck)
  state1_backbones_surviving  backbones admitting >=1 designable state-2 (baseline 107/354)
  af2_relaxed_hits            passing sequences at the AF2 gate (baseline 166)
  backbones_with_hit          independent hits (baseline 29) -- the number that matters
  apo_ipae_enrichment         state-2 interface tail enrichment vs scramble (baseline 2.8x)
  median_binder_ca_rmsd       inter-state conformational change
  median_interface_jaccard    interface sharing; guards the degenerate both-states-alike cheat

Usage:

    # 1. plan + submit a partial_t sweep (cheap)
    python src/param_sweep.py plan   --base configs/pdl1_pcna_protein_only_nogate.yaml \
        --sweep apo_target.partial_t=2,5,8,10,15 --mode geometry --name pt_sweep
    python src/param_sweep.py submit --name pt_sweep

    # 2. once the jobs finish
    python src/param_sweep.py collect --name pt_sweep
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import os
import subprocess
import sys

import pandas as pd
import yaml

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_ROOT = os.path.join(WS, "outputs", "_sweeps")


# Config derivation

def set_by_path(cfg: dict, dotted: str, value) -> None:
    """Set cfg['a']['b']['c'] from 'a.b.c'. Raises if the path does not exist.

    Deliberately strict: a typo'd key would otherwise add a new, ignored setting
    and the sweep would silently compare N identical runs.
    """
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"{dotted}: '{k}' is not a mapping in the base config")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(
            f"{dotted}: '{keys[-1]}' not present in the base config. "
            "Sweeps may only override settings that already exist.")
    node[keys[-1]] = value


def _coerce(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    return text


def parse_sweep(specs: list[str]) -> list[tuple[str, list]]:
    """['a.b=1,2'] -> [('a.b', [1,2])]. Use '|' when a VALUE contains a comma.

    RFD3 contigs ("A19-115,/0,80") are the motivating case: comma is both the
    natural list separator and part of the value. If '|' appears anywhere in the
    value list it becomes the separator instead.
    """
    out = []
    for spec in specs:
        if "=" not in spec:
            sys.exit(f"bad --sweep '{spec}', expected dotted.key=v1,v2,... "
                     "(or dotted.key=v1|v2 when a value contains a comma)")
        key, vals = spec.split("=", 1)
        sep = "|" if "|" in vals else ","
        out.append((key.strip(), [_coerce(v) for v in vals.split(sep) if v != ""]))
    return out


def _grid(axes: list[tuple[str, list]]) -> list[dict]:
    points = [{}]
    for key, values in axes:
        points = [{**p, key: v} for p in points for v in values]
    return points


_LABEL_SAFE = re.compile(r"[^A-Za-z0-9]+")


def _label(point: dict) -> str:
    """Filesystem-safe grid-point label.

    Values may be arbitrary config strings (contigs contain '/' and ','), so
    everything outside [A-Za-z0-9] collapses to '_'. A '/' left in here silently
    became a directory separator and the config write failed with ENOENT.
    """
    parts = []
    for key, value in point.items():
        name = key.split(".")[-1]
        safe = _LABEL_SAFE.sub("_", str(value).replace(".", "p")).strip("_")
        parts.append(f"{name}_{safe}")
    return "__".join(parts)


# Plan

def cmd_plan(a) -> None:
    with open(a.base) as fh:
        base = yaml.safe_load(fh)
    axes = parse_sweep(a.sweep)
    points = _grid(axes)
    root = os.path.join(SWEEP_ROOT, a.name)
    cfg_dir = os.path.join(root, "configs")
    os.makedirs(cfg_dir, exist_ok=True)

    entries = []
    for point in points:
        cfg = copy.deepcopy(base)
        for key, value in point.items():
            set_by_path(cfg, key, value)          # raises on a typo'd key
        label = _label(point)
        cfg_path = os.path.join(cfg_dir, f"{label}.yaml")
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        entries.append({"label": label, "config": cfg_path,
                        "run_name": f"sweep_{a.name}_{label}", **point})

    manifest = {"name": a.name, "base": os.path.abspath(a.base), "mode": a.mode,
                "scale": a.scale, "axes": [k for k, _ in axes], "runs": entries}
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(pd.DataFrame(entries)[["label"] + [k for k, _ in axes]].to_string(index=False))
    print(f"\n{len(entries)} grid points -> {root}/")
    print(f"next: python src/param_sweep.py submit --name {a.name}")


# Submit

SUBMIT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --partition=cpu-single
#SBATCH --mem=8G
#SBATCH --time={walltime}
#SBATCH --output={root}/logs/%x_%j.out
#SBATCH --error={root}/logs/%x_%j.err
set -euo pipefail
export PATH="$(printf '%s' "$PATH" | tr ':' '\\n' | grep -v '/\\.conda/envs/' | paste -sd ':' -)"
source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate protflow
cd "{ws}"
python -u src/switch_pipeline.py --config {config} --run-name {run_name} {flags}
"""


def cmd_submit(a) -> None:
    root = os.path.join(SWEEP_ROOT, a.name)
    with open(os.path.join(root, "manifest.json")) as fh:
        man = json.load(fh)
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    flags = "--geometry-calibration" if man["mode"] == "geometry" else ""
    if man["scale"] == "smoke":
        flags += " --smoke"
    walltime = "04:00:00" if man["mode"] == "geometry" else "24:00:00"

    for entry in man["runs"]:
        script = os.path.join(root, f"submit_{entry['label']}.sh")
        with open(script, "w") as fh:
            fh.write(SUBMIT_TEMPLATE.format(
                job=f"sw_{a.name}_{entry['label']}"[:40], root=root, ws=WS,
                config=entry["config"], run_name=entry["run_name"],
                flags=flags.strip(), walltime=walltime))
        os.chmod(script, 0o755)
        if a.dry_run:
            print(f"  would submit: {script}")
            continue
        r = subprocess.run(["sbatch", script], capture_output=True, text=True)
        print(f"  {entry['label']:32s} {r.stdout.strip() or r.stderr.strip()}")
    if a.dry_run:
        print("\n--dry-run: nothing submitted")


# Collect

def _funnel(run_dir: str) -> dict:
    path = os.path.join(run_dir, "funnel_summary.csv")
    if not os.path.isfile(path):
        return {}
    f = pd.read_csv(path)
    return dict(zip(f["step"], f["n_designs"]))


def _backfill_hits(run_dir: str) -> None:
    """Generate the hit artifacts if the run did not write them itself.

    Gate-only runs made before the 2026-07-29 fix exited without calling
    tier_hits, so `hits_summary_by_arm.csv` is absent and every hit-derived sweep
    metric would be blank. These tools read output CSVs only, so regenerating
    them post-hoc is exact -- not an approximation of what the run would have done.
    """
    if os.path.isfile(os.path.join(run_dir, "hits_summary_by_arm.csv")):
        return
    if not os.path.isfile(os.path.join(run_dir, "s5_5_af2_gate_all.csv")):
        return                                    # run never reached the AF2 gate
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import tier_hits
        tier_hits.write_tier_hits(run_dir)
        print(f"  backfilled hit artifacts for {os.path.basename(run_dir)}")
    except Exception as exc:
        print(f"  backfill failed for {os.path.basename(run_dir)}: "
              f"{type(exc).__name__}: {exc}")


def harvest(run_dir: str) -> dict:
    """Pull every available decision metric from one finished run directory."""
    _backfill_hits(run_dir)
    out: dict = {}
    fn = _funnel(run_dir)
    out["s1_backbones"] = fn.get("s1_5_designability")
    out["s2_candidates"] = fn.get("s2_rfd3_apo")
    out["s2_geometry_pass"] = fn.get("s2_geometry_gate")
    out["state1_backbones_surviving"] = fn.get("s2_5_state2_designability")
    if out.get("s2_candidates") and out.get("s2_geometry_pass"):
        out["geometry_pass_rate"] = out["s2_geometry_pass"] / out["s2_candidates"]

    s2 = os.path.join(run_dir, "s2_5_state2_designability.csv")
    if os.path.isfile(s2):
        d = pd.read_csv(s2)
        if "state2_designability_pass" in d:
            out["state2_designable_rate"] = float(
                d["state2_designability_pass"].fillna(False).mean())
        for col, name in (("binder_ca_rmsd", "median_binder_ca_rmsd"),
                          ("interface_jaccard", "median_interface_jaccard")):
            if col in d:
                out[name] = float(pd.to_numeric(d[col], errors="coerce").median())

    # Written by tier_hits.py; absent on a --geometry-calibration run.
    summary = os.path.join(run_dir, "hits_summary_by_arm.csv")
    if os.path.isfile(summary):
        h = pd.read_csv(summary)
        dyn = h[h["arm"] == "dynamicmpnn"]
        if len(dyn):
            out["af2_relaxed_hits"] = int(dyn["n_relaxed"].iloc[0])
            out["backbones_with_hit"] = int(dyn["n_backbones_relaxed"].iloc[0])
            out["null_relaxed_floor"] = int(dyn["n_relaxed_null"].iloc[0])
    enr = os.path.join(run_dir, "hits_dynamicmpnn_null_tail_enrichment.csv")
    if os.path.isfile(enr):
        e = pd.read_csv(enr).set_index("metric")["enrichment"]
        out["apo_ipae_enrichment"] = float(e.get("af2_apo_i_pae", float("nan")))
        out["holo_ipae_enrichment"] = float(e.get("af2_holo_i_pae", float("nan")))
    return out


def cmd_collect(a) -> None:
    root = os.path.join(SWEEP_ROOT, a.name)
    with open(os.path.join(root, "manifest.json")) as fh:
        man = json.load(fh)
    axes = man["axes"]
    rows = []
    for entry in man["runs"]:
        run_dir = os.path.join(WS, "outputs", entry["run_name"])
        row = {"label": entry["label"], **{k: entry[k] for k in axes},
               "run_exists": os.path.isdir(run_dir)}
        row.update(harvest(run_dir) if row["run_exists"] else {})
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(root, "sweep_results.csv")
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    missing = df[~df["run_exists"]]
    if len(missing):
        print(f"\n{len(missing)} run(s) not found yet: "
              f"{', '.join(missing['label'])}")
    plot_sweep(df, axes, root)
    print(f"\nwrote {out_csv}" + (" + sweep_results.png" if len(axes) == 1 else ""))


def plot_sweep(df: pd.DataFrame, axes: list[str], root: str) -> str | None:
    """One panel per metric against the swept axis. Only for 1-D sweeps."""
    if len(axes) != 1 or df.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    axis = axes[0]
    # A swept value may be a config STRING (an RFD3 contig), which cannot be a
    # numeric x-axis. Derive one from its trailing integer -- for contigs that is
    # the binder length, which is the quantity actually being compared. Fall back
    # to categorical positions when no number can be recovered.
    xnum = pd.to_numeric(df[axis], errors="coerce")
    if xnum.isna().any():
        extracted = df[axis].astype(str).str.extract(r"(\d+)\s*$")[0]
        xnum = pd.to_numeric(extracted, errors="coerce")
    derived = not pd.to_numeric(df[axis], errors="coerce").notna().all()
    if xnum.isna().any():
        df = df.copy()
        df["_x"] = range(len(df))
        xcol, xticks = "_x", list(df[axis].astype(str))
        xlabel = axis.split(".")[-1]
    else:
        df = df.copy()
        df["_x"] = xnum
        xcol, xticks = "_x", None
        # Name what the number MEANS. "contig" on the axis of an 80/100/120/150
        # sweep reads as nonsense; the trailing field of an RFD3 contig is the
        # binder length, which is what the sweep is actually comparing.
        xlabel = ("binder length (residues)" if derived and "contig" in axis
                  else f"{axis.split('.')[-1]} (from {axis.split('.')[-1]})"
                  if derived else axis.split(".")[-1])
    metrics = [("state2_designable_rate", "state-2 designable rate"),
               ("state1_backbones_surviving", "state-1 backbones surviving"),
               ("geometry_pass_rate", "geometry gate pass rate"),
               ("median_binder_ca_rmsd", "median binder CA RMSD (Å)"),
               ("median_interface_jaccard", "median interface jaccard"),
               ("backbones_with_hit", "backbones with an AF2 hit")]
    metrics = [(c, lab) for c, lab in metrics
               if c in df.columns and df[c].notna().any()]
    if not metrics:
        return None
    d = df.sort_values(xcol)
    n = len(metrics)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axarr = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                              squeeze=False)
    INK, ACCENT = "#20242b", "#3d6fb4"
    for ax, (col, label) in zip(axarr.flat, metrics):
        ax.plot(d[xcol], d[col], "-o", color=ACCENT, lw=2, ms=6, zorder=3)
        if xticks:
            ax.set_xticks(list(d[xcol])); ax.set_xticklabels(xticks, rotation=30, ha="right", fontsize=7)
        for x, y in zip(d[xcol], d[col]):
            if pd.notna(y):
                ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7.5, color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c8ccd2")
        ax.set_axisbelow(True)
        ax.grid(color="#e6e8ec", lw=0.8)
    for ax in axarr.flat[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(f"Parameter sweep: {axis}", x=0.01, ha="left", color=INK,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(root, "sweep_results.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write one derived config per grid point")
    p.add_argument("--base", required=True)
    p.add_argument("--sweep", action="append", required=True,
                   help="dotted.key=v1,v2,... (repeatable for a multi-axis grid)")
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=["geometry", "pipeline"], default="geometry",
                   help="geometry = Steps 1-2 only (cheap, right for partial_t)")
    p.add_argument("--scale", choices=["smoke", "full"], default="smoke")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("submit", help="sbatch one run per grid point")
    p.add_argument("--name", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("collect", help="harvest metrics into a table + figure")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_collect)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
