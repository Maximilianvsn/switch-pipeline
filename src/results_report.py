"""Assemble a small, ordered `results/` folder from a finished run.

A production run writes approximately 46 top-level files and 48 directories,
predominantly intermediate compute (3.3 GB of LigandMPNN poses, 1.6 GB of
RFdiffusion3 output) and diagnostics. This module copies the reader-facing subset
into `results/`, numbered in reading order, and prunes empty directories; nothing
is deleted or moved.

The curated set addresses the two questions the run exists to answer: which
binders to take forward and with what supporting evidence, and whether
DynamicMPNN or ProteinMPNN-MSD produces the better designs. All remaining
artefacts stay in place.

    python src/results_report.py outputs/<run>
"""
from __future__ import annotations

import os
import shutil
import sys

import pandas as pd

# Columns required by a wet-lab shortlist, in reading order.
_CANDIDATE_COLS = [
    "rank", "design_id", "arm", "backbone", "sequence", "length",
    "af2_switch_plddt", "af2_holo_plddt", "af2_apo_plddt",
    "af2_holo_i_pae", "af2_apo_i_pae", "af2_holo_i_ptm", "af2_apo_i_ptm",
    "evidence", "backbone_auc_min", "variant_of_backbone", "n_null_wins",
    "binder_ca_rmsd", "interface_jaccard",
    "scored_by_boltz", "af2_holo_pdb", "af2_apo_pdb",
]


def _read(run_dir: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, name)
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def build_candidates(run_dir: str) -> pd.DataFrame:
    """One row per design worth testing, best-first, with its evidence attached.

    Joins the per-design hit list to the per-backbone discrimination table, so a
    reader sees both "this sequence scored well" and "this backbone's designs
    beat their own scrambles" — which are different claims and both matter.
    """
    hits = _read(run_dir, "hits_top.csv")
    if hits is None or hits.empty:
        return pd.DataFrame()
    ev = _read(run_dir, "hits_by_backbone.csv")
    if ev is not None and not ev.empty:
        ev = ev[["arm", "backbone", "auc_min", "validated"]].rename(
            columns={"auc_min": "backbone_auc_min", "validated": "backbone_validated"})
        hits = hits.merge(ev, on=["arm", "backbone"], how="left")
    else:
        hits["backbone_auc_min"], hits["backbone_validated"] = float("nan"), False

    hits = hits.rename(columns={"poses_description": "design_id"})
    hits["length"] = hits["sequence"].astype(str).str.len()

    # Evidence tiers, most to least defensible. `strict_metrics` means the design
    # clears the absolute literature-style bar in both states; `backbone_validated`
    # means its backbone's pool separates from its own scrambles. They are
    # independent, so a design can have one without the other.
    strict = ((pd.to_numeric(hits.get("af2_holo_i_pae"), errors="coerce") < 0.34)
              & (pd.to_numeric(hits.get("af2_apo_i_pae"), errors="coerce") < 0.34)
              & (pd.to_numeric(hits.get("af2_holo_i_ptm"), errors="coerce") > 0.50)
              & (pd.to_numeric(hits.get("af2_apo_i_ptm"), errors="coerce") > 0.50))
    validated = hits.get("backbone_validated", pd.Series(False, index=hits.index)).fillna(False)
    hits["evidence"] = "metrics_only"
    hits.loc[validated, "evidence"] = "backbone_validated"
    hits.loc[strict, "evidence"] = "strict_metrics"
    hits.loc[strict & validated, "evidence"] = "strict_and_validated"

    order = {"strict_and_validated": 0, "strict_metrics": 1,
             "backbone_validated": 2, "metrics_only": 3}
    hits["_o"] = hits["evidence"].map(order).fillna(9)
    hits = hits.sort_values(["_o", "af2_switch_plddt"], ascending=[True, False])
    hits = hits.drop(columns=["_o"]).reset_index(drop=True)
    hits["rank"] = range(1, len(hits) + 1)
    # Sequence variants of one backbone are not independent hits. On the
    # reference production run the entire strict tier was 8 variants of a single
    # backbone, which reads as 8 candidates unless this is made explicit.
    # variant_of_backbone = 1 marks the best design per backbone; order a wet-lab
    # panel by that column first to get diverse scaffolds rather than 8 siblings.
    hits["variant_of_backbone"] = hits.groupby(["arm", "backbone"]).cumcount() + 1
    return hits[[c for c in _CANDIDATE_COLS if c in hits.columns]]


def prune_empty_dirs(run_dir: str) -> list[str]:
    """Remove directories containing no files at any depth.

    `plots/` and `scores/` are created unconditionally by ProtFlow's
    `Poses.set_work_dir` and are never written to by this pipeline, so every run
    ends with at least two empty directories. Pruning is safe: a directory with
    no files anywhere beneath it holds nothing to lose.
    """
    removed = []
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path) or name == "results":
            continue
        if not any(files for _, _, files in os.walk(path)):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name)
    return removed


def _summary_text(run_dir: str, candidates: pd.DataFrame) -> str:
    lines = [f"RUN: {os.path.basename(os.path.abspath(run_dir))}", "=" * 64, ""]
    ev = _read(run_dir, "hits_by_backbone.csv")
    arms = _read(run_dir, "hits_summary_by_arm.csv")

    lines.append("WET-LAB SHORTLIST")
    if candidates.empty:
        lines.append("  no design passed the AF2 two-state gate")
    else:
        counts = candidates["evidence"].value_counts()
        for tier in ("strict_and_validated", "strict_metrics",
                     "backbone_validated", "metrics_only"):
            if tier in counts:
                lines.append(f"  {tier:24s} {counts[tier]:4d}")
        n_bb = candidates["backbone"].nunique()
        n_first = int((candidates["variant_of_backbone"] == 1).sum())
        lines.append(f"  {'TOTAL':24s} {len(candidates):4d}"
                     f"   over {n_bb} backbones ({n_first} distinct scaffolds)")
        top = candidates.head(10)
        if (top["variant_of_backbone"] > 1).any():
            lines.append(f"  NOTE: the top 10 rows span only "
                         f"{top['backbone'].nunique()} backbone(s) -- they are sequence")
            lines.append( "        variants, not independent hits. Filter"
                          " variant_of_backbone==1 for a diverse panel.")
    lines.append("")

    lines.append("METHOD COMPARISON (equal sequence budget per backbone)")
    if arms is not None and not arms.empty:
        for _, r in arms.iterrows():
            lines.append(f"  {r['label']:18s} hits {int(r.get('n_relaxed', 0)):4d}"
                         f"   backbones {int(r.get('n_backbones_relaxed', 0)):3d}"
                         f"   null floor {int(r.get('n_relaxed_null', 0)):3d}")
    if ev is not None and not ev.empty:
        v = ev.groupby("arm")["validated"].sum()
        lines.append("  backbones validated against their own scrambles: "
                     + ", ".join(f"{k} {int(x)}" for k, x in v.items()))
    lines.append("")

    lines.append("EVIDENCE DEFINITIONS")
    lines += [
        "  strict_metrics      i_pae < 0.34 and i_ptm > 0.50 in BOTH states",
        "                      (i_pae 0.34 ~ 10 A, the literature interface-PAE bar)",
        "  backbone_validated  every AF2 metric beats this backbone's OWN scrambles",
        "                      at AUC > 0.70 -- a per-backbone claim, independent of",
        "                      the run-level stop/go verdict",
        "",
        "LIMITS -- these metrics are predictor CONFIDENCE, not affinity.",
        "  No thermodynamic bistability, binding free energy, state population or",
        "  switching kinetics is established by any number in this folder.",
    ]
    return "\n".join(lines) + "\n"


_README = """# results/

Read in order. Everything here is copied from the run directory; nothing is moved.

| file | question it answers |
|---|---|
| `1_candidates.csv` | Which binders should I try, best first, with evidence and structure paths |
| `2_backbone_evidence.csv` | Which backbones beat their OWN scrambles (per-backbone, not run-level) |
| `3_method_comparison.csv` | DynamicMPNN vs ProteinMPNN-MSD at equal budget |
| `4_run_summary.txt` | Funnel counts, evidence tallies, and what may NOT be claimed |
| `figures/` | The plots worth looking at |

## Reading `1_candidates.csv`

Sorted by evidence tier, then score. `evidence` is the column to trust:

- **`strict_and_validated`** — clears the absolute metric bar in both states AND
  its backbone separates from its own scrambles. Strongest available.
- **`strict_metrics`** — clears the absolute bar; backbone not individually validated.
- **`backbone_validated`** — backbone separates from its scrambles; design is
  below the absolute bar.
- **`metrics_only`** — passed the run's relative gate and nothing more.

`af2_holo_pdb` / `af2_apo_pdb` point at the predicted structures for that exact
sequence, if it reached structure capture.

## What these numbers are not

Every metric here is **predictor confidence**. None is an affinity, a free energy,
or a state population. A design at the top of this list is a hypothesis worth
testing, not a binder.
"""


def write_results(run_dir: str) -> str:
    """Build `results/`, prune empty directories, return the results path."""
    out = os.path.join(run_dir, "results")
    figs = os.path.join(out, "figures")
    os.makedirs(figs, exist_ok=True)

    candidates = build_candidates(run_dir)
    if not candidates.empty:
        candidates.to_csv(os.path.join(out, "1_candidates.csv"), index=False)
    for src, dst in (("hits_by_backbone.csv", "2_backbone_evidence.csv"),
                     ("method_comparison_summary.csv", "3_method_comparison.csv")):
        path = os.path.join(run_dir, src)
        if os.path.isfile(path):
            shutil.copyfile(path, os.path.join(out, dst))
    with open(os.path.join(out, "4_run_summary.txt"), "w") as fh:
        fh.write(_summary_text(run_dir, candidates))
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(_README)

    for fig in ("hits_method_comparison.png", "hits_metric_enrichment.png",
                "sequence_budget_rarefaction.png"):
        path = os.path.join(run_dir, fig)
        if os.path.isfile(path):
            shutil.copyfile(path, os.path.join(figs, fig))
    plots_dir = os.path.join(run_dir, "evaluation_plots")
    if os.path.isdir(plots_dir):
        for name in sorted(os.listdir(plots_dir)):
            if name.endswith(".png"):
                shutil.copyfile(os.path.join(plots_dir, name),
                                os.path.join(figs, name))

    removed = prune_empty_dirs(run_dir)
    if removed:
        print(f"  pruned {len(removed)} empty director{'y' if len(removed)==1 else 'ies'}: "
              f"{', '.join(removed)}")
    return out


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        sys.exit(f"not a run directory: {run_dir}")
    out = write_results(run_dir)
    print(f"  results -> {out}")
    summary = os.path.join(out, "4_run_summary.txt")
    if os.path.isfile(summary):
        print()
        print(open(summary).read())


if __name__ == "__main__":
    main()
