"""Marginal return of the sequences-per-backbone budget.

The AF2 gate dominates the compute and scales linearly with
`sampling.dmpnn_nseq` and `sampling.mpnn_msd_nseq`, each sequence costing two
AF2 initial-guess predictions and twice that again with the paired null. Backbone
generation does not scale in this way, RFdiffusion3 being a small fraction of the
same budget, so the sequence budget is the quantity to trade for additional
backbones once the marginal sequence ceases to contribute.

This is measured directly from a finished run by rarefaction: k of the k_max
sequences scored per backbone are resampled, giving the yield a run with
`nseq = k` would have produced. The sequences per backbone are exchangeable draws
from the same sampler at one temperature, so subsampling them estimates the
smaller-budget run.

Two summary quantities are reported:

  backbones_with_hit  independent-hit count; 32 passing sequences on one backbone
                      constitute one design rather than 32.
  E[best]             expected best two-state AF2 pLDDT per backbone at budget k.

A variance decomposition accompanies them, giving the share of af2_switch_plddt
variance attributable to backbone identity. A high share indicates that backbone
identity rather than the sequence search determines design success, and that
compute is better spent upstream.

    python src/sequence_budget.py outputs/<run> [--draws 400]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ARMS = [
    ("DynamicMPNN", "s5_5_af2_gate_all.csv", "s1_rfd3_holo_description", "#3d6fb4"),
    ("ProteinMPNN-MSD", "s7_5_msd_af2_gate_all.csv", "backbone", "#c9822e"),
]
BUDGETS = [1, 2, 4, 8, 12, 16, 24, 32]


def rarefy(df: pd.DataFrame, backbone_col: str, draws: int = 400,
           seed: int = 0) -> pd.DataFrame:
    """Expected yield at each sequence budget k, by resampling within backbone."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_pass"] = (df.get("af2_relaxed", False).fillna(False)
                   | df.get("af2_strict", False).fillna(False))
    groups = {
        key: (g["_pass"].to_numpy(),
              pd.to_numeric(g["af2_switch_plddt"], errors="coerce").to_numpy())
        for key, g in df.groupby(backbone_col)
    }
    k_max = max(len(p) for p, _ in groups.values())
    rows = []
    for k in [b for b in BUDGETS if b <= k_max]:
        nb, nh, best, top = [], [], [], []
        for _ in range(draws):
            bb = hits = 0
            per_bb = []
            for passes, plddt in groups.values():
                idx = rng.choice(len(passes), size=min(k, len(passes)), replace=False)
                s = int(passes[idx].sum())
                hits += s
                bb += s > 0
                vals = plddt[idx]
                per_bb.append(np.nanmax(vals) if np.isfinite(vals).any() else np.nan)
            nb.append(bb)
            nh.append(hits)
            best.append(np.nanmean(per_bb))
            top.append(np.nanmax(per_bb))
        rows.append({
            "k_seqs_per_backbone": k,
            "backbones_with_hit": float(np.mean(nb)),
            "total_hits": float(np.mean(nh)),
            "mean_best_plddt_per_backbone": float(np.mean(best)),
            "best_plddt_in_run": float(np.mean(top)),
            # 2 AF2-IG predictions per sequence per state-pair, doubled by the
            # paired null: the compute this budget actually costs.
            "af2_predictions": int(2 * 2 * k * len(groups)),
        })
    out = pd.DataFrame(rows)
    full = out.iloc[-1]
    out["frac_backbones_vs_full"] = out["backbones_with_hit"] / full["backbones_with_hit"]
    out["frac_compute_vs_full"] = out["af2_predictions"] / full["af2_predictions"]
    return out


def reallocation(curve: pd.DataFrame, n_backbones: int,
                 upstream_preds_per_backbone: float = 0.0,
                 n_arms: int = 1) -> pd.DataFrame:
    """Compare k sequences on N backbones against k/2 sequences on 2N, at equal AF2 cost.

    Per-backbone hit PROBABILITY at budget k is estimated from this run, then
    extrapolated across backbone count — the honest direction, since backbones
    enter the yield linearly and independently while sequences share a backbone
    and saturate.

    `upstream_preds_per_backbone` is what a backbone costs BEFORE its sequences
    are gated (state-2 generation and designability triage, amortised over the
    backbones that survive it). Ignoring it badly overstates the gain from cutting
    k: an extra backbone is not free, so the affordable backbone count does not
    scale with 1/k. Measure it with `upstream_af2_cost()` rather than guessing.

    Caveat that no arithmetic here can remove: this extrapolates the per-backbone
    hit rate of THIS target's backbones to more of them. It is a compute-allocation
    estimate, not a prediction about a different target.
    """
    rows = []
    for _, r in curve.iterrows():
        k = int(r["k_seqs_per_backbone"])
        p_hit = r["backbones_with_hit"] / n_backbones
        # 2 states x n_arms sequence-gate predictions, plus the paired null.
        gate = 4.0 * n_arms * k
        rows.append({
            "k_seqs_per_backbone": k,
            "per_backbone_hit_rate": p_hit,
            "gate_preds_per_backbone": gate,
            "total_preds_per_backbone": upstream_preds_per_backbone + gate,
        })
    out = pd.DataFrame(rows)
    budget = float(n_backbones * out["total_preds_per_backbone"].iloc[-1])
    out["af2_budget"] = budget
    out["backbones_affordable"] = budget / out["total_preds_per_backbone"]
    out["expected_hits_at_equal_cost"] = (out["backbones_affordable"]
                                          * out["per_backbone_hit_rate"])
    out["vs_current"] = (out["expected_hits_at_equal_cost"]
                         / out["expected_hits_at_equal_cost"].iloc[-1])
    return out


def upstream_af2_cost(out_dir: str, n_forwarded: int) -> float:
    """AF2 predictions spent per FORWARDED backbone before sequence gating.

    Counted from the designability steps' own manifests, then amortised over the
    backbones that actually reached the sequence gate — state-2 triage discards
    most candidates, and that waste is part of what a backbone costs.
    """
    total = 0
    for step in ("s1_5_desig_af2", "s2_5_desig_af2"):
        man = os.path.join(out_dir, step, "af2", "manifest.csv")
        if not os.path.isfile(man):
            for root, _, files in os.walk(os.path.join(out_dir, step)):
                if "manifest.csv" in files:
                    man = os.path.join(root, "manifest.csv")
                    break
        if os.path.isfile(man):
            total += max(0, sum(1 for _ in open(man)) - 1)
    return total / n_forwarded if n_forwarded else 0.0


def backbone_variance_share(df: pd.DataFrame, backbone_col: str) -> float:
    """Share of af2_switch_plddt variance attributable to backbone identity."""
    y = pd.to_numeric(df["af2_switch_plddt"], errors="coerce")
    d = pd.DataFrame({"y": y, "b": df[backbone_col]}).dropna()
    if d["b"].nunique() < 2:
        return float("nan")
    means = d.groupby("b")["y"].mean()
    n = d.groupby("b")["y"].size()
    ss_between = float((n * (means - d["y"].mean()) ** 2).sum())
    ss_within = float(((d["y"] - d["b"].map(means)) ** 2).sum())
    total = ss_between + ss_within
    return ss_between / total if total > 0 else float("nan")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c8ccd2")
    ax.tick_params(colors="#20242b", length=3)
    ax.set_axisbelow(True)
    ax.grid(color="#e6e8ec", lw=0.8)


def plot(out_dir: str, curves: list[tuple]) -> str | None:
    """Rarefaction curves: independent hits and best score vs sequence budget."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    INK, MUTED = "#20242b", "#5b6270"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for label, color, curve, share in curves:
        for ax, col in zip(axes, ["backbones_with_hit",
                                  "mean_best_plddt_per_backbone"]):
            ax.plot(curve["k_seqs_per_backbone"], curve[col], "-o", color=color,
                    lw=2, ms=6, label=f"{label} ({share:.0%} of variance is backbone)",
                    zorder=3)
    for ax, ylab, title in zip(
            axes,
            ["backbones with ≥1 passing design", "mean best two-state AF2 pLDDT"],
            ["Independent hits vs sequence budget",
             "Design quality vs sequence budget"]):
        ax.set_xlabel("sequences sampled per backbone (k)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xticks(BUDGETS)
        _style(ax)
    # Mark the halving that the curves are being read to justify.
    for ax in axes:
        ax.axvline(16, color=MUTED, lw=1.2, ls=":", zorder=2)
    axes[0].annotate("k=16", (16, axes[0].get_ylim()[0]), xytext=(4, 6),
                     textcoords="offset points", fontsize=8.5, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    # Be precise about what each panel shows: QUALITY saturates, hit COUNT does
    # not. The case for cutting k is not that the left curve is flat — it is that
    # the left curve is sub-linear in k while backbone count enters linearly, so
    # the same AF2 spend buys more hits as backbones than as sequences.
    fig.text(0.01, 0.02,
             "Quality (right) saturates by k≈8-12. Hit count (left) keeps rising but "
             "SUB-linearly, while backbones enter linearly — so equal AF2 spend buys "
             "more hits as backbones than as sequences.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(out_dir, "sequence_budget_rarefaction.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def analyse(out_dir: str, draws: int = 400) -> pd.DataFrame:
    tables, curves = [], []
    for label, csv, bb, color in ARMS:
        path = os.path.join(out_dir, csv)
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        if df.empty or bb not in df.columns or "af2_switch_plddt" not in df.columns:
            continue
        curve = rarefy(df, bb, draws=draws)
        share = backbone_variance_share(df, bb)
        curve.insert(0, "arm", label)
        curve["backbone_variance_share"] = share
        tables.append(curve)
        curves.append((label, color, curve, share))
    if not tables:
        return pd.DataFrame()
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(os.path.join(out_dir, "sequence_budget_rarefaction.csv"), index=False)
    plot(out_dir, curves)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--arms", type=int, default=1,
                    help="sequence-design arms kept per backbone in the PLANNED "
                         "run (1 = DynamicMPNN only; 2 = also ProteinMPNN-MSD)")
    a = ap.parse_args()
    out = analyse(a.run, draws=a.draws)
    if out.empty:
        raise SystemExit(f"no AF2 gate tables in {a.run}")
    pd.set_option("display.width", 200)
    for arm, g in out.groupby("arm", sort=False):
        print(f"\n=== {arm}  (backbone explains "
              f"{g['backbone_variance_share'].iloc[0]:.1%} of score variance)")
        print(g[["k_seqs_per_backbone", "backbones_with_hit", "total_hits",
                 "mean_best_plddt_per_backbone", "best_plddt_in_run",
                 "af2_predictions", "frac_backbones_vs_full",
                 "frac_compute_vs_full"]].round(3).to_string(index=False))
        nb = int(g["af2_predictions"].iloc[0] / (4 * g["k_seqs_per_backbone"].iloc[0]))
        up = upstream_af2_cost(a.run, nb)
        print(f"\n  reallocation at EQUAL AF2 cost ({nb} backbones at k="
              f"{int(g['k_seqs_per_backbone'].iloc[-1])} today; upstream "
              f"state-1/state-2 triage costs {up:.0f} AF2 preds per forwarded "
              f"backbone, counted as part of a backbone's price):")
        print(reallocation(g, nb, upstream_preds_per_backbone=up,
                           n_arms=a.arms).round(3).to_string(index=False))
    print(f"\nwrote sequence_budget_rarefaction.{{csv,png}} to {a.run}/")


if __name__ == "__main__":
    main()
