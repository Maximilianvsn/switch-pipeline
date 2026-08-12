"""
Cross-replicate comparison for independent production runs of switch_pipeline.py.

Each replicate receives its own full evaluation at the end of the pipeline run,
written to outputs/<run-name>/evaluation_plots/. This script addresses a separate
question: whether a result reproduces across independent runs, or reflects a
single diffusion trajectory, backbone sample or scramble draw. A single run
cannot resolve this by construction.

Throughout, replicates are runs with identical configuration and code but
different RFdiffusion3 and DynamicMPNN random draws, submitted separately.
RFdiffusion3 does not apply a fixed seed across invocations: the radius of
gyration of identically named backbones differs between two independent runs,
so distinct --run-name submissions already constitute independent samples.

Usage:
    conda activate protflow
    python compare_replicates.py --run-names prod_v2_r1 prod_v2_r2 prod_v2_r3
    python compare_replicates.py --outputs-dirs outputs/prod_v2_r1 outputs/prod_v2_r2 outputs/prod_v2_r3
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_style import (  # one shared definition of the house style
    _wilson_ci, apply_house_style, METHOD_COLORS, REF_LINE_KW,
    DIVERGING_POS, DIVERGING_NEG, INK, INK_MUTED,
)


def parse_args():
    p = argparse.ArgumentParser(description="Cross-replicate statistics for independent production runs")
    p.add_argument("--run-names", nargs="+", help="run-names under outputs/ (e.g. prod_v2_r1 prod_v2_r2 prod_v2_r3)")
    p.add_argument("--outputs-dirs", nargs="+", help="explicit outputs dirs (alternative to --run-names)")
    p.add_argument("--out-dir", default=None, help="where to write the comparison report (default: outputs/_replicate_comparison)")
    return p.parse_args()


# ── per-replicate loaders (each isolated: a missing file in one replicate must
# never crash the whole comparison, same discipline as evaluate_results.py) ──

def _load_funnel_step(outputs_dir: str, step: str) -> int | None:
    path = os.path.join(outputs_dir, "funnel_summary.csv")
    if not os.path.isfile(path):
        return None
    f = pd.read_csv(path)
    row = f[f["step"] == step]
    return int(row["n_designs"].iloc[-1]) if len(row) else None


def _load_af2_null_separation(outputs_dir: str) -> pd.DataFrame | None:
    # The live protein-only pipeline writes this at the run root. Retain the
    # two evaluation/legacy locations so older completed runs remain usable.
    candidates = (
        os.path.join(outputs_dir, "af2_null_separation.csv"),
        os.path.join(outputs_dir, "evaluation_plots", "af2_paired_null_audit.csv"),
        os.path.join(outputs_dir, "evaluation_plots", "af2_gate_null_separation.csv"),
    )
    for path in candidates:
        if os.path.isfile(path):
            return pd.read_csv(path)
    return None


def _load_af2_gate(outputs_dir: str) -> pd.DataFrame | None:
    path = os.path.join(outputs_dir, "s5_5_af2_gate_all.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


def _load_msd_af2_gate(outputs_dir: str) -> pd.DataFrame | None:
    path = os.path.join(outputs_dir, "s7_5_msd_af2_gate_all.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


def _load_designability(outputs_dir: str) -> tuple[int, int] | None:
    """(kept, total) backbones from the Step 1.5 AF2 designability pre-filter."""
    path = os.path.join(outputs_dir, "funnel_summary.csv")
    if not os.path.isfile(path):
        return None
    f = pd.read_csv(path)
    before = f[f["step"] == "s1_rfd3_holo"]
    after = f[f["step"] == "s1_5_designability"]
    if len(before) and len(after):
        return int(after["n_designs"].iloc[-1]), int(before["n_designs"].iloc[-1])
    return None


def _load_specificity(outputs_dir: str) -> pd.DataFrame | None:
    path = os.path.join(outputs_dir, "specificity_report.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


def _load_conditionality(outputs_dir: str) -> pd.DataFrame | None:
    path = os.path.join(outputs_dir, "conditionality_report.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


def _load_step_runtime_total(outputs_dir: str) -> float | None:
    path = os.path.join(outputs_dir, "funnel_summary.csv")
    if not os.path.isfile(path):
        return None
    f = pd.read_csv(path)
    if len(f) < 2:
        return None
    t0 = pd.to_datetime(f["timestamp"].iloc[0])
    t1 = pd.to_datetime(f["timestamp"].iloc[-1])
    return (t1 - t0).total_seconds() / 3600.0


def _top_design_overlap(dfs: list[pd.DataFrame | None], seq_col: str, top_n: int = 10) -> float | None:
    """Mean pairwise sequence identity between each replicate's TOP-N binder
    sequences (by af2_switch_plddt) and every other replicate's top-N. High =
    independent runs are converging on similar solutions (a real attractor in
    sequence space); low = each run finds unrelated one-off solutions."""
    tops = []
    for df in dfs:
        if df is None or seq_col not in df.columns or "af2_switch_plddt" not in df.columns:
            continue
        t = df.sort_values("af2_switch_plddt", ascending=False).head(top_n)[seq_col].dropna()
        tops.append([str(s).split(":")[-1] for s in t])
    if len(tops) < 2:
        return None
    ids = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            for s1 in tops[i]:
                for s2 in tops[j]:
                    n = min(len(s1), len(s2))
                    if n == 0:
                        continue
                    ids.append(sum(a == b for a, b in zip(s1[:n], s2[:n])) / n)
    return float(np.mean(ids)) if ids else None


def compare_replicates(outputs_dirs: list[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    apply_house_style()
    names = [os.path.basename(d.rstrip("/")) for d in outputs_dirs]
    n_rep = len(outputs_dirs)
    lines = [f"=== Cross-replicate comparison ({n_rep} independent runs) ===", "",
             f"Replicates: {', '.join(names)}", ""]

    # 1. Designability pre-filter rate (Wilson CI per replicate)
    desig = [_load_designability(d) for d in outputs_dirs]
    rows, desig_missing = [], []
    for name, r in zip(names, desig):
        if r is None:
            desig_missing.append(name)
            continue
        k, n = r
        p, lo, hi = _wilson_ci(k, n)
        rows.append({"replicate": name, "kept": k, "total": n, "rate": p, "lo": lo, "hi": hi})
    desig_df = pd.DataFrame(rows)
    if len(desig_df) or desig_missing:
        lines.append("Backbone designability rate (AF2 pre-filter kept/total), Wilson 95% CI:")
    for name in desig_missing:
        lines.append(f"  ({name}: no designability pre-filter data -- older run or af2.designability.enabled: false)")
    if len(desig_df):
        desig_df.to_csv(os.path.join(out_dir, "designability_rate_by_replicate.csv"), index=False)
        overlapping = True
        for _, r in desig_df.iterrows():
            lines.append(f"  {r['replicate']:<20} {r['kept']:>4}/{int(r['total']):<4} "
                         f"= {r['rate']:.1%}  [{r['lo']:.1%}, {r['hi']:.1%}]")
        if len(desig_df) > 1:
            overlapping = desig_df["lo"].max() <= desig_df["hi"].min()
            lines.append(f"  CIs {'OVERLAP (consistent across replicates)' if overlapping else 'DO NOT ALL OVERLAP -- investigate a replicate-specific effect (bad diffusion draw, etc.)'}")
        lines.append("")

        fig, ax = plt.subplots(figsize=(1.2 * n_rep + 2, 4))
        ax.bar(desig_df["replicate"], desig_df["rate"],
               yerr=[desig_df["rate"] - desig_df["lo"], desig_df["hi"] - desig_df["rate"]],
               capsize=4, color=METHOD_COLORS["DynamicMPNN"])
        ax.set_ylabel("designability rate"); ax.set_title("Backbone designability rate per replicate (Wilson 95% CI)")
        plt.xticks(rotation=20, ha="right"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "designability_rate_by_replicate.png"), dpi=150); plt.close()

    # 2. AF2-gate null-separation (does discrimination itself replicate?)
    seps = [_load_af2_null_separation(d) for d in outputs_dirs]
    sep_rows = []
    for name, s in zip(names, seps):
        if s is None:
            continue
        for _, r in s.iterrows():
            sep_rows.append({"replicate": name, **r.to_dict()})
    sep_df = pd.DataFrame(sep_rows)
    if len(sep_df):
        sep_df.to_csv(os.path.join(out_dir, "af2_null_separation_by_replicate.csv"), index=False)
        lines.append("AF2-gate null-separation (AUC) per replicate — the validity check itself, replicated:")
        for metric in sep_df["metric"].unique():
            sub = sep_df[sep_df["metric"] == metric]
            aucs = ", ".join(f"{r['replicate']}={r['auc']:.2f}" for _, r in sub.iterrows())
            all_disc = bool((sub["auc"] >= 0.7).all())
            lines.append(f"  {metric:<18} {aucs}  [{'ALL discriminate' if all_disc else 'INCONSISTENT -- do not trust blindly'}]")
        lines.append("")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        metrics_order = list(sep_df["metric"].unique())
        width = 0.8 / max(n_rep, 1)
        for i, name in enumerate(names):
            sub = sep_df[sep_df["replicate"] == name].set_index("metric").reindex(metrics_order)
            xs = np.arange(len(metrics_order)) + i * width
            ax.bar(xs, sub["auc"], width=width, label=name)
        ax.axhline(0.7, **REF_LINE_KW)
        ax.set_xticks(np.arange(len(metrics_order)) + width * (n_rep - 1) / 2)
        ax.set_xticklabels(metrics_order, rotation=20, ha="right")
        ax.set_ylabel("AUC (real vs null)"); ax.set_ylim(0, 1.05)
        ax.set_title("AF2-gate discrimination (AUC) across replicates\n(dashed line = 0.7 discrimination threshold)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "af2_null_separation_by_replicate.png"), dpi=150); plt.close()

    # 3. Tier rates (strict/relaxed/fail) per replicate
    gates = [_load_af2_gate(d) for d in outputs_dirs]
    tier_rows = []
    for name, g in zip(names, gates):
        if g is None or "af2_tier" not in g.columns:
            continue
        vc = g["af2_tier"].value_counts()
        n = len(g)
        for tier in ("strict", "relaxed", "fail"):
            k = int(vc.get(tier, 0))
            p, lo, hi = _wilson_ci(k, n)
            tier_rows.append({"replicate": name, "tier": tier, "k": k, "n": n, "rate": p, "lo": lo, "hi": hi})
    tier_df = pd.DataFrame(tier_rows)
    if len(tier_df):
        tier_df.to_csv(os.path.join(out_dir, "af2_tier_rates_by_replicate.csv"), index=False)
        lines.append("AF2-gate tier rates per replicate (all real sequences scored at the gate):")
        for tier in ("strict", "relaxed", "fail"):
            sub = tier_df[tier_df["tier"] == tier]
            if not len(sub):
                continue
            vals = ", ".join(f"{r['replicate']}={r['rate']:.1%}" for _, r in sub.iterrows())
            lines.append(f"  {tier:<8} {vals}")
        lines.append("")

    # 4. DynamicMPNN vs ProteinMPNN-MSD win rate, per replicate
    msd_gates = [_load_msd_af2_gate(d) for d in outputs_dirs]
    win_rows = []
    for name, g_dmpnn, g_msd in zip(names, gates, msd_gates):
        if g_dmpnn is None or g_msd is None:
            continue
        if "af2_switch_plddt" not in g_dmpnn.columns or "af2_switch_plddt" not in g_msd.columns:
            continue
        dm_best = g_dmpnn.groupby("s1_rfd3_holo_description")["af2_switch_plddt"].max() \
            if "s1_rfd3_holo_description" in g_dmpnn.columns else pd.Series(dtype=float)
        msd_best = g_msd.groupby("backbone")["af2_switch_plddt"].max() if "backbone" in g_msd.columns else pd.Series(dtype=float)
        shared = dm_best.index.intersection(msd_best.index)
        if not len(shared):
            continue
        dmpnn_wins = int((dm_best.loc[shared] > msd_best.loc[shared]).sum())
        win_rows.append({"replicate": name, "n_shared_backbones": len(shared),
                         "dmpnn_wins": dmpnn_wins, "dmpnn_win_rate": dmpnn_wins / len(shared)})
    win_df = pd.DataFrame(win_rows)
    if len(win_df):
        win_df.to_csv(os.path.join(out_dir, "dmpnn_vs_msd_winrate_by_replicate.csv"), index=False)
        lines.append("DynamicMPNN-vs-MSD win rate (per shared backbone, by AF2 harmonic pLDDT), per replicate:")
        for _, r in win_df.iterrows():
            lines.append(f"  {r['replicate']:<20} DynamicMPNN wins {r['dmpnn_wins']}/{r['n_shared_backbones']} "
                         f"({r['dmpnn_win_rate']:.0%})")
        spread = win_df["dmpnn_win_rate"].max() - win_df["dmpnn_win_rate"].min() if len(win_df) > 1 else 0.0
        lines.append(f"  spread across replicates: {spread:.0%} "
                     f"{'(consistent)' if spread < 0.20 else '(NOTABLE spread -- treat the method-comparison claim cautiously)'}")
        lines.append("")

    # 5. Top-design cross-replicate sequence overlap
    overlap = _top_design_overlap(gates, "s5_dynamicmpnn_sequence" if any(
        g is not None and "s5_dynamicmpnn_sequence" in g.columns for g in gates) else "_msd_seq")
    if overlap is not None:
        lines.append(f"Top-10-design cross-replicate sequence identity: {overlap:.1%} "
                     f"({'high -- independent runs converge on similar solutions' if overlap > 0.4 else 'low -- each run finds largely unrelated solutions (expected for de novo design at this diversity)'})")
        lines.append("")

    # 6. Runtime per replicate (resource planning)
    rts = [(name, _load_step_runtime_total(d)) for name, d in zip(names, outputs_dirs)]
    rts = [(n, t) for n, t in rts if t is not None]
    if rts:
        lines.append("Total wall-clock per replicate (funnel first->last timestamp; queue-wait included):")
        for n, t in rts:
            lines.append(f"  {n:<20} {t:.1f} h")
        lines.append("")

    # Write summary
    summary_path = os.path.join(out_dir, "cross_replicate_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {summary_path} + supporting CSVs/plots to {out_dir}")


def main():
    args = parse_args()
    WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.run_names:
        outputs_dirs = [os.path.join(WS, "outputs", r) for r in args.run_names]
    elif args.outputs_dirs:
        outputs_dirs = [os.path.abspath(d) for d in args.outputs_dirs]
    else:
        raise SystemExit("Provide --run-names or --outputs-dirs (need >=2 replicates)")
    if len(outputs_dirs) < 2:
        raise SystemExit("Provide at least 2 replicate output directories")
    missing = [d for d in outputs_dirs if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f"Outputs dir(s) not found: {missing}")
    out_dir = args.out_dir or os.path.join(WS, "outputs", "_replicate_comparison_" + "_".join(
        os.path.basename(d.rstrip("/")) for d in outputs_dirs))
    compare_replicates(outputs_dirs, out_dir)


if __name__ == "__main__":
    main()
