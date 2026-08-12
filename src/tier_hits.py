"""Persist every design that passes the AF2 two-state metric gate, per arm.

Rationale
---------
`final_{strict,relaxed,consensus}.csv` are derived from the post-forwarding
frame, so they hold only the top-K designs carried into Boltz-2 scoring
(`POST_AF2_TOP_K`). Every other sequence passing the AF2 gate was scored and
tiered but not written; on one production run this omitted 142 of 166 passing
DynamicMPNN sequences.

Two further gaps are addressed:

  * The ProteinMPNN-MSD arm wrote no tier files, only
    `msd_final_all_ranked.csv`, giving no equivalent of `final_relaxed.csv`.
  * Tiers were vetoed run-wide by `af2_null_discriminates`, a single boolean
    from `paired_nulls.passes_stop_go`. That statistic compares the per-backbone
    median sequence against the null and therefore measures generator quality
    rather than the presence of individual hits; one weak metric (apo i_pae)
    emptied every tier while three others separated cleanly.

The hits written here are gate-level and per-arm, and carry the null evidence as
data, per-design margins together with a tail-enrichment test, rather than as a
veto. The run-level stop/go verdict is retained in `null_gate_supported` and in
the per-arm summary, so that a caller may still filter on it while a negative
verdict no longer suppresses the file.

Tiers are recomputed from the raw AF2 metrics through
`switch_gating.assign_af2_tiers`, disregarding any veto already applied to
`af2_relaxed`, so that the hit list is reproducible from the metrics alone.

Runnable standalone to backfill a finished run:

    python src/tier_hits.py outputs/<run>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import switch_gating

# Per-arm wiring. Each arm has its own AF2 gate table, its own scramble null,
# and its own backbone/sequence column names.
ARMS = [
    {
        "arm": "dynamicmpnn",
        "label": "DynamicMPNN",
        "gate_csv": "s5_5_af2_gate_all.csv",
        "null_csv": "af2_gate_null.csv",
        "backbone_col": "s1_rfd3_holo_description",
        "seq_col": "s5_dynamicmpnn_sequence",
        "color": "#3d6fb4",
    },
    {
        "arm": "proteinmpnn_msd",
        "label": "ProteinMPNN-MSD",
        "gate_csv": "s7_5_msd_af2_gate_all.csv",
        "null_csv": "msd_af2_gate_null.csv",
        "backbone_col": "backbone",
        "seq_col": "_msd_seq",
        "color": "#c9822e",
    },
]

METRICS = {
    "af2_holo_plddt": "higher",
    "af2_apo_plddt": "higher",
    "af2_holo_i_pae": "lower",
    "af2_apo_i_pae": "lower",
}

# Columns worth carrying into a hit file, when the arm's table has them.
_REPORT_COLS = [
    "af2_tier", "af2_switch_plddt", "af2_worst_ipae",
    "af2_holo_plddt", "af2_apo_plddt",
    "af2_holo_i_pae", "af2_apo_i_pae",
    "af2_holo_i_ptm", "af2_apo_i_ptm",
    "binder_ca_rmsd", "interface_jaccard", "geometry_pass",
    "n_null_wins", "worst_null_margin", "null_gate_supported",
]


def _margins(hits: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    """Per-design margin against that design's OWN backbone-matched scramble.

    The run-level AUC is a population statistic; this is the per-design question
    the reader of a hit list actually has ("is THIS design better than its own
    null?"). Designs whose scramble is missing get NaN margins and are counted as
    zero wins rather than silently dropped.
    """
    cols = [c for c in METRICS if c in hits.columns and c in null.columns]
    if not cols or "_real_design_id" not in null.columns:
        hits["n_null_wins"] = np.nan
        hits["worst_null_margin"] = np.nan
        return hits
    ref = (null.dropna(subset=["_real_design_id"])
               .drop_duplicates(subset="_real_design_id", keep="first")
               .set_index("_real_design_id"))
    keys = hits["poses_description"]
    marg_cols = []
    for c in cols:
        sign = 1.0 if METRICS[c] == "higher" else -1.0
        paired = keys.map(ref[c])
        name = f"{c}_null_margin"
        hits[name] = sign * (pd.to_numeric(hits[c], errors="coerce")
                             - pd.to_numeric(paired, errors="coerce"))
        marg_cols.append(name)
    hits["n_null_wins"] = (hits[marg_cols] > 0).sum(axis=1)
    hits["worst_null_margin"] = hits[marg_cols].min(axis=1)
    return hits


def _tail_enrichment(real: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    """Per-metric tail test: how many real vs null designs clear the null's own tail.

    This is the statistic the relaxed tier is actually built on, and the one the
    median-based stop/go misses when most designs are bad but a few are good.
    """
    thr = switch_gating.null_thresholds(null, METRICS)
    rows = []
    for c, direction in METRICS.items():
        if c not in thr or c not in real.columns:
            continue
        rv = pd.to_numeric(real[c], errors="coerce")
        nv = pd.to_numeric(null[c], errors="coerce")
        if direction == "higher":
            n_real, n_null = int((rv > thr[c]).sum()), int((nv > thr[c]).sum())
        else:
            n_real, n_null = int((rv < thr[c]).sum()), int((nv < thr[c]).sum())
        rows.append({
            "metric": c,
            "direction": direction,
            "null_tail_threshold": thr[c],
            "n_real_in_tail": n_real,
            "n_null_in_tail": n_null,
            "enrichment": (n_real / n_null) if n_null else np.nan,
        })
    return pd.DataFrame(rows)


def write_arm_hits(out_dir: str, arm_spec: dict) -> dict | None:
    """Write gate-level strict/relaxed hit files for one arm.

    Returns a summary dict (counts per tier, for the cross-arm comparison), or
    None when the arm did not run in this output directory.
    """
    gate_path = os.path.join(out_dir, arm_spec["gate_csv"])
    if not os.path.isfile(gate_path):
        return None
    df = pd.read_csv(gate_path)
    if df.empty or "af2_holo_plddt" not in df.columns:
        return None

    null_path = os.path.join(out_dir, arm_spec["null_csv"])
    null = pd.read_csv(null_path) if os.path.isfile(null_path) else None

    # Recompute from raw metrics so a veto already stamped onto af2_relaxed
    # upstream cannot erase the hits here.
    scored = switch_gating.assign_af2_tiers(
        df, "af2_holo_plddt", "af2_apo_plddt", "af2_holo_i_pae", "af2_apo_i_pae",
        null_df=null, holo_iptm="af2_holo_i_ptm", apo_iptm="af2_apo_i_ptm",
    )
    # Preserve the run-level stop/go verdict as data, not as a filter.
    scored["null_gate_supported"] = bool(
        df.get("af2_null_discriminates", pd.Series([False])).fillna(False).iloc[0]
    ) if "af2_null_discriminates" in df.columns else False

    if null is not None and len(null):
        scored = _margins(scored, null)
        enrich = _tail_enrichment(scored, null)
        enrich.insert(0, "arm", arm_spec["arm"])
        enrich.to_csv(
            os.path.join(out_dir, f"hits_{arm_spec['arm']}_null_tail_enrichment.csv"),
            index=False)
    else:
        scored["n_null_wins"] = np.nan
        scored["worst_null_margin"] = np.nan
        enrich = pd.DataFrame()

    bb, seq = arm_spec["backbone_col"], arm_spec["seq_col"]
    front = ["poses_description"] + [c for c in (bb, seq) if c in scored.columns]
    keep = front + [c for c in _REPORT_COLS if c in scored.columns]
    keep += [c for c in scored.columns if c.endswith("_null_margin") and c not in keep]

    summary = {"arm": arm_spec["arm"], "label": arm_spec["label"],
               "color": arm_spec["color"],
               "n_scored": len(scored),
               "n_backbones": int(scored[bb].nunique()) if bb in scored.columns else np.nan,
               "null_gate_supported": bool(scored["null_gate_supported"].iloc[0])}

    for tier, mask in (("strict", scored["af2_strict"].fillna(False)),
                       ("relaxed", scored["af2_relaxed"].fillna(False)
                        | scored["af2_strict"].fillna(False))):
        hits = scored[mask].sort_values("af2_switch_plddt", ascending=False)
        hits = hits[keep].copy()
        hits.insert(0, "rank", range(1, len(hits) + 1))
        path = os.path.join(out_dir, f"hits_{arm_spec['arm']}_{tier}.csv")
        hits.to_csv(path, index=False)
        summary[f"n_{tier}"] = len(hits)
        summary[f"n_backbones_{tier}"] = (
            int(hits[bb].nunique()) if bb in hits.columns else np.nan)
        # The false-positive floor: how many SCRAMBLES clear the same bar.
        if null is not None and len(null):
            n_null = switch_gating.assign_af2_tiers(
                null, "af2_holo_plddt", "af2_apo_plddt", "af2_holo_i_pae", "af2_apo_i_pae",
                null_df=null, holo_iptm="af2_holo_i_ptm", apo_iptm="af2_apo_i_ptm",
            )
            nm = (n_null["af2_strict"].fillna(False) if tier == "strict"
                  else n_null["af2_relaxed"].fillna(False) | n_null["af2_strict"].fillna(False))
            summary[f"n_{tier}_null"] = int(nm.sum())
        else:
            summary[f"n_{tier}_null"] = np.nan
    return summary


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c8ccd2")
    ax.tick_params(colors="#20242b", length=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e8ec", lw=0.8)


def plot_comparison(out_dir: str, summaries: list[dict]) -> str | None:
    """Grouped bars: passing designs per arm, sequences and backbones.

    Real arms take the two fixed categorical slots; the scramble null is neutral
    gray + hatched, so it reads as a reference floor rather than a third method
    (identity never rests on color alone).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not summaries:
        return None

    INK, MUTED, NULLC = "#20242b", "#5b6270", "#9aa0aa"
    tiers = ["strict", "relaxed"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    series = [(s["label"], s["color"], s) for s in summaries]
    x = np.arange(len(tiers))

    for ax, (unit, key, nkey) in zip(
            axes, [("sequences", "n_{}", "n_{}_null"),
                   ("backbones with ≥1 hit", "n_backbones_{}", None)]):
        # Each panel sizes its own group: only the sequence panel carries the
        # null floor bar, so the backbone panel must not reserve a slot for it
        # or its bars sit off-centre under the tick.
        n_bar = len(series) + (1 if nkey else 0)
        width = 0.8 / n_bar
        for i, (label, color, s) in enumerate(series):
            vals = [s.get(key.format(t), 0) or 0 for t in tiers]
            pos = x + (i - (n_bar - 1) / 2) * width
            ax.bar(pos, vals, width * 0.92, label=label, color=color,
                   edgecolor="white", linewidth=1.2, zorder=3)
            for p, v in zip(pos, vals):
                ax.annotate(f"{int(v)}", (p, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=9, color=INK)
        if nkey:
            # One combined null floor bar per tier (both arms use the same
            # scramble construction and the same n, so a single floor is honest).
            vals = [max(s.get(nkey.format(t), 0) or 0 for _, _, s in series)
                    for t in tiers]
            pos = x + (len(series) - (n_bar - 1) / 2) * width
            ax.bar(pos, vals, width * 0.92, label="scramble null (FP floor)",
                   color=NULLC, edgecolor="white", linewidth=1.2, hatch="///",
                   zorder=3)
            for p, v in zip(pos, vals):
                ax.annotate(f"{int(v)}", (p, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=9, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels(["strict\n(absolute bar)", "relaxed\n(beats null tail)"])
        ax.set_ylabel(f"passing {unit}")
        _style(ax)
        ax.margins(y=0.20)

    axes[0].set_title("Sequences clearing the two-state AF2 gate",
                      loc="left", color=INK, fontsize=11)
    axes[1].set_title("Independent backbones represented",
                      loc="left", color=INK, fontsize=11)
    # Upper LEFT: the tall bar is always the relaxed tier on the right, so the
    # left shoulder is the only reliably empty corner for the legend.
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    verdict = " / ".join(
        f"{s['label']}: run-level null stop-go "
        f"{'PASS' if s['null_gate_supported'] else 'FAIL'}"
        for s in summaries)
    fig.text(0.01, 0.035, verdict, ha="left", fontsize=8, color=MUTED)
    fig.text(0.01, 0.008,
             "Hits are written regardless of that verdict: stop-go compares the "
             "MEDIAN sequence per backbone, so it rates the generator, not a design.",
             ha="left", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = os.path.join(out_dir, "hits_method_comparison.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def _auc(real, null, higher_is_better: bool) -> float:
    """Common-language AUC: P(a random real design beats a random null)."""
    r = pd.to_numeric(pd.Series(real), errors="coerce").dropna().to_numpy()
    n = pd.to_numeric(pd.Series(null), errors="coerce").dropna().to_numpy()
    if len(r) < 3 or len(n) < 3:
        return float("nan")
    wins = (r[:, None] > n[None, :]).mean() + 0.5 * (r[:, None] == n[None, :]).mean()
    return float(wins if higher_is_better else 1.0 - wins)


def write_backbone_evidence(out_dir: str, min_auc: float = 0.70) -> pd.DataFrame:
    """Per-backbone discrimination against that backbone's OWN scrambles.

    The run-level stop/go compares the MEDIAN sequence per backbone and ANDs
    across metrics, so it answers "is the typical design better than chance" --
    a generator-quality question. It reported null_discriminates=False on a run
    that in fact contained 166 hits over 29 backbones.

    But the paired nulls are built PER DESIGN, so each backbone carries its own
    matched control set. That makes the far more useful question computable:
    does THIS backbone's sequence pool beat THIS backbone's scrambles? On the
    same production data that the run-level test failed, 18 of 80 backbones
    clear AUC>0.7 on all four metrics -- including apo i_pae, which carries
    almost no signal at population level.

    A backbone is `validated` when every metric beats its own null at `min_auc`.
    That is a per-backbone claim and does not depend on the run-level verdict.
    """
    frames = []
    for spec in ARMS:
        gate = os.path.join(out_dir, spec["gate_csv"])
        null_path = os.path.join(out_dir, spec["null_csv"])
        if not (os.path.isfile(gate) and os.path.isfile(null_path)):
            continue
        df, null = pd.read_csv(gate), pd.read_csv(null_path)
        bb = spec["backbone_col"]
        if bb not in df.columns or "_null_backbone" not in null.columns:
            continue
        scored = switch_gating.assign_af2_tiers(
            df, "af2_holo_plddt", "af2_apo_plddt", "af2_holo_i_pae", "af2_apo_i_pae",
            null_df=null, holo_iptm="af2_holo_i_ptm", apo_iptm="af2_apo_i_ptm",
        )
        passes = scored["af2_relaxed"].fillna(False) | scored["af2_strict"].fillna(False)
        scored = scored.assign(_pass=passes)
        null_by_bb = dict(tuple(null.groupby(null["_null_backbone"].astype(str))))
        rows = []
        for backbone, group in scored.groupby(scored[bb].astype(str)):
            ctrl = null_by_bb.get(backbone)
            if ctrl is None or len(ctrl) < 3:
                continue
            row = {"arm": spec["arm"], "backbone": backbone,
                   "n_sequences": len(group), "n_hits": int(group["_pass"].sum()),
                   "best_af2_switch_plddt": float(group["af2_switch_plddt"].max())}
            for metric, direction in METRICS.items():
                row[f"auc_{metric.replace('af2_', '')}"] = _auc(
                    group[metric], ctrl[metric], direction == "higher")
            aucs = [v for k, v in row.items() if k.startswith("auc_")]
            row["auc_min"] = float(np.nanmin(aucs)) if aucs else float("nan")
            row["validated"] = bool(row["auc_min"] >= min_auc)
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values(
        ["validated", "auc_min", "n_hits"], ascending=[False, False, False])
    out.insert(0, "rank", range(1, len(out) + 1))
    out.to_csv(os.path.join(out_dir, "hits_by_backbone.csv"), index=False)
    return out


def write_top_hits(out_dir: str, top_n: int = 15) -> pd.DataFrame:
    """One cross-arm leaderboard, normalised so both arms are directly comparable.

    Per-arm hit files use different backbone/sequence column names, which makes
    "show me the best designs in this run" a three-file join. This collapses them
    into `hits_top.csv`: one row per passing design, ranked, with the arm named,
    the AF2 structures attached where the design reached expensive scoring, and
    one design per backbone in the printed view, so that a single backbone with 21
    passing sequences cannot fill the whole leaderboard.
    """
    frames = []
    for spec in ARMS:
        path = os.path.join(out_dir, f"hits_{spec['arm']}_relaxed.csv")
        if not os.path.isfile(path):
            continue
        h = pd.read_csv(path)
        if h.empty:
            continue
        h = h.rename(columns={spec["backbone_col"]: "backbone",
                              spec["seq_col"]: "sequence"})
        h["arm"] = spec["arm"]
        frames.append(h)
    if not frames:
        return pd.DataFrame()
    allh = pd.concat(frames, ignore_index=True)

    # Attach structures + orthogonal Boltz/consensus columns for the designs that
    # were forwarded to expensive scoring (a minority — the rest are AF2-only).
    ranked_path = os.path.join(out_dir, "final_all_ranked.csv")
    extra = ["af2_holo_pdb", "af2_apo_pdb", "s6a_boltz_holo_iptm",
             "s6b_boltz_apo_iptm", "af2_rmsd_boltz_vs_af2_holo",
             "af2_rmsd_boltz_vs_af2_apo"]
    if os.path.isfile(ranked_path):
        fr = pd.read_csv(ranked_path)
        cols = ["poses_description"] + [c for c in extra if c in fr.columns]
        if len(cols) > 1:
            allh = allh.merge(fr[cols].drop_duplicates("poses_description"),
                              on="poses_description", how="left")
    allh["scored_by_boltz"] = (allh["s6a_boltz_holo_iptm"].notna()
                               if "s6a_boltz_holo_iptm" in allh.columns else False)
    allh = allh.sort_values("af2_switch_plddt", ascending=False).reset_index(drop=True)
    allh.insert(0, "overall_rank", range(1, len(allh) + 1))
    allh.to_csv(os.path.join(out_dir, "hits_top.csv"), index=False)

    best = (allh.sort_values("af2_switch_plddt", ascending=False)
                .drop_duplicates(subset=["arm", "backbone"], keep="first")
                .head(top_n))
    show = ["overall_rank", "poses_description", "arm", "af2_switch_plddt",
            "af2_holo_plddt", "af2_apo_plddt", "af2_holo_i_pae", "af2_apo_i_pae",
            "n_null_wins", "scored_by_boltz"]
    show = [c for c in show if c in best.columns]
    print(f"\n{'=' * 78}\nTOP HITS — best design per backbone, ranked by two-state "
          f"AF2 harmonic pLDDT\n{'=' * 78}")
    print(best[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    n_strict_like = int(((allh["af2_holo_i_pae"] < 0.34)
                         & (allh["af2_apo_i_pae"] < 0.34)).sum())
    print(f"\n  {len(allh)} passing designs over "
          f"{allh['backbone'].nunique()} backbones; "
          f"{n_strict_like} clear the strict i_pae bar (<0.34) in BOTH states.")
    print("  full leaderboard: hits_top.csv   per-arm: hits_<arm>_relaxed.csv")
    return allh


def plot_metric_enrichment(out_dir: str, summaries: list[dict]) -> str | None:
    """Per-metric tail enrichment, by arm — which metrics actually carry signal.

    Enrichment = (real designs in the null's tail) / (nulls in their own tail).
    1.0 is the no-signal line: at that point the metric cannot tell a design from
    a composition-matched scramble, so it must not be used to rank.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    frames = []
    for s in summaries:
        p = os.path.join(out_dir, f"hits_{s['arm']}_null_tail_enrichment.csv")
        if os.path.isfile(p):
            e = pd.read_csv(p)
            e["label"], e["color"] = s["label"], s["color"]
            frames.append(e)
    if not frames:
        return None
    e = pd.concat(frames, ignore_index=True)
    labels = {"af2_holo_plddt": "holo pLDDT", "af2_apo_plddt": "apo pLDDT",
              "af2_holo_i_pae": "holo i_pae", "af2_apo_i_pae": "apo i_pae"}
    order = list(labels)
    INK, MUTED = "#20242b", "#5b6270"

    arms = e[["label", "color"]].drop_duplicates().to_dict("records")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(order))
    h = 0.8 / len(arms)
    for i, a in enumerate(arms):
        sub = e[e["label"] == a["label"]].set_index("metric")
        vals = [sub["enrichment"].get(m, np.nan) for m in order]
        pos = y + (i - (len(arms) - 1) / 2) * h
        ax.barh(pos, vals, h * 0.92, label=a["label"], color=a["color"],
                edgecolor="white", linewidth=1.2, zorder=3)
        for p, v in zip(pos, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}×", (v, p), textcoords="offset points",
                            xytext=(4, 0), va="center", fontsize=9, color=INK)
    ax.axvline(1.0, color="#b3271e", lw=1.4, ls="--", zorder=4,
               label="1× = indistinguishable from scramble")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("tail enrichment over composition-matched scramble null (×)")
    ax.set_title("Which AF2 metrics carry real signal", loc="left",
                 color=INK, fontsize=11)
    _style(ax)
    ax.grid(axis="x", color="#e6e8ec", lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.margins(x=0.16)
    fig.text(0.01, 0.02, "apo i_pae is the state-2 interface: near 1× it cannot "
             "rank designs, and it is the metric that vetoed this run.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(out_dir, "hits_metric_enrichment.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def write_tier_hits(out_dir: str) -> pd.DataFrame:
    """Write per-arm hit files + the cross-arm comparison table and figures."""
    summaries = [s for s in (write_arm_hits(out_dir, a) for a in ARMS) if s]
    if not summaries:
        return pd.DataFrame()
    table = pd.DataFrame(summaries)
    table.to_csv(os.path.join(out_dir, "hits_summary_by_arm.csv"), index=False)
    plot_comparison(out_dir, summaries)
    plot_metric_enrichment(out_dir, summaries)
    write_backbone_evidence(out_dir)
    write_top_hits(out_dir)
    return table


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out_dir = sys.argv[1]
    table = write_tier_hits(out_dir)
    if table.empty:
        sys.exit(f"no AF2 gate tables found in {out_dir}")
    cols = [c for c in ("label", "n_scored", "n_backbones", "n_strict", "n_relaxed",
                        "n_backbones_relaxed", "n_relaxed_null",
                        "null_gate_supported") if c in table.columns]
    print(table[cols].to_string(index=False))
    print(f"\nwrote hits_<arm>_{{strict,relaxed}}.csv, hits_summary_by_arm.csv, "
          f"hits_method_comparison.png to {out_dir}/")


if __name__ == "__main__":
    main()
