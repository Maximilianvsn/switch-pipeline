"""Extended DynamicMPNN vs ProteinMPNN-MSD comparison + binder-diagnostic plots.

Consumes the per-sequence AF2 gate tables and nulls that the pipeline already
writes, and produces method-vs-method and per-state/per-metric figures. Wired
into protein_only_evaluation.run_protein_only_evaluation (non-fatal), and also
runnable standalone for backfill:

    python src/method_comparison_plots.py outputs/<run>
"""
from __future__ import annotations
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DMP = "#3d6fb4"   # DynamicMPNN
MSD = "#c9822e"   # ProteinMPNN-MSD
REAL = "#3d6fb4"
NULLC = "#9aa0aa"
INK = "#20242b"

# AF2 metrics: (column, human label, "higher"|"lower" is better)
AF2_METRICS = [
    ("af2_holo_plddt", "holo pLDDT", "higher"),
    ("af2_apo_plddt", "apo pLDDT", "higher"),
    ("af2_holo_i_pae", "holo i_pae", "lower"),
    ("af2_apo_i_pae", "apo i_pae", "lower"),
    ("af2_holo_i_ptm", "holo i_ptm", "higher"),
    ("af2_apo_i_ptm", "apo i_ptm", "higher"),
]


def _read(path):
    return pd.read_csv(path) if os.path.isfile(path) else None


def _best_per_backbone(df, bb_col, metric, direction):
    if df is None or bb_col not in df.columns or metric not in df.columns:
        return None
    v = df[[bb_col, metric]].copy()
    v[metric] = pd.to_numeric(v[metric], errors="coerce")
    g = v.groupby(bb_col)[metric]
    return (g.max() if direction == "higher" else g.min()).dropna()


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#e6e8ec", linewidth=0.7)
    ax.set_axisbelow(True)


def _load(outputs_dir):
    d = outputs_dir
    return {
        "dmp_gate": _read(os.path.join(d, "s5_5_af2_gate_all.csv")),
        "msd_gate": _read(os.path.join(d, "s7_5_msd_af2_gate_all.csv")),
        "dmp_null": _read(os.path.join(d, "af2_gate_null.csv")),
        "msd_null": _read(os.path.join(d, "msd_af2_gate_null.csv")),
        "paired": _read(os.path.join(d, "method_comparison_paired.csv")),
        "funnel": _read(os.path.join(d, "funnel_summary.csv")),
        "final": _read(os.path.join(d, "final_all_ranked.csv")),
        "msd_final": _read(os.path.join(d, "msd_final_all_ranked.csv")),
    }


def _bb(df):
    for c in ("s1_rfd3_holo_description", "backbone"):
        if df is not None and c in df.columns:
            return c
    return None


# ----------------------------------------------------------------------------- plots
def plot_delta_distribution(data, plots_dir):
    paired = data["paired"]
    if paired is None or "delta_best_af2_switch_plddt" not in paired.columns:
        return None
    delta = pd.to_numeric(paired["delta_best_af2_switch_plddt"], errors="coerce").dropna()
    if delta.empty:
        return None
    frac = float((delta > 0).mean())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(delta, bins=24, color=DMP, alpha=0.85, edgecolor="white")
    ax.axvline(0, color=INK, lw=1.2, ls="--")
    ax.axvline(delta.median(), color=MSD, lw=1.8, label=f"median = {delta.median():+.3f}")
    _style(ax)
    ax.set(xlabel="best switch-pLDDT:  DynamicMPNN  -  ProteinMPNN-MSD  (per backbone)",
           ylabel="backbones",
           title=f"DynamicMPNN advantage per backbone  (n={len(delta)})")
    ax.text(0.98, 0.95, f"DynamicMPNN better on {frac*100:.0f}% of backbones",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#eef1f5", ec="none"))
    ax.legend(fontsize=8, frameon=False)
    p = os.path.join(plots_dir, "method_delta_distribution.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def plot_metric_violins(data, plots_dir):
    dmp, msd = data["dmp_gate"], data["msd_gate"]
    bb_d, bb_m = _bb(dmp), _bb(msd)
    if bb_d is None or bb_m is None:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    made = False
    for ax, (col, lab, direction) in zip(axes.ravel(), AF2_METRICS):
        d = _best_per_backbone(dmp, bb_d, col, direction)
        m = _best_per_backbone(msd, bb_m, col, direction)
        if d is None or m is None:
            ax.axis("off"); continue
        made = True
        parts = ax.violinplot([d.values, m.values], showmedians=True, widths=0.85)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([DMP, MSD][i]); pc.set_alpha(0.55)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set_color(INK); parts[key].set_linewidth(1.0)
        _style(ax)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["DynamicMPNN", "MSD"], fontsize=8)
        ax.set_title(f"{lab}  ({'higher' if direction=='higher' else 'lower'} better)", fontsize=9)
    if not made:
        plt.close(fig); return None
    fig.suptitle("Per-metric, best-per-backbone: DynamicMPNN vs ProteinMPNN-MSD",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(plots_dir, "method_metric_violins.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


def plot_win_rate_per_metric(data, plots_dir):
    dmp, msd = data["dmp_gate"], data["msd_gate"]
    bb_d, bb_m = _bb(dmp), _bb(msd)
    if bb_d is None or bb_m is None:
        return None
    labels, rates = [], []
    for col, lab, direction in AF2_METRICS + [("af2_switch_plddt", "switch-pLDDT", "higher")]:
        d = _best_per_backbone(dmp, bb_d, col, direction)
        m = _best_per_backbone(msd, bb_m, col, direction)
        if d is None or m is None:
            continue
        j = pd.concat([d.rename("d"), m.rename("m")], axis=1).dropna()
        if j.empty:
            continue
        win = (j["d"] > j["m"]) if direction == "higher" else (j["d"] < j["m"])
        labels.append(lab); rates.append(float(win.mean()))
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = [DMP if r >= 0.5 else MSD for r in rates]
    ax.bar(labels, rates, color=colors, alpha=0.9, edgecolor="white")
    ax.axhline(0.5, color=INK, ls="--", lw=1.1)
    _style(ax); ax.set_ylim(0, 1)
    ax.set(ylabel="fraction of backbones DynamicMPNN wins",
           title="Backbone-paired win rate, per metric (DynamicMPNN vs MSD)")
    ax.tick_params(axis="x", rotation=20)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    p = os.path.join(plots_dir, "method_win_rate_per_metric.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def plot_holo_apo_scatter(data, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    panels = [("af2_holo_plddt", "af2_apo_plddt", "pLDDT", "higher", (0.2, 0.95)),
              ("af2_holo_i_pae", "af2_apo_i_pae", "i_pae", "lower", (0.1, 0.9))]
    any_made = False
    for ax, (hx, ay, lab, direction, lim) in zip(axes, panels):
        for gate_key, bb_key, name, col in [("dmp_gate", None, "DynamicMPNN", DMP),
                                            ("msd_gate", None, "MSD", MSD)]:
            df = data[gate_key]; bb = _bb(df)
            if df is None or bb is None or hx not in df.columns or ay not in df.columns:
                continue
            hs = _best_per_backbone(df, bb, hx, direction)
            aps = _best_per_backbone(df, bb, ay, direction)
            j = pd.concat([hs.rename("h"), aps.rename("a")], axis=1).dropna()
            if j.empty:
                continue
            any_made = True
            ax.scatter(j["h"], j["a"], s=22, alpha=0.7, color=col, label=name, edgecolor="none")
        ax.plot(lim, lim, ls="--", color=INK, lw=1)
        _style(ax)
        ax.set(xlabel=f"holo {lab}", ylabel=f"apo {lab}",
               title=f"holo vs apo {lab}  (per backbone, best)")
        ax.legend(fontsize=8, frameon=False)
    if not any_made:
        plt.close(fig); return None
    fig.suptitle("Where the switch is limited: apo is the weak axis",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(plots_dir, "per_state_holo_apo_scatter.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


def _metric_series(df, col):
    """Return a metric column, computing af2_switch_plddt (harmonic mean of the
    two states' pLDDT) for a table that does not store it, such as the null."""
    if df is None:
        return None
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").dropna()
    if col == "af2_switch_plddt" and {"af2_holo_plddt", "af2_apo_plddt"} <= set(df.columns):
        h = pd.to_numeric(df["af2_holo_plddt"], errors="coerce")
        a = pd.to_numeric(df["af2_apo_plddt"], errors="coerce")
        return (2 * h * a / (h + a)).dropna()
    return None


def plot_real_vs_null(data, plots_dir):
    specs = [("af2_switch_plddt", "switch-pLDDT", "switch_plddt_real_vs_null.png",
              "Switch-pLDDT: real designs vs composition scramble"),
             ("af2_apo_i_pae", "apo i_pae", "apo_ipae_real_vs_null.png",
              "apo i_pae: the metric that fails the null (real is NOT better than scramble)")]
    out = []
    for col, lab, fname, title in specs:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharex=True, sharey=True)
        made = False
        for ax, (gk, nk, name) in zip(axes, [("dmp_gate", "dmp_null", "DynamicMPNN"),
                                             ("msd_gate", "msd_null", "ProteinMPNN-MSD")]):
            rv = _metric_series(data[gk], col)
            if rv is None or rv.empty:
                ax.axis("off"); continue
            made = True
            ax.hist(rv, bins=26, color=REAL, alpha=0.72, density=True, label="real", edgecolor="none")
            ax.axvline(rv.median(), color=REAL, lw=1.9)
            nv = _metric_series(data[nk], col)
            note = f"real median   {rv.median():.2f}"
            if nv is not None and not nv.empty:
                ax.hist(nv, bins=26, color=NULLC, alpha=0.55, density=True, label="scramble null", edgecolor="none")
                ax.axvline(nv.median(), color="#4d525a", lw=1.9, ls="--")
                note += f"\nnull median   {nv.median():.2f}"
            _style(ax)
            ax.set(title=name, xlabel=lab)
            ax.legend(fontsize=8, frameon=False, loc="upper left")
            ax.text(0.97, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="#f3f5f8", ec="none"))
        axes[0].set_ylabel("density")
        if not made:
            plt.close(fig); continue
        fig.suptitle(title, fontsize=10.5, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = os.path.join(plots_dir, fname)
        fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
        out.append(p)
    return out


def plot_stage_runtime(data, plots_dir):
    f = data["funnel"]
    if f is None or "timestamp" not in f.columns or "step" not in f.columns:
        return None
    f = f.copy()
    f["t"] = pd.to_datetime(f["timestamp"], errors="coerce")
    f = f.dropna(subset=["t"]).reset_index(drop=True)
    if len(f) < 3:
        return None
    f["dur_min"] = f["t"].diff().dt.total_seconds().div(60)
    f = f.dropna(subset=["dur_min"])
    f = f[f["dur_min"] > 0.05]
    if f.empty:
        return None
    # colour the two design/scoring arms distinctly
    def colour(step):
        s = str(step)
        if "dynamicmpnn" in s: return DMP
        if "msd" in s: return MSD
        if "boltz" in s or "s6" in s or "s7" in s: return "#b08a2e"
        if "af2" in s or "designab" in s: return "#4a9d8f"
        if "rfd3" in s or "geometry" in s: return "#556070"
        return "#9aa0aa"
    fig, ax = plt.subplots(figsize=(8.6, max(4, 0.32 * len(f))))
    ax.barh(f["step"], f["dur_min"], color=[colour(s) for s in f["step"]], alpha=0.9)
    ax.invert_yaxis()
    _style(ax); ax.grid(axis="x", color="#e6e8ec")
    ax.set(xlabel="wall-clock minutes (step-to-step)",
           title="Pipeline stage runtime (this run)")
    ax.tick_params(labelsize=6.5)
    p = os.path.join(plots_dir, "stage_runtime.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


def plot_best_of_n(data, plots_dir):
    """Sample-efficiency: mean best switch-pLDDT vs # sequences considered per
    backbone, averaged over random orderings — the value of extra sampling."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    made = False
    rng = np.random.default_rng(0)
    for gate_key, name, col in [("dmp_gate", "DynamicMPNN", DMP), ("msd_gate", "ProteinMPNN-MSD", MSD)]:
        df = data[gate_key]; bb = _bb(df)
        if df is None or bb is None or "af2_switch_plddt" not in df.columns:
            continue
        groups = [pd.to_numeric(g["af2_switch_plddt"], errors="coerce").dropna().values
                  for _, g in df.groupby(bb)]
        groups = [g for g in groups if len(g) >= 2]
        if not groups:
            continue
        nmax = min(min(len(g) for g in groups), 32)
        curves = []
        for g in groups:
            draws = []
            for _ in range(40):
                order = rng.permutation(len(g))[:nmax]
                draws.append(np.maximum.accumulate(g[order]))
            curves.append(np.mean(draws, axis=0))
        curves = np.array(curves)              # backbones x nmax
        mean = curves.mean(0); sem = curves.std(0) / np.sqrt(len(curves))
        xs = np.arange(1, nmax + 1)
        ax.plot(xs, mean, color=col, lw=2, label=name)
        ax.fill_between(xs, mean - sem, mean + sem, color=col, alpha=0.18)
        made = True
    if not made:
        plt.close(fig); return None
    _style(ax)
    ax.set(xlabel="sequences sampled per backbone (N)",
           ylabel="mean best-of-N switch-pLDDT",
           title="Sample efficiency: value of additional sequences per backbone")
    ax.legend(fontsize=8, frameon=False)
    p = os.path.join(plots_dir, "method_sample_efficiency.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def plot_conformational_change(data, plots_dir):
    """Achieved binder conformational change (holo->apo Ca-RMSD) of the final
    designs, and whether more change trades off against switch-pLDDT."""
    df = data["final"]
    if df is None or "binder_ca_rmsd" not in df.columns:
        return None
    r = pd.to_numeric(df["binder_ca_rmsd"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].hist(r.dropna(), bins=20, color="#556070", alpha=0.85, edgecolor="white")
    axes[0].axvline(r.median(), color=MSD, lw=1.8, label=f"median {r.median():.1f} A")
    _style(axes[0]); axes[0].legend(fontsize=8, frameon=False)
    axes[0].set(xlabel="binder Ca-RMSD holo vs apo (A)", ylabel="designs",
                title="Achieved conformational change")
    if "af2_switch_plddt" in df.columns:
        s = pd.to_numeric(df["af2_switch_plddt"], errors="coerce")
        j = pd.concat([r.rename("r"), s.rename("s")], axis=1).dropna()
        axes[1].scatter(j["r"], j["s"], s=22, alpha=0.7, color=DMP, edgecolor="none")
        _style(axes[1])
        axes[1].set(xlabel="binder Ca-RMSD holo vs apo (A)", ylabel="switch-pLDDT",
                    title="More change vs designability")
        if len(j) > 3:
            rr = np.corrcoef(j["r"], j["s"])[0, 1]
            axes[1].text(0.97, 0.95, f"r = {rr:+.2f}", transform=axes[1].transAxes,
                         ha="right", va="top", fontsize=9,
                         bbox=dict(boxstyle="round", fc="#eef1f5", ec="none"))
    else:
        axes[1].axis("off")
    fig.tight_layout()
    p = os.path.join(plots_dir, "conformational_change.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


def plot_predictor_agreement(data, plots_dir):
    """AF2 vs Boltz per design, per state — do the two predictors agree?"""
    df = data["final"]
    if df is None:
        return None
    panels = [("af2_holo_i_ptm", "s6a_boltz_holo_iptm", "holo (PD-L1)"),
              ("af2_apo_i_ptm", "s6b_boltz_apo_iptm", "apo (PCNA)")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharex=True, sharey=True)
    made = False
    for ax, (ax2c, bzc, lab) in zip(axes, panels):
        if ax2c not in df.columns or bzc not in df.columns:
            ax.axis("off"); continue
        j = pd.concat([pd.to_numeric(df[ax2c], errors="coerce").rename("a"),
                       pd.to_numeric(df[bzc], errors="coerce").rename("b")], axis=1).dropna()
        if j.empty:
            ax.axis("off"); continue
        made = True
        ax.scatter(j["a"], j["b"], s=24, alpha=0.7, color="#4a9d8f", edgecolor="none")
        ax.plot([0, 1], [0, 1], ls="--", color=INK, lw=1)
        _style(ax); ax.set(xlabel="AF2 interface pTM", ylabel="Boltz ipTM", title=lab)
        if len(j) > 3:
            rr = np.corrcoef(j["a"], j["b"])[0, 1]
            ax.text(0.05, 0.95, f"r = {rr:+.2f}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, bbox=dict(boxstyle="round", fc="#eef1f5", ec="none"))
    if not made:
        plt.close(fig); return None
    fig.suptitle("Predictor agreement: AF2 vs Boltz (per design)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(plots_dir, "predictor_agreement.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


_AA = "ACDEFGHIKLMNPQRSTVWY"


def _seqs(df, col):
    if df is None or col not in df.columns:
        return []
    out = []
    for s in df[col].dropna().astype(str):
        s = s.split(":")[0].strip().upper()
        if s and set(s) <= set(_AA + "X"):
            out.append(s)
    return out


def plot_sequence_properties(data, plots_dir):
    """AA composition and within-method sequence diversity for the two methods."""
    dseq = _seqs(data["dmp_gate"], "s5_dynamicmpnn_sequence")
    mseq = _seqs(data["msd_gate"], "_msd_seq")
    if not dseq or not mseq:
        return None

    def comp(seqs):
        from collections import Counter
        c = Counter("".join(seqs)); tot = sum(c.values()) or 1
        return np.array([c.get(a, 0) / tot for a in _AA])

    def mean_pairwise_identity(seqs, rng, n=1500):
        seqs = [s for s in seqs if len(s) > 0]
        if len(seqs) < 2:
            return np.nan
        ids = []
        for _ in range(n):
            a, b = rng.integers(0, len(seqs), 2)
            if a == b:
                continue
            x, y = seqs[a], seqs[b]; L = min(len(x), len(y))
            if L == 0:
                continue
            ids.append(sum(1 for i in range(L) if x[i] == y[i]) / L)
        return float(np.mean(ids)) if ids else np.nan

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(_AA)); w = 0.4
    axes[0].bar(x - w/2, comp(dseq), w, color=DMP, label="DynamicMPNN")
    axes[0].bar(x + w/2, comp(mseq), w, color=MSD, label="ProteinMPNN-MSD")
    axes[0].set_xticks(x); axes[0].set_xticklabels(list(_AA), fontsize=8)
    _style(axes[0]); axes[0].legend(fontsize=8, frameon=False)
    axes[0].set(ylabel="fraction", title="Amino-acid composition")
    di = mean_pairwise_identity(dseq, rng); mi = mean_pairwise_identity(mseq, rng)
    axes[1].bar(["DynamicMPNN", "MSD"], [di, mi], color=[DMP, MSD], alpha=0.9)
    _style(axes[1]); axes[1].set(ylim=(0, 1), ylabel="mean pairwise identity",
                                 title="Sequence diversity\n(lower = more diverse)")
    for i, v in enumerate([di, mi]):
        if v == v:
            axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    p = os.path.join(plots_dir, "sequence_properties.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


def plot_boltz_metric_violins(data, plots_dir):
    """Orthogonal predictor view: Boltz per-state ipTM, DynamicMPNN vs MSD."""
    dmp, msd = data["final"], data["msd_final"]
    specs = [("s6a_boltz_holo_iptm", "s7a_msd_boltz_holo_iptm", "holo ipTM"),
             ("s6b_boltz_apo_iptm", "s7b_msd_boltz_apo_iptm", "apo ipTM")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    made = False
    for ax, (dc, mc, lab) in zip(axes, specs):
        if dmp is None or msd is None or dc not in dmp.columns or mc not in msd.columns:
            ax.axis("off"); continue
        d = pd.to_numeric(dmp[dc], errors="coerce").dropna().values
        m = pd.to_numeric(msd[mc], errors="coerce").dropna().values
        if len(d) < 2 or len(m) < 2:
            ax.axis("off"); continue
        made = True
        parts = ax.violinplot([d, m], showmedians=True, widths=0.85)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([DMP, MSD][i]); pc.set_alpha(0.55)
        for k in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[k].set_color(INK)
        _style(ax); ax.set_xticks([1, 2]); ax.set_xticklabels(["DynamicMPNN", "MSD"], fontsize=8)
        ax.set_title(lab, fontsize=9)
    if not made:
        plt.close(fig); return None
    fig.suptitle("Orthogonal predictor (Boltz) interface confidence, per method",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(plots_dir, "boltz_metric_violins.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


PLOTS_DOC = """# Evaluation plots — what each one shows and why it matters

Generated automatically by the pipeline. Metrics: pLDDT and interface-pTM are
higher-is-better (0-1); interface PAE (i_pae) is lower-is-better; switch-pLDDT is
the harmonic mean of the two states' pLDDT (a switch is only as good as its weaker
state). "Null" everywhere = one composition-matched sequence scramble per real
design, scored identically — the baseline a real design must beat.

## Gates & null audits (is there signal at all?)
- **state_pair_geometry.png** — distributions of the geometry-gate metrics (binder
  Ca-RMSD, interface reuse, target-target clash). *Merit:* shows the conformational
  change and mutual-exclusion actually generated, before any sequence work.
- **af2_paired_null_audit.png** — per-metric AUC + backbone-paired win-rate (with
  95% CI) of real vs scramble. *Merit:* THE stop/go gate — a metric at AUC~0.5 is
  worthless; this is where apo i_pae fails.
- **boltz_interface_null_audit.png** — the same audit for the orthogonal Boltz
  interface metric. *Merit:* keeps the diagnostic honest (only trusted if it beats
  its own null).

## DynamicMPNN vs ProteinMPNN-MSD (which method is better?)
- **af2_method_comparison.png** — per-backbone scatter of best switch-pLDDT.
  *Merit:* the headline equal-budget comparison; points above the diagonal = DMPNN wins.
- **method_delta_distribution.png** — histogram of (DMPNN - MSD) per backbone.
  *Merit:* one-number summary of the advantage and how often it holds.
- **method_win_rate_per_metric.png** — backbone-paired win rate for each metric.
  *Merit:* shows the win is broad across metrics, not one lucky axis.
- **method_metric_violins.png** — both methods' distributions for all 6 AF2 metrics.
  *Merit:* reveals where each wins/ties (apo i_pae is a tie = shared bottleneck).
- **boltz_method_comparison.png / boltz_metric_violins.png** — the same comparison
  on the orthogonal Boltz predictor. *Merit:* checks the result isn't an AF2 artifact.
- **af2_method_success_rates.png** — fraction of backbones that pass, per method,
  with Wilson CIs. *Merit:* backbone-level yield, the fair unit of comparison.
- **method_sample_efficiency.png** — best-of-N switch-pLDDT vs sequences sampled.
  *Merit:* the compute view — which method gets more per sample, and where sampling
  saturates.
- **sequence_properties.png** — amino-acid composition + within-method diversity.
  *Merit:* explains *how* the methods differ (e.g. MSD's Ala/Gly low-complexity collapse).

## Binder / switch diagnostics (what did we actually make?)
- **per_state_holo_apo_scatter.png** — holo vs apo on pLDDT and i_pae, per method.
  *Merit:* localizes the failure — apo is the weak axis.
- **switch_plddt_real_vs_null.png** — real vs null switch-pLDDT, with medians.
  *Merit:* the fold signal that DOES clear the null.
- **apo_ipae_real_vs_null.png** — real vs null apo interface PAE, with medians.
  *Merit:* the metric that does NOT clear the null; the left tail here is the
  handful of genuinely good-apo designs worth pulling out (see query_designs.py).
- **conformational_change.png** — achieved holo->apo binder RMSD (+ vs switch-pLDDT).
  *Merit:* did designs actually switch, and does more motion cost designability?
- **predictor_agreement.png** — AF2 vs Boltz per design. *Merit:* consensus check;
  agreement is the strongest evidence given AF2 is both selector and anchor.
- **stage_runtime.png** — wall-clock per pipeline stage. *Merit:* where compute goes.
- **stage_funnel.png** — design/pair counts surviving each stage. *Merit:* the
  attrition overview — where the funnel narrows most.

## Structures
- **structure_snapshots/top_binders_montage.png** — the top-N binders, each in both
  bound conformations (PD-L1 | PCNA), binder coloured by pLDDT, target in grey.
  *Merit:* qualitative check that the top designs fold and dock in both states.

## Metric glossary (what each metric is and how it is computed)

All AF2 metrics come from AlphaFold2 initial-guess (colabdesign) run on the design
backbone with the sequence threaded on; colabdesign normalises them to 0-1.

- **pLDDT** — predicted Local Distance Difference Test, AF2's per-residue confidence
  in local structure (0-1; native AF2 reports 0-100). Reported per state as the mean
  over binder residues. *How:* AF2's pLDDT head. Higher = more confident fold.
- **Interface PAE (i_pae)** — Predicted Aligned Error averaged over the binder<->target
  residue pairs across the interface (0-1 here; x31 ~ Angstrom). *How:* the AF2 PAE
  matrix restricted to the inter-chain block, then averaged. Lower = better-defined
  interface. The load-bearing binding signal (Bennett initial-guess uses <10 A).
- **Interface pTM (i_ptm)** — predicted TM-score for the two-chain complex (0-1):
  confidence that binder and target are correctly docked relative to each other.
  *How:* AF2's pTM head restricted to inter-chain pairs.
- **switch-pLDDT** — harmonic mean of the two states' pLDDT: 2*h*a/(h+a). *Why:* a
  switch is only as good as its weaker state; the harmonic mean punishes imbalance.
- **AUC (common-language effect size)** — probability a real design beats its paired
  scramble: (#real>null + 0.5*#ties)/(n_real*n_null); equals the ROC AUC. 0.5 = no
  signal, 1 = perfect, <0.5 = worse than random.
- **Paired win-rate** — per backbone take the median real value and median null value;
  the win-rate is the fraction of backbones where real beats null. Reported with a
  bootstrap 95% CI (resampling backbones). Controls for backbone quality.
- **Wilson 95% CI** — confidence interval for a binomial rate (fraction of backbones
  that pass); accurate at small n, unlike the normal approximation.
- **binder Ca-RMSD (holo vs apo)** — RMSD between the binder's Ca atoms in the two
  states after superposing the two *targets* (Angstrom). The conformational-change
  magnitude; must be real (gate 1-8 A) for a true two-state object.
- **interface Jaccard** — |A intersect B| / |A union B| of the two states' interface-
  residue sets (0-1): symmetric overlap of the two binding surfaces.
- **interface reuse fraction** — fraction of one state's interface residues also used
  in the other state (0-1): the "shared surface" behind mutual exclusivity.
- **target-target clash pairs** — number of clashing atom pairs when both targets are
  placed on the shared binder surface (after superposition). Geometric proxy for
  mutual exclusion (the two targets cannot bind at once).
- **Boltz ipTM** — interface pTM (0-1) from Boltz-2, an AF3-style diffusion predictor
  independent of AF2. Used as an orthogonal diagnostic / consensus check.
- **Boltz interface diagnostic** — sqrt(ipTM * exp(-iPAE/10 A)) taken as a harmonic
  mean across states with a sqrt(min binder pLDDT) penalty. A single combined
  readout; never interpreted as affinity.
- **self-consistency scRMSD** — RMSD between the design's predicted structure and its
  design backbone (Angstrom): does the sequence actually encode the intended shape.
- **sequence pairwise identity** — mean fraction of identical positions over sampled
  sequence pairs within a method; a diversity proxy (lower = more diverse).
"""


def write_plots_readme(plots_dir):
    p = os.path.join(plots_dir, "PLOTS_README.md")
    with open(p, "w") as fh:
        fh.write(PLOTS_DOC)
    return p


def plot_stage_funnel(data, plots_dir):
    """Design/pair counts surviving each pipeline stage — the attrition funnel."""
    f = data["funnel"]
    if f is None or "step" not in f.columns or "n_designs" not in f.columns:
        return None
    f = f.copy()
    f["n"] = pd.to_numeric(f["n_designs"], errors="coerce")
    f = f.dropna(subset=["n"])
    if f.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.6, max(4, 0.30 * len(f))))
    ax.barh(f["step"], f["n"], color="#4a7cb5", alpha=0.9)
    ax.invert_yaxis()
    _style(ax); ax.grid(axis="x", color="#e6e8ec")
    for y, n in enumerate(f["n"]):
        ax.text(n, y, f" {int(n)}", va="center", fontsize=6.5)
    ax.set(xlabel="designs / pairs remaining", title="Pipeline attrition funnel (count per stage)")
    ax.tick_params(labelsize=6.5)
    p = os.path.join(plots_dir, "stage_funnel.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


def render_extended_plots(outputs_dir, plots_dir=None):
    plots_dir = plots_dir or os.path.join(outputs_dir, "evaluation_plots")
    os.makedirs(plots_dir, exist_ok=True)
    data = _load(outputs_dir)
    made = []
    for fn in (plot_delta_distribution, plot_metric_violins, plot_win_rate_per_metric,
               plot_holo_apo_scatter, plot_stage_runtime,
               plot_best_of_n, plot_conformational_change, plot_predictor_agreement,
               plot_sequence_properties, plot_boltz_metric_violins, plot_stage_funnel):
        try:
            p = fn(data, plots_dir)
            if p: made.append(os.path.basename(p))
        except Exception as e:  # never break evaluation over a plot
            print(f"  [extended-plots] {fn.__name__} skipped: {e}")
    try:
        for p in plot_real_vs_null(data, plots_dir) or []:
            made.append(os.path.basename(p))
    except Exception as e:
        print(f"  [extended-plots] real_vs_null skipped: {e}")
    try:
        write_plots_readme(plots_dir)
    except Exception as e:
        print(f"  [extended-plots] PLOTS_README skipped: {e}")
    return made


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    names = render_extended_plots(outdir)
    print("wrote:", ", ".join(names) if names else "(nothing — data missing)")
