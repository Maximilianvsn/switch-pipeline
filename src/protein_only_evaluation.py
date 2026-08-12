"""Evaluation and reporting for two-state design runs."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_AF2_NULL_METRICS = {
    "af2_holo_plddt", "af2_apo_plddt", "af2_holo_i_pae", "af2_apo_i_pae",
}


def _read(path: str):
    return pd.read_csv(path) if os.path.isfile(path) else None


def _run_settings(outputs_dir: str) -> dict:
    settings = {"min_null_auc": 0.70, "min_null_pairs": 20, "mode": "unknown"}
    path = os.path.join(outputs_dir, "run_provenance.json")
    if not os.path.isfile(path):
        return settings
    with open(path) as handle:
        provenance = json.load(handle)
    config = provenance.get("config", {})
    settings.update(config.get("evaluation", {}))
    settings["mode"] = provenance.get("mode", "unknown")
    # Smoke overrides control integration flow only. Reported results must use
    # the preregistered production evidence requirements from `evaluation`.
    return settings


def _geometry_report(outputs_dir: str, plots_dir: str) -> dict:
    frame = _read(os.path.join(outputs_dir, "s2_state_pair_geometry.csv"))
    if frame is None or frame.empty:
        return {"available": False, "n_pairs": 0, "n_pass": 0, "pass_rate": None}
    passed = frame.get("geometry_pass", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    summary = {
        "available": True,
        "n_pairs": int(len(frame)),
        "n_pass": int(passed.sum()),
        "pass_rate": float(passed.mean()),
        "median_binder_ca_rmsd": float(frame["binder_ca_rmsd"].median()),
        "median_interface_jaccard": float(frame["interface_jaccard"].median()),
        "median_interface_reuse_fraction": float(frame["interface_reuse_fraction"].median()),
        "median_target_target_clash_pairs": float(frame["target_target_clash_pairs"].median()),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(plots_dir, "state_pair_geometry_summary.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    color = np.where(passed, "#2a78d6", "#b7b7b2")
    axes[0].scatter(frame["binder_ca_rmsd"], frame["interface_reuse_fraction"], c=color, alpha=0.8)
    axes[0].set(xlabel="binder Cα RMSD (Å)", ylabel="directional interface reuse",
                title="Conformational change vs interface reuse")
    axes[1].hist(frame["interface_jaccard"].dropna(), bins=20, color="#2a78d6", alpha=0.85)
    axes[1].set(xlabel="interface Jaccard", ylabel="backbone pairs", title="Shared binder surface")
    axes[2].hist(np.log10(frame["target_target_clash_pairs"].clip(lower=0) + 1), bins=20,
                 color="#1baf7a", alpha=0.85)
    axes[2].set(xlabel="log10(target-target clash pairs + 1)", ylabel="backbone pairs",
                title="Simultaneous-binding incompatibility")
    fig.suptitle(f"Pre-sequence state-pair geometry: {passed.sum()}/{len(frame)} pass")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "state_pair_geometry.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return summary


def _null_report(outputs_dir: str, plots_dir: str) -> dict:
    settings = _run_settings(outputs_dir)
    min_auc = float(settings.get("min_null_auc", 0.70))
    min_pairs = int(settings.get("min_null_pairs", 20))
    min_win = 0.70
    source = os.path.join(outputs_dir, "af2_null_separation.csv")
    frame = _read(source)
    if frame is None or frame.empty:
        return {"available": False, "all_metrics_pass": False, "n_metrics": 0}
    required = {
        "metric", "auc", "paired_win_rate", "paired_win_rate_ci_low",
        "paired_win_rate_ci_high", "n_pairs",
    }
    if not required.issubset(frame.columns):
        return {"available": True, "all_metrics_pass": False, "n_metrics": int(len(frame))}
    frame = frame.copy()
    observed_metrics = set(frame.get("metric", pd.Series(dtype=str)).astype(str))
    complete_metric_set = (
        "metric" in frame
        and not frame["metric"].astype(str).duplicated().any()
        and observed_metrics == EXPECTED_AF2_NULL_METRICS
    )
    frame["passes"] = (
        (frame["auc"] >= min_auc)
        & (frame["paired_win_rate"] >= min_win)
        & (frame["paired_win_rate_ci_low"] > 0.50)
        & (frame["n_pairs"] >= min_pairs)
    )
    frame.to_csv(os.path.join(plots_dir, "af2_paired_null_audit.csv"), index=False)

    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x, frame["auc"], color=np.where(frame["passes"], "#2a78d6", "#e34948"))
    axes[0].axhline(min_auc, color="#55554f", linestyle="--")
    axes[0].set(xticks=x, xticklabels=frame["metric"], ylim=(0, 1), ylabel="pooled AUC",
                title="Real sequence vs paired scramble")
    axes[0].tick_params(axis="x", rotation=20)
    lower = frame["paired_win_rate"] - frame["paired_win_rate_ci_low"]
    upper = frame["paired_win_rate_ci_high"] - frame["paired_win_rate"]
    axes[1].errorbar(x, frame["paired_win_rate"], yerr=[lower, upper], fmt="o", capsize=4,
                     color="#2a78d6")
    axes[1].axhline(0.50, color="#55554f", linestyle="--", label="chance")
    axes[1].axhline(min_win, color="#8a8a86", linestyle=":", label="target")
    axes[1].set(xticks=x, xticklabels=frame["metric"], ylim=(0, 1), ylabel="backbone-paired win rate",
                title="Bootstrap 95% confidence intervals")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False)
    fig.suptitle("AF2 stop/go audit: every metric must pass")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "af2_paired_null_audit.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "available": True,
        "all_metrics_pass": bool(complete_metric_set and frame["passes"].all()),
        "n_metrics": int(len(frame)),
        "expected_metrics": sorted(EXPECTED_AF2_NULL_METRICS),
        "missing_metrics": sorted(EXPECTED_AF2_NULL_METRICS - observed_metrics),
        "unexpected_metrics": sorted(observed_metrics - EXPECTED_AF2_NULL_METRICS),
        "minimum_auc": float(frame["auc"].min()),
        "minimum_paired_win_rate": float(frame["paired_win_rate"].min()),
        "minimum_bootstrap_ci_low": float(frame["paired_win_rate_ci_low"].min()),
        "minimum_observed_pairs": int(frame["n_pairs"].min()),
        "required_minimum_auc": min_auc,
        "required_minimum_pairs": min_pairs,
        "required_minimum_paired_win_rate": min_win,
    }


def _method_report(outputs_dir: str, plots_dir: str) -> dict:
    summary = _read(os.path.join(outputs_dir, "method_comparison_summary.csv"))
    paired = _read(os.path.join(outputs_dir, "method_comparison_paired.csv"))
    if summary is None or paired is None or paired.empty:
        return {"available": False, "n_paired_backbones": 0}
    summary.to_csv(os.path.join(plots_dir, "method_comparison_summary.csv"), index=False)
    paired.to_csv(os.path.join(plots_dir, "method_comparison_paired.csv"), index=False)
    x = paired["best_af2_switch_plddt_msd"]
    y = paired["best_af2_switch_plddt_dynamicmpnn"]
    limit = max(float(x.max()), float(y.max()), 0.01) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c=np.where(y > x, "#2a78d6", "#1baf7a"), alpha=0.85)
    ax.plot([0, limit], [0, limit], color="#777770", linestyle="--")
    ax.set(xlim=(0, limit), ylim=(0, limit),
           xlabel="ProteinMPNN-MSD best AF2 harmonic pLDDT",
           ylabel="DynamicMPNN best AF2 harmonic pLDDT",
           title=f"Equal-budget comparison by backbone (n={len(paired)})")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "af2_method_comparison.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    n = pd.to_numeric(summary["n_backbones"], errors="coerce").fillna(0).to_numpy(dtype=float)
    successes = pd.to_numeric(summary["backbones_with_pass"], errors="coerce").fillna(0).to_numpy(dtype=float)
    rate = np.divide(successes, n, out=np.zeros_like(successes), where=n > 0)
    z = 1.96
    denom = 1 + z * z / np.maximum(n, 1)
    center = (rate + z * z / (2 * np.maximum(n, 1))) / denom
    half = z * np.sqrt(rate * (1 - rate) / np.maximum(n, 1) + z * z / (4 * np.maximum(n, 1) ** 2)) / denom
    fig, ax = plt.subplots(figsize=(7, 4.8))
    xpos = np.arange(len(summary))
    colors = ["#2a78d6" if m == "dynamicmpnn" else "#1baf7a" for m in summary["method"]]
    ax.bar(xpos, rate, color=colors, alpha=0.8)
    ax.errorbar(xpos, center, yerr=half, fmt="none", ecolor="#333330", capsize=5)
    ax.set(xticks=xpos, xticklabels=summary["method"].map({"dynamicmpnn": "DynamicMPNN", "proteinmpnn_msd": "ProteinMPNN-MSD"}),
           ylim=(0, 1), ylabel="backbones with >=1 AF2 pass",
           title="Equal-budget success rate (Wilson 95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "af2_method_success_rates.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    delta = paired["delta_best_af2_switch_plddt"].dropna()
    return {
        "available": True,
        "n_paired_backbones": int(len(paired)),
        "dynamicmpnn_win_rate": float((delta > 0).mean()) if len(delta) else None,
        "median_delta_best_af2_switch_plddt": float(delta.median()) if len(delta) else None,
    }


def _boltz_report(outputs_dir: str, plots_dir: str) -> dict:
    all_scores = _read(os.path.join(outputs_dir, "boltz_interface_all.csv"))
    paired = _read(os.path.join(outputs_dir, "boltz_interface_paired.csv"))
    audit = _read(os.path.join(outputs_dir, "boltz_interface_null_audit.csv"))
    status_path = os.path.join(outputs_dir, "boltz_interface_status.json")
    if all_scores is None or paired is None or not os.path.isfile(status_path):
        return {"available": False, "validated_against_null": False, "n_paired_backbones": 0}
    with open(status_path) as handle:
        status = json.load(handle)

    labels = {"dynamicmpnn": "DynamicMPNN", "proteinmpnn_msd": "ProteinMPNN-MSD"}
    methods = [m for m in labels if m in set(all_scores["method"])]
    score_data = [all_scores.loc[all_scores["method"] == m, "boltz_final_interface_score"].dropna()
                  for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if score_data:
        boxes = axes[0].boxplot(score_data, tick_labels=[labels[m] for m in methods], patch_artist=True)
        for patch, color in zip(boxes["boxes"], ["#2a78d6", "#1baf7a"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    axes[0].set(ylabel="final Boltz interface confidence",
                title="Sequence-level score distributions")
    x = paired.get("best_boltz_final_interface_score_msd", pd.Series(dtype=float))
    y = paired.get("best_boltz_final_interface_score_dynamicmpnn", pd.Series(dtype=float))
    valid = x.notna() & y.notna()
    if valid.any():
        low = min(float(x[valid].min()), float(y[valid].min()), 0.0)
        high = max(float(x[valid].max()), float(y[valid].max()), 0.01) * 1.05
        axes[1].scatter(x[valid], y[valid], c=np.where(y[valid] > x[valid], "#2a78d6", "#1baf7a"), alpha=0.85)
        axes[1].plot([low, high], [low, high], color="#777770", linestyle="--")
        axes[1].set(xlim=(low, high), ylim=(low, high))
    axes[1].set(xlabel="ProteinMPNN-MSD best score", ylabel="DynamicMPNN best score",
                title=f"Paired by backbone (n={int(valid.sum())})")
    fig.suptitle("Final Boltz interface confidence (diagnostic, not affinity)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "boltz_method_comparison.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    if audit is not None and not audit.empty:
        xloc = np.arange(len(audit))
        passes = ((audit["auc"] >= 0.70) & (audit["paired_win_rate"] >= 0.70)
                  & (audit["paired_win_rate_ci_low"] > 0.50) & (audit["n_pairs"] >= 20))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        axes[0].bar(xloc, audit["auc"], color=np.where(passes, "#2a78d6", "#e34948"))
        axes[0].axhline(0.70, color="#55554f", linestyle="--")
        axes[0].set(xticks=xloc, xticklabels=audit["metric"], ylim=(0, 1),
                    ylabel="pooled AUC", title="Real vs paired scramble")
        lower = audit["paired_win_rate"] - audit["paired_win_rate_ci_low"]
        upper = audit["paired_win_rate_ci_high"] - audit["paired_win_rate"]
        axes[1].errorbar(xloc, audit["paired_win_rate"], yerr=[lower, upper], fmt="o", capsize=4)
        axes[1].axhline(0.50, color="#55554f", linestyle="--")
        axes[1].axhline(0.70, color="#8a8a86", linestyle=":")
        axes[1].set(xticks=xloc, xticklabels=audit["metric"], ylim=(0, 1),
                    ylabel="paired win rate", title="Backbone bootstrap 95% CI")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25)
        fig.suptitle("Boltz interface metric null audit")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "boltz_interface_null_audit.png"), dpi=170, bbox_inches="tight")
        plt.close(fig)

    return {
        "available": True,
        "validated_against_null": bool(status.get("validated_against_null", False)),
        "n_paired_backbones": int(status.get("n_paired_backbones", len(paired))),
        "interpretation": status.get("interpretation", "predictor-confidence diagnostic"),
    }


def _render_structure_snapshots(outputs_dir: str, n: int = 5) -> None:
    """PyMOL snapshots of the top-N binders in both conformations. Best-effort:
    needs a PyMOL env (rendering runs out-of-process) and is fully non-fatal."""
    import shutil
    import subprocess
    import sys
    if not os.path.isfile(os.path.join(outputs_dir, "final_all_ranked.csv")):
        return
    pymol = next((p for p in (
        os.path.expanduser("~/.conda/envs/mdplot/bin/pymol"), shutil.which("pymol"))
        if p and os.path.exists(p)), None)
    if not pymol:
        print("  [eval] structure snapshots skipped: no PyMOL env found")
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for arm, sub, tag, ranked in (
        ("dynamicmpnn", "structure_snapshots", "DynamicMPNN", "final_all_ranked.csv"),
        ("msd", "structure_snapshots_msd", "ProteinMPNN-MSD", "msd_final_all_ranked.csv"),
    ):
        if not os.path.isfile(os.path.join(outputs_dir, ranked)):
            continue
        try:
            subprocess.run([pymol, "-cq", os.path.join(here, "render_top_binders.py"),
                            "--", "--run", outputs_dir, "--arm", arm, "--n", str(n)],
                           check=True, timeout=1200, capture_output=True, text=True)
            subprocess.run([sys.executable, os.path.join(here, "montage_snapshots.py"),
                            outputs_dir, sub, tag],
                           check=True, timeout=300, capture_output=True, text=True)
            print(f"  [eval] structure snapshots ({tag}): top {n} rendered (both states) + montage")
        except Exception as exc:
            print(f"  [eval] structure snapshots ({tag}) skipped: {exc}")


def run_protein_only_evaluation(outputs_dir: str, plots_dir: str | None = None) -> str:
    plots_dir = plots_dir or os.path.join(outputs_dir, "evaluation_plots")
    os.makedirs(plots_dir, exist_ok=True)
    geometry = _geometry_report(outputs_dir, plots_dir)
    null = _null_report(outputs_dir, plots_dir)
    method = _method_report(outputs_dir, plots_dir)
    boltz = _boltz_report(outputs_dir, plots_dir)
    # Extended DynamicMPNN-vs-MSD + per-state/per-metric diagnostic figures.
    # Non-fatal: a plotting failure must never sink the evaluation.
    try:
        from method_comparison_plots import render_extended_plots
        extended = render_extended_plots(outputs_dir, plots_dir)
        print(f"  [eval] extended comparison plots: {len(extended)} figure(s)")
    except Exception as exc:
        print(f"  [eval] extended comparison plots skipped: {exc}")
    _render_structure_snapshots(outputs_dir)
    tiers = _read(os.path.join(outputs_dir, "s5_5_af2_gate_all.csv"))
    final_all = _read(os.path.join(outputs_dir, "final_all_ranked.csv"))
    final_supported = _read(os.path.join(outputs_dir, "final_relaxed.csv"))
    final_consensus = _read(os.path.join(outputs_dir, "final_consensus.csv"))
    tier_counts = ({str(k): int(v) for k, v in tiers["af2_tier"].value_counts().items()}
                   if tiers is not None and "af2_tier" in tiers else {})

    n_null_supported = 0
    if tiers is not None and "af2_tier" in tiers:
        supported = tiers["af2_tier"].fillna("fail").ne("fail")
        if "af2_null_discriminates" in tiers:
            supported &= tiers["af2_null_discriminates"].fillna(False).astype(bool)
        n_null_supported = int(supported.sum())
    n_final_supported = int(len(final_supported)) if final_supported is not None else 0
    n_final_consensus = int(len(final_consensus)) if final_consensus is not None else 0
    minimum_method_pairs = int(null.get("required_minimum_pairs", 20))

    required_artifacts_available = bool(
        geometry.get("available", False)
        and null.get("available", False)
        and method.get("available", False)
        and boltz.get("available", False)
        and tiers is not None
        and final_all is not None
    )
    evidence_ready = bool(
        required_artifacts_available
        and _run_settings(outputs_dir).get("mode") == "production"
        and geometry.get("n_pass", 0) > 0
        and null.get("all_metrics_pass", False)
        and n_null_supported > 0
        and method.get("n_paired_backbones", 0) >= minimum_method_pairs
        and boltz.get("validated_against_null", False)
        and n_final_supported > 0
    )
    allowed_claim = (
        "At least one de novo sequence is computationally compatible with two mutually exclusive "
        "target-bound conformations; DynamicMPNN is compared with an equal-budget MSD baseline."
        if evidence_ready else
        "No positive two-state claim is supported by this run; treat its outputs as integration "
        "diagnostics until every preregistered evidence condition passes."
    )
    audit = {
        "pipeline_mode": "protein_only_two_state",
        "pipeline_implementation_ready": required_artifacts_available,
        "run_evidence_ready_for_thesis_claim": evidence_ready,
        "geometry": geometry,
        "paired_null_validation": null,
        "equal_budget_method_comparison": method,
        "boltz_interface_diagnostic": boltz,
        "af2_tier_counts": tier_counts,
        "n_null_supported_af2_sequences": n_null_supported,
        "n_final_supported_sequences": n_final_supported,
        "n_final_consensus_sequences": n_final_consensus,
        "allowed_claim": allowed_claim,
        "disallowed_claim": (
            "Thermodynamic bistability, state populations, binding affinity, or switching kinetics "
            "without experimental or free-energy validation."
        ),
    }
    with open(os.path.join(plots_dir, "thesis_readiness.json"), "w") as handle:
        json.dump(audit, handle, indent=2)

    lines = [
        "PROTEIN-ONLY TWO-STATE RUN AUDIT",
        "",
        f"Required stage artifacts available: {required_artifacts_available}",
        f"This run provides complete thesis evidence: {evidence_ready}",
        "",
        f"Geometry pairs: {geometry.get('n_pass', 0)}/{geometry.get('n_pairs', 0)} pass",
        f"All four AF2 metrics pass paired-null stop/go: {null.get('all_metrics_pass', False)}",
        f"AF2 sequences supported by a discriminating null: {n_null_supported}",
        f"Equal-budget method comparison pairs: {method.get('n_paired_backbones', 0)}",
        f"Boltz interface diagnostic available: {boltz.get('available', False)}",
        f"Boltz metric validated against paired null: {boltz.get('validated_against_null', False)}",
        f"Final supported/consensus sequences: {n_final_supported}/{n_final_consensus}",
        f"AF2 tiers: {tier_counts}",
        "",
        "Allowed claim:", audit["allowed_claim"],
        "",
        "Do not claim:", audit["disallowed_claim"],
    ]
    with open(os.path.join(plots_dir, "summary_protein_only.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return plots_dir
