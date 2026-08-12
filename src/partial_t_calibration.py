"""Comparison of RFdiffusion3 partial-diffusion noise levels."""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def summarize_partial_t(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize production geometry-gate performance for each noise level."""
    required = {
        "partial_t", "geometry_pass", "binder_ca_rmsd", "interface_jaccard",
        "interface_reuse_fraction", "target_target_clash_pairs",
        "s1_rfd3_holo_description",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing partial_t calibration columns: {sorted(missing)}")

    rows = []
    for partial_t, group in frame.groupby("partial_t", sort=True):
        passed = group["geometry_pass"].fillna(False).astype(bool)
        passing = group.loc[passed]
        n_pairs = int(len(group))
        n_pass = int(passed.sum())
        backbone_pass = group.assign(_pass=passed).groupby(
            "s1_rfd3_holo_description"
        )["_pass"].any()
        n_backbones = int(len(backbone_pass))
        n_backbones_with_pass = int(backbone_pass.sum())
        pair_ci_low, pair_ci_high = _wilson_interval(n_pass, n_pairs)
        backbone_ci_low, backbone_ci_high = _wilson_interval(
            n_backbones_with_pass, n_backbones
        )
        rows.append({
            "partial_t_angstrom": float(partial_t),
            "n_state1_backbones": n_backbones,
            "n_backbones_with_geometry_pass": n_backbones_with_pass,
            "backbone_success_rate": (
                float(n_backbones_with_pass / n_backbones) if n_backbones else float("nan")
            ),
            "backbone_success_rate_ci_low": backbone_ci_low,
            "backbone_success_rate_ci_high": backbone_ci_high,
            "n_state2_pairs": n_pairs,
            "n_geometry_pass": n_pass,
            "geometry_pass_rate": float(n_pass / n_pairs) if n_pairs else float("nan"),
            "geometry_pass_rate_ci_low": pair_ci_low,
            "geometry_pass_rate_ci_high": pair_ci_high,
            "median_rmsd_all_angstrom": float(group["binder_ca_rmsd"].median()),
            "q25_rmsd_all_angstrom": float(group["binder_ca_rmsd"].quantile(0.25)),
            "q75_rmsd_all_angstrom": float(group["binder_ca_rmsd"].quantile(0.75)),
            "median_rmsd_passing_angstrom": (
                float(passing["binder_ca_rmsd"].median()) if n_pass else float("nan")
            ),
            "median_interface_jaccard": float(group["interface_jaccard"].median()),
            "median_interface_reuse_fraction": float(group["interface_reuse_fraction"].median()),
            "median_target_target_clash_pairs": float(group["target_target_clash_pairs"].median()),
        })
    return pd.DataFrame(rows).sort_values("partial_t_angstrom").reset_index(drop=True)


def choose_partial_t(summary: pd.DataFrame, target_rmsd: float = 3.0) -> dict:
    """Apply a predeclared, auditable selection rule to the calibration table.

    Primary criterion is the lower Wilson confidence bound of backbone-level
    success. Values within two percentage points are tied; among ties, prefer
    the passing-pair median RMSD closest to the configured target, then less noise.
    """
    eligible = summary[
        (summary["n_geometry_pass"] >= 3)
        & (summary["n_backbones_with_geometry_pass"] >= 2)
        & summary["backbone_success_rate_ci_low"].notna()
        & summary["median_rmsd_passing_angstrom"].notna()
    ].copy()
    rule = (
        "Require at least three production-geometry passes spanning at least two "
        "state-1 backbones; maximize the lower bound of the Wilson 95% confidence "
        "interval for backbone success (at least one passing trajectory). Treat values "
        "within 0.02 as tied, then choose the passing-pair median binder RMSD "
        f"closest to {target_rmsd:g} A, then the smaller partial_t."
    )
    if eligible.empty:
        return {
            "recommended_partial_t_angstrom": None,
            "selection_rule": rule,
            "reason": "No noise level produced three passes spanning at least two backbones.",
        }

    best_lower_bound = float(eligible["backbone_success_rate_ci_low"].max())
    tied = eligible[
        eligible["backbone_success_rate_ci_low"] >= best_lower_bound - 0.02
    ].copy()
    tied["_target_rmsd_distance"] = (
        tied["median_rmsd_passing_angstrom"] - float(target_rmsd)
    ).abs()
    chosen = tied.sort_values(
        ["_target_rmsd_distance", "partial_t_angstrom"], ascending=[True, True]
    ).iloc[0]
    return {
        "recommended_partial_t_angstrom": float(chosen["partial_t_angstrom"]),
        "selection_rule": rule,
        "reason": (
            f"Selected partial_t={chosen['partial_t_angstrom']:g} A: "
            f"{chosen['n_backbones_with_geometry_pass']:.0f}/"
            f"{chosen['n_state1_backbones']:.0f} backbones produced at least one "
            f"production-gate pass ({chosen['backbone_success_rate']:.1%}; Wilson "
            f"95% CI {chosen['backbone_success_rate_ci_low']:.1%}-"
            f"{chosen['backbone_success_rate_ci_high']:.1%}). Pair-level yield was "
            f"{chosen['n_geometry_pass']:.0f}/{chosen['n_state2_pairs']:.0f} "
            f"({chosen['geometry_pass_rate']:.1%}), with median passing "
            f"binder RMSD {chosen['median_rmsd_passing_angstrom']:.2f} A."
        ),
    }


def _write_plot(frame: pd.DataFrame, summary: pd.DataFrame, output_path: str, geometry_cfg: dict):
    levels = summary["partial_t_angstrom"].tolist()
    labels = [f"{value:g}" for value in levels]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(levels)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    rates = summary["backbone_success_rate"].to_numpy(float)
    lower = rates - summary["backbone_success_rate_ci_low"].to_numpy(float)
    upper = summary["backbone_success_rate_ci_high"].to_numpy(float) - rates
    x = np.arange(len(levels))
    axes[0, 0].bar(x, rates, color=colors)
    axes[0, 0].errorbar(x, rates, yerr=[lower, upper], fmt="none", color="#22221f", capsize=5)
    for xpos, (_, row) in enumerate(summary.iterrows()):
        axes[0, 0].text(
            xpos, min(0.97, float(row["backbone_success_rate"]) + 0.04),
            f"{int(row['n_backbones_with_geometry_pass'])}/{int(row['n_state1_backbones'])}",
            ha="center", va="bottom", fontsize=9,
        )
    axes[0, 0].set(xticks=x, xticklabels=labels, ylim=(0, 1.08), xlabel="partial_t (Å)",
                   ylabel="backbones with ≥1 geometry pass",
                   title="Backbone-level success (Wilson 95% CI)")

    rmsd_groups = [
        frame.loc[frame["partial_t"].eq(level), "binder_ca_rmsd"].dropna().to_numpy()
        for level in levels
    ]
    box = axes[0, 1].boxplot(rmsd_groups, tick_labels=labels, patch_artist=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    min_rmsd = float(geometry_cfg.get("min_binder_ca_rmsd", 1.0))
    max_rmsd = float(geometry_cfg.get("max_binder_ca_rmsd", 8.0))
    axes[0, 1].axhspan(min_rmsd, max_rmsd, color="#1baf7a", alpha=0.10)
    axes[0, 1].axhline(min_rmsd, color="#55554f", linestyle="--")
    axes[0, 1].axhline(max_rmsd, color="#55554f", linestyle="--")
    axes[0, 1].set(xlabel="partial_t (Å)", ylabel="binder Cα RMSD (Å)",
                   title="Conformational-change distribution")

    for level, color in zip(levels, colors):
        subset = frame[frame["partial_t"].eq(level)]
        axes[1, 0].scatter(
            subset["binder_ca_rmsd"], subset["interface_reuse_fraction"],
            label=f"{level:g} Å", color=color, alpha=0.75, s=28,
        )
    axes[1, 0].axvline(min_rmsd, color="#55554f", linestyle="--")
    axes[1, 0].axvline(max_rmsd, color="#55554f", linestyle="--")
    axes[1, 0].axhline(float(geometry_cfg.get("min_interface_reuse_fraction", 0.60)),
                       color="#55554f", linestyle=":")
    axes[1, 0].set(xlabel="binder Cα RMSD (Å)", ylabel="interface reuse fraction",
                   title="Motion–interface trade-off")
    axes[1, 0].legend(frameon=False, ncol=2)

    width = 0.36
    axes[1, 1].bar(x - width / 2, summary["median_interface_jaccard"], width,
                   label="median interface Jaccard", color="#2a78d6")
    axes[1, 1].bar(x + width / 2, summary["median_interface_reuse_fraction"], width,
                   label="median reuse fraction", color="#1baf7a")
    axes[1, 1].axhline(float(geometry_cfg.get("min_interface_jaccard", 0.25)),
                       color="#2a78d6", linestyle="--", alpha=0.7)
    axes[1, 1].axhline(float(geometry_cfg.get("min_interface_reuse_fraction", 0.60)),
                       color="#1baf7a", linestyle="--", alpha=0.7)
    axes[1, 1].set(xticks=x, xticklabels=labels, ylim=(0, 1), xlabel="partial_t (Å)",
                   ylabel="median interface metric", title="Interface preservation")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("RFD3 partial-diffusion calibration on matched state-1 backbones", fontsize=15)
    fig.text(
        0.5, 0.01,
        "Production geometry thresholds applied uniformly; calibration evaluates geometry, not binding affinity or bistability.",
        ha="center", fontsize=9, color="#55554f",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_calibration_artifacts(
    frame: pd.DataFrame,
    outputs_dir: str,
    geometry_cfg: dict,
    target_rmsd: float = 3.0,
) -> dict:
    os.makedirs(outputs_dir, exist_ok=True)
    frame = frame.sort_values(
        ["partial_t", "s1_rfd3_holo_description", "state_pair_id"]
    ).reset_index(drop=True)
    frame.to_csv(os.path.join(outputs_dir, "partial_t_geometry_all.csv"), index=False)
    summary = summarize_partial_t(frame)
    summary.to_csv(os.path.join(outputs_dir, "partial_t_geometry_summary.csv"), index=False)
    recommendation = choose_partial_t(summary, target_rmsd=target_rmsd)
    with open(os.path.join(outputs_dir, "partial_t_recommendation.json"), "w") as handle:
        json.dump(recommendation, handle, indent=2)
    _write_plot(
        frame, summary, os.path.join(outputs_dir, "partial_t_geometry_comparison.png"),
        geometry_cfg,
    )

    lines = [
        "# RFD3 partial_t calibration",
        "",
        "All noise levels were evaluated on the same state-1 backbones and with the same number of state-2 trajectories. Production geometry thresholds were applied without smoke overrides.",
        "",
        "## Predeclared selection rule",
        "",
        recommendation["selection_rule"],
        "",
        "## Result",
        "",
        recommendation["reason"],
        "",
        "## Thesis-safe interpretation",
        "",
        "This calibration selects the RFD3 coordinate-noise level that most reliably generates backbone pairs satisfying the preregistered conformational-change, interface-reuse, contact, and mutual-exclusion geometry criteria. It does not demonstrate binding affinity, state populations, kinetics, or thermodynamic bistability.",
        "",
        "See `partial_t_geometry_summary.csv` for exact values, `partial_t_geometry_all.csv` for every pair, and `partial_t_geometry_comparison.png` for the comparison figure.",
        "",
    ]
    with open(os.path.join(outputs_dir, "PARTIAL_T_CALIBRATION.md"), "w") as handle:
        handle.write("\n".join(lines))
    return recommendation
