"""Null-audited final Boltz interface metrics for both sequence methods.

These metrics summarize predictor confidence; they are not affinity or free
energy estimates. The composite is only promoted as a validated evaluation
metric when every component family separates real designs from paired
composition-matched scrambles.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import paired_nulls


PAE_SCALE_ANGSTROM = 10.0


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _harmonic(a: pd.Series, b: pd.Series) -> pd.Series:
    denominator = a + b
    return (2.0 * a * b / denominator.where(denominator > 0)).where(a.notna() & b.notna())


def derive_interface_metrics(
    frame: pd.DataFrame,
    method: str,
    backbone_col: str,
    description_col: str,
    state_a: dict[str, str],
    state_b: dict[str, str],
) -> pd.DataFrame:
    """Create common final-interface metrics from method-specific columns."""
    out = pd.DataFrame({
        "method": method,
        "backbone": frame[backbone_col].astype(str),
        "description": frame[description_col].astype(str),
    })
    for label, mapping in (("state_a", state_a), ("state_b", state_b)):
        out[f"{label}_iptm"] = _numeric(frame, mapping["iptm"])
        out[f"{label}_ipae"] = _numeric(frame, mapping["ipae"])
        out[f"{label}_binder_plddt"] = _numeric(frame, mapping["plddt"])
        # ipTM measures interface confidence; exp(-iPAE/10A) penalizes uncertain
        # interfaces on an explicit, interpretable PAE length scale.
        out[f"{label}_interface_confidence"] = np.sqrt(
            out[f"{label}_iptm"].clip(0, 1)
            * np.exp(-out[f"{label}_ipae"].clip(lower=0) / PAE_SCALE_ANGSTROM)
        )

    out["boltz_harmonic_iptm"] = _harmonic(out["state_a_iptm"], out["state_b_iptm"])
    out["boltz_worst_ipae"] = out[["state_a_ipae", "state_b_ipae"]].max(axis=1, skipna=False)
    out["boltz_min_binder_plddt"] = out[
        ["state_a_binder_plddt", "state_b_binder_plddt"]
    ].min(axis=1, skipna=False)
    out["boltz_interface_harmonic"] = _harmonic(
        out["state_a_interface_confidence"], out["state_b_interface_confidence"]
    )
    # Fold confidence is included as a soft penalty. sqrt prevents pLDDT from
    # dominating the explicitly interface-focused term.
    out["boltz_final_interface_score"] = (
        out["boltz_interface_harmonic"]
        * np.sqrt(out["boltz_min_binder_plddt"].clip(0, 1))
    )
    return out


def _method_frame(path: str, method: str):
    if not os.path.isfile(path):
        return None
    frame = pd.read_csv(path)
    if method == "dynamicmpnn":
        return derive_interface_metrics(
            frame, method, "s1_rfd3_holo_description", "poses_description",
            {"iptm": "s6a_boltz_holo_iptm", "ipae": "s6a_boltz_holo_ipae",
             "plddt": "s6a_boltz_holo_plddt_mean"},
            {"iptm": "s6b_boltz_apo_iptm", "ipae": "s6b_boltz_apo_ipae",
             "plddt": "s6b_boltz_apo_plddt_mean"},
        )
    return derive_interface_metrics(
        frame, method, "backbone", "poses_description",
        {"iptm": "s7a_msd_boltz_holo_iptm", "ipae": "s7a_msd_boltz_holo_ipae",
         "plddt": "s7a_msd_boltz_holo_plddt_mean"},
        {"iptm": "s7b_msd_boltz_apo_iptm", "ipae": "s7b_msd_boltz_apo_ipae",
         "plddt": "s7b_msd_boltz_apo_plddt_mean"},
    )


def _null_frame(path: str):
    if not os.path.isfile(path):
        return None
    frame = pd.read_csv(path)
    if "_null_backbone" not in frame:
        return None
    return derive_interface_metrics(
        frame, "scramble_null", "_null_backbone", "poses_description",
        {"iptm": "scr_holo_iptm", "ipae": "scr_holo_ipae",
         "plddt": "scr_holo_plddt_mean"},
        {"iptm": "scr_apo_iptm", "ipae": "scr_apo_ipae",
         "plddt": "scr_apo_plddt_mean"},
    )


def write_boltz_interface_evaluation(outputs_dir: str) -> dict:
    dynamic = _method_frame(os.path.join(outputs_dir, "final_all_ranked.csv"), "dynamicmpnn")
    msd = _method_frame(os.path.join(outputs_dir, "msd_final_all_ranked.csv"), "proteinmpnn_msd")
    if dynamic is None or msd is None:
        return {"available": False, "validated_against_null": False}
    combined = pd.concat([dynamic, msd], ignore_index=True)
    combined.to_csv(os.path.join(outputs_dir, "boltz_interface_all.csv"), index=False)

    aggregate_columns = {
        "n_sequences": ("description", "size"),
        "best_boltz_final_interface_score": ("boltz_final_interface_score", "max"),
        "median_boltz_final_interface_score": ("boltz_final_interface_score", "median"),
        "best_boltz_interface_harmonic": ("boltz_interface_harmonic", "max"),
        "best_boltz_harmonic_iptm": ("boltz_harmonic_iptm", "max"),
        "best_boltz_worst_ipae": ("boltz_worst_ipae", "min"),
    }
    by_backbone = combined.groupby(["method", "backbone"]).agg(**aggregate_columns).reset_index()
    by_backbone.to_csv(os.path.join(outputs_dir, "boltz_interface_by_backbone.csv"), index=False)
    summary = by_backbone.groupby("method").agg(
        n_backbones=("backbone", "nunique"),
        median_best_interface_score=("best_boltz_final_interface_score", "median"),
        median_best_interface_harmonic=("best_boltz_interface_harmonic", "median"),
        median_best_harmonic_iptm=("best_boltz_harmonic_iptm", "median"),
        median_best_worst_ipae=("best_boltz_worst_ipae", "median"),
    ).reset_index()
    summary.to_csv(os.path.join(outputs_dir, "boltz_interface_summary.csv"), index=False)

    dyn_bb = by_backbone[by_backbone["method"] == "dynamicmpnn"]
    msd_bb = by_backbone[by_backbone["method"] == "proteinmpnn_msd"]
    paired = dyn_bb.merge(msd_bb, on="backbone", suffixes=("_dynamicmpnn", "_msd"))
    for metric in (
        "best_boltz_final_interface_score", "best_boltz_interface_harmonic",
        "best_boltz_harmonic_iptm", "best_boltz_worst_ipae",
    ):
        paired[f"delta_{metric}"] = paired[f"{metric}_dynamicmpnn"] - paired[f"{metric}_msd"]
    paired.to_csv(os.path.join(outputs_dir, "boltz_interface_paired.csv"), index=False)

    null = _null_frame(os.path.join(outputs_dir, "scramble_null.csv"))
    audit = None
    validated = False
    if null is not None and not null.empty:
        null.to_csv(os.path.join(outputs_dir, "boltz_interface_null.csv"), index=False)
        directions = {
            "boltz_final_interface_score": "higher",
            "boltz_interface_harmonic": "higher",
            "boltz_harmonic_iptm": "higher",
            "boltz_worst_ipae": "lower",
            "boltz_min_binder_plddt": "higher",
        }
        audit = paired_nulls.separation_table(
            dynamic, null, directions,
            real_backbone_col="backbone", null_backbone_col="backbone",
        )
        audit.to_csv(os.path.join(outputs_dir, "boltz_interface_null_audit.csv"), index=False)
        validated = paired_nulls.passes_stop_go(audit, min_auc=0.70, min_pairs=20)

    status = {
        "available": True,
        "validated_against_null": bool(validated),
        "n_dynamic_backbones": int(dynamic["backbone"].nunique()),
        "n_msd_backbones": int(msd["backbone"].nunique()),
        "n_paired_backbones": int(len(paired)),
        "metric_definition": (
            "harmonic across states of sqrt(ipTM*exp(-interface_PAE/10A)), "
            "softly penalized by sqrt(min binder pLDDT)"
        ),
        "interpretation": "predictor-confidence diagnostic, not affinity or free energy",
    }
    with open(os.path.join(outputs_dir, "boltz_interface_status.json"), "w") as handle:
        json.dump(status, handle, indent=2)
    return status
