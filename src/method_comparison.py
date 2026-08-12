"""Backbone-clustered DynamicMPNN versus ProteinMPNN-MSD summaries."""
from __future__ import annotations

import os

import pandas as pd


def _collapse(frame: pd.DataFrame, method: str, backbone_col: str) -> pd.DataFrame:
    work = frame.copy()
    tier_pass = work.get("af2_tier", "fail") != "fail"
    if "af2_null_discriminates" in work:
        tier_pass &= work["af2_null_discriminates"].fillna(False).astype(bool)
    work["_passes"] = tier_pass
    work["_score"] = pd.to_numeric(work.get("af2_switch_plddt"), errors="coerce")
    grouped = work.groupby(backbone_col, dropna=False)
    result = grouped.agg(
        n_sequences=("_passes", "size"),
        any_af2_pass=("_passes", "max"),
        n_af2_pass=("_passes", "sum"),
        best_af2_switch_plddt=("_score", "max"),
        median_af2_switch_plddt=("_score", "median"),
    ).reset_index().rename(columns={backbone_col: "backbone"})
    result["method"] = method
    return result


def write_backbone_comparison(
    dynamic_csv: str,
    msd_csv: str,
    out_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dynamic = pd.read_csv(dynamic_csv)
    msd = pd.read_csv(msd_csv)
    dynamic_bb = _collapse(dynamic, "dynamicmpnn", "s1_rfd3_holo_description")
    msd_bb = _collapse(msd, "proteinmpnn_msd", "backbone")
    backbone = pd.concat([dynamic_bb, msd_bb], ignore_index=True)
    backbone.to_csv(os.path.join(out_dir, "method_comparison_by_backbone.csv"), index=False)

    summary = backbone.groupby("method").agg(
        n_backbones=("backbone", "nunique"),
        backbones_with_pass=("any_af2_pass", "sum"),
        pass_rate=("any_af2_pass", "mean"),
        median_best_af2_switch_plddt=("best_af2_switch_plddt", "median"),
    ).reset_index()
    summary.to_csv(os.path.join(out_dir, "method_comparison_summary.csv"), index=False)

    paired = dynamic_bb.merge(msd_bb, on="backbone", suffixes=("_dynamicmpnn", "_msd"))
    paired["delta_best_af2_switch_plddt"] = (
        paired["best_af2_switch_plddt_dynamicmpnn"]
        - paired["best_af2_switch_plddt_msd"]
    )
    paired.to_csv(os.path.join(out_dir, "method_comparison_paired.csv"), index=False)
    return summary, paired
