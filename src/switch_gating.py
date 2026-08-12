"""
Two-tier AF2 gating for the switch pipeline.

Selection uses the AF2 initial-guess metrics, which are discriminative and
orthogonal to Boltz-2 ipTM (AUC ~ 0.5 on this problem). Two tiers are defined on
different bases:

  strict   absolute thresholds from the literature (Bennett et al.; BindCraft).
           The tier may be empty at marginal design quality, which is a valid
           outcome.
  relaxed  relative to the scramble null of the run itself. A design passes if it
           exceeds the composition-matched null, pLDDT above the null p95 and
           i_pae below the null p05. The bar is re-derived from the null in every
           run.

Both tiers require both states to pass, a switch being bounded by its weaker
state. Ranking within a tier uses the harmonic mean of the two states' AF2 pLDDT,
subject to the i_pae gate.

All colabdesign metrics are normalised to 0-1, with i_pae*31 approximately in
Angstroms, so the strict i_pae threshold of 0.34 corresponds to a
pae_interaction of approximately 10.5 A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Absolute, literature-grade bar (colabdesign-normalized units).
# i_pae 0.34*31 ~ 10.5 A (Bennett <10); pLDDT 0.80; i_ptm 0.50.
STRICT_ABS = {"plddt": 0.80, "i_pae": 0.34, "i_ptm": 0.50}


def harmonic(x: pd.Series | float, y: pd.Series | float):
    """AND-logic combiner: collapses toward 0 if EITHER state is weak."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = x + y
    out = np.where(denom > 0, 2 * x * y / denom, 0.0)
    return out


def null_thresholds(null_df: pd.DataFrame, specs: dict[str, str]) -> dict[str, float]:
    """Per-metric threshold at the null's p95 (for 'higher-is-better' metrics)
    or p05 ('lower-is-better'). `specs` maps column -> 'higher'|'lower'. Missing
    or all-NaN columns are skipped (no threshold -> that gate is not applied).
    """
    thr = {}
    for col, direction in specs.items():
        if col not in null_df.columns:
            continue
        s = pd.to_numeric(null_df[col], errors="coerce").dropna()
        if len(s) < 5:  # too few nulls to set a stable percentile
            continue
        thr[col] = float(s.quantile(0.95 if direction == "higher" else 0.05))
    return thr


def null_separation(real: pd.Series, null: pd.Series, better: str) -> dict:
    """Report how well a metric separates real designs from the null: z-score of
    real-mean vs null, and the common-language effect-size AUC (P a real design
    beats a random null in the 'better' direction). A metric with AUC~0.5 must
    not be used to gate; this is how the Boltz ipTM metric was identified.
    """
    r = pd.to_numeric(real, errors="coerce").dropna()
    n = pd.to_numeric(null, errors="coerce").dropna()
    if len(r) < 3 or len(n) < 3:
        return {"z": np.nan, "auc": np.nan, "n_real": len(r), "n_null": len(n)}
    sd = n.std()
    z = (r.mean() - n.mean()) / sd if sd > 0 else np.nan
    # common-language effect size P(real > null), then orient to 'better'
    gt = sum((r.values[:, None] > n.values[None, :]).mean(axis=1)) / len(r)
    auc = gt if better == "higher" else 1 - gt
    return {"z": float(z), "auc": float(auc), "n_real": len(r), "n_null": len(n)}


def assign_af2_tiers(
    df: pd.DataFrame,
    holo_plddt: str, apo_plddt: str, holo_ipae: str, apo_ipae: str,
    null_df: pd.DataFrame | None = None,
    strict_abs: dict | None = None,
    holo_iptm: str | None = None, apo_iptm: str | None = None,
) -> pd.DataFrame:
    """Add AF2 ranking + tier columns to `df` (returns the same df, mutated):

      af2_switch_plddt : harmonic mean of the two states' AF2 pLDDT (rank column)
      af2_worst_ipae   : max(holo_ipae, apo_ipae) — the weaker interface
      af2_strict       : bool, passes the absolute bar in both states
      af2_relaxed      : bool, beats the null (p95 pLDDT, p05 i_pae) in both states
      af2_tier         : 'strict' | 'relaxed' | 'fail'

    null_df (this run's scramble scores, same column names) is required for the
    relaxed tier; without it only the strict tier is computed.
    """
    strict_abs = strict_abs or STRICT_ABS
    df = df.copy()

    df["af2_switch_plddt"] = harmonic(df[holo_plddt], df[apo_plddt])
    df["af2_worst_ipae"] = np.maximum(
        pd.to_numeric(df[holo_ipae], errors="coerce"),
        pd.to_numeric(df[apo_ipae], errors="coerce"),
    )

    # Strict tier: absolute thresholds, both states
    strict = (
        (pd.to_numeric(df[holo_plddt], errors="coerce") > strict_abs["plddt"])
        & (pd.to_numeric(df[apo_plddt], errors="coerce") > strict_abs["plddt"])
        & (pd.to_numeric(df[holo_ipae], errors="coerce") < strict_abs["i_pae"])
        & (pd.to_numeric(df[apo_ipae], errors="coerce") < strict_abs["i_pae"])
    )
    if holo_iptm and apo_iptm and holo_iptm in df.columns and apo_iptm in df.columns:
        strict &= (pd.to_numeric(df[holo_iptm], errors="coerce") > strict_abs["i_ptm"]) \
                  & (pd.to_numeric(df[apo_iptm], errors="coerce") > strict_abs["i_ptm"])
    df["af2_strict"] = strict.fillna(False)

    # Relaxed tier: relative to this run's null, both states
    if null_df is not None and len(null_df):
        specs = {holo_plddt: "higher", apo_plddt: "higher",
                 holo_ipae: "lower", apo_ipae: "lower"}
        thr = null_thresholds(null_df, specs)
        if len(thr) != len(specs):
            df["af2_relaxed"] = False
        else:
            relaxed = pd.Series(True, index=df.index)
            for col, direction in specs.items():
                v = pd.to_numeric(df[col], errors="coerce")
                relaxed &= (v > thr[col]) if direction == "higher" else (v < thr[col])
            df["af2_relaxed"] = relaxed.fillna(False)
    else:
        df["af2_relaxed"] = False

    df["af2_tier"] = np.where(df["af2_strict"], "strict",
                       np.where(df["af2_relaxed"], "relaxed", "fail"))
    return df
