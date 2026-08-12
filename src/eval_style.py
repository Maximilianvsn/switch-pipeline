"""Shared figure styling and small statistics helpers for the evaluation scripts.

`protein_only_evaluation.py` and `compare_replicates.py` share a single style
definition. The colour assignments carry meaning:

  * `METHOD_COLORS` identifies a method and is fixed across all figures, so that
    a colour denotes the same method throughout. The scramble null is a neutral
    grey outside the categorical pair, being a statistical baseline rather than a
    competing method.
  * Sequential magnitude uses one hue from light to dark.
  * Reference and threshold lines use a neutral dashed style.
  * Text uses the ink tokens rather than a series colour.
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Shared plot styling
# Categorical colors identify a METHOD (DynamicMPNN vs ProteinMPNN-MSD vs
# the scrambled-sequence null) and stay FIXED across every plot in this
# file so a color always means the same thing. Blue/aqua are a fixed-order
# categorical pair; the null control gets a neutral gray deliberately
# outside that pair, being a statistical baseline rather than a method.
METHOD_COLORS = {
    "DynamicMPNN": "#2a78d6",
    "ProteinMPNN-MSD": "#1baf7a",
    "Scramble (null)": "#8a8a86",
}
# Diverging encodings (e.g. positive vs. negative correlation around zero)
# use this blue/red pair — blue matches METHOD_COLORS so "positive" reads
# consistently across plots.
DIVERGING_POS, DIVERGING_NEG = "#2a78d6", "#e34948"
# Reference/threshold lines (not data) always use this neutral gray dashed
# style so they read as "guide," never as another data series.
REF_LINE_KW = dict(color="#8a8a86", linestyle="--", linewidth=1)

# Ink tokens — text/labels wear these, never a series color (dataviz rule).
INK, INK_MUTED, GRID = "#22221f", "#55554f", "#e8e8e4"


def apply_house_style():
    """One consistent look for every figure (spines, grid, fonts, dpi), applied
    once at the start of run_evaluation so all plots read as a single system
    instead of drifting per-figure. Only global rcParams — no per-plot logic."""
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 170, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#c9c9c4", "axes.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlepad": 9,
        "axes.labelsize": 10, "axes.labelcolor": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "text.color": INK, "legend.frameon": False, "font.size": 10,
        "figure.titlesize": 13, "figure.titleweight": "bold",
    })


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a binomial proportion k/n — honest small-sample
    error bars for success / designability rates (n is often only ~40)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


# Combined switch scoring (evaluation-side re-ranking)
# The pipeline ranks by switch_score = holo_iptm + apo_iptm (a SUM). A sum
# lets one strong state mask a weak one: (holo=0.9, apo=0.5) and
# (holo=0.7, apo=0.7) both sum to 1.4, yet only the second is a real switch
# (both states bind). These combiners instead PUNISH the weaker state, so a
# design must satisfy both states to score well:
#
#   switch_harmonic  = 2ha/(h+a)   F1-analogue; dominated by the smaller of
#                                  the two -> (0.9,0.5)=0.64 < (0.7,0.7)=0.70
#   switch_geometric = sqrt(h*a)   milder penalty than harmonic
#   switch_min       = min(h,a)    most conservative ("a chain is only as
#                                  strong as its weakest link")
#
# Harmonic is the headline: parameter-free, bounded 0-1 like ipTM itself,
# and it collapses toward 0 as either state fails. All are computed purely
# from columns the pipeline already wrote — this is post-hoc re-scoring,
# it does not change any design or prediction step.
