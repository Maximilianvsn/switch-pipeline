"""Backbone-balanced sequence nulls and discrimination diagnostics."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd


def _single_chain_sequence(sequence: str) -> str:
    parts = str(sequence).split(":")
    if len(parts) != 1 or not parts[0]:
        raise ValueError(
            "Paired nulls require one binder-only sequence, not a multi-chain sequence"
        )
    return parts[0]


def _shuffle(sequence: str, rng: random.Random) -> str:
    residues = list(_single_chain_sequence(sequence))
    original = residues.copy()
    if len(set(residues)) < 2:
        raise ValueError("A composition-preserving negative control is impossible for a homopolymer")
    for _ in range(100):
        rng.shuffle(residues)
        if residues != original:
            return "".join(residues)
    # Deterministic fallback for an extremely unlucky shuffle sequence.
    for offset in range(1, len(original)):
        rotated = original[offset:] + original[:offset]
        if rotated != original:
            return "".join(rotated)
    raise RuntimeError("Could not construct a non-identical composition-preserving scramble")


def balanced_scrambles(
    df: pd.DataFrame,
    sequence_col: str,
    backbone_col: str,
    description_col: str = "poses_description",
    n: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create one composition-matched scramble per searched real sequence.

    `n` caps the number of backbones, not the number of sequences. This keeps
    real and null search multiplicity matched (for example, 16 real sequences
    and 16 scrambles per retained backbone) while retaining backbone-clustered
    inference downstream.
    """
    if backbone_col not in df:
        raise KeyError(f"Backbone column {backbone_col} is required for paired nulls")
    if description_col not in df or sequence_col not in df:
        raise KeyError("Description and sequence columns are required for paired nulls")
    rng = random.Random(seed)
    grouped = [(str(backbone), group) for backbone, group in df.groupby(backbone_col, sort=True)]
    if n is not None and n > 0 and len(grouped) > n:
        rng.shuffle(grouped)
        grouped = grouped[:n]
    candidates = []
    for backbone, group in grouped:
        for _, source in group.iterrows():
            row = source.copy()
            real_id = str(row[description_col])
            row["_null_backbone"] = backbone
            row["_real_design_id"] = real_id
            row["_scr_seq"] = _shuffle(row[sequence_col], rng)
            row[description_col] = f"{real_id}_paired_scramble"
            candidates.append(row)
    result = pd.DataFrame(candidates).reset_index(drop=True)
    if not result.empty and result[description_col].astype(str).duplicated().any():
        raise RuntimeError("Paired-null descriptions must be unique")
    return result


def common_language_auc(real, null, better: str) -> float:
    real = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    null = pd.to_numeric(null, errors="coerce").dropna().to_numpy()
    if len(real) < 3 or len(null) < 3:
        return np.nan
    greater = (real[:, None] > null[None, :]).mean()
    ties = (real[:, None] == null[None, :]).mean()
    auc = greater + 0.5 * ties
    return float(auc if better == "higher" else 1.0 - auc)


def separation_table(
    real_df: pd.DataFrame,
    null_df: pd.DataFrame,
    metric_directions: dict[str, str],
    real_backbone_col: str,
    null_backbone_col: str = "_null_backbone",
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Global AUC plus backbone-paired win rate for every requested metric."""
    rows = []
    for metric_index, (metric, direction) in enumerate(metric_directions.items()):
        if metric not in real_df or metric not in null_df:
            rows.append({"metric": metric, "direction": direction, "auc": np.nan,
                         "paired_win_rate": np.nan, "paired_win_rate_ci_low": np.nan,
                         "paired_win_rate_ci_high": np.nan, "n_pairs": 0})
            continue
        auc = common_language_auc(real_df[metric], null_df[metric], direction)
        real_grouped = real_df.assign(
            _value=pd.to_numeric(real_df[metric], errors="coerce")
        ).groupby(real_backbone_col)["_value"].median()
        null_grouped = null_df.assign(
            _value=pd.to_numeric(null_df[metric], errors="coerce")
        ).groupby(null_backbone_col)["_value"].median()
        paired = pd.concat([real_grouped.rename("real"), null_grouped.rename("null")], axis=1).dropna()
        if direction == "higher":
            wins = paired["real"] > paired["null"]
        else:
            wins = paired["real"] < paired["null"]
        if len(wins):
            rng = np.random.default_rng(seed + metric_index)
            boot = rng.choice(wins.astype(float).to_numpy(), size=(bootstrap_samples, len(wins)), replace=True).mean(axis=1)
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        else:
            ci_low = ci_high = np.nan
        rows.append({
            "metric": metric,
            "direction": direction,
            "auc": auc,
            "paired_win_rate": float(wins.mean()) if len(wins) else np.nan,
            "paired_win_rate_ci_low": float(ci_low),
            "paired_win_rate_ci_high": float(ci_high),
            "n_pairs": int(len(wins)),
        })
    return pd.DataFrame(rows)


def passes_stop_go(
    table: pd.DataFrame, min_auc: float = 0.60, min_pairs: int = 10,
    min_paired_win_rate: float = 0.60,
) -> bool:
    if table.empty:
        return False
    return bool(
        table["auc"].notna().all()
        and (table["auc"] >= min_auc).all()
        and (table["n_pairs"] >= min_pairs).all()
        and table["paired_win_rate"].notna().all()
        and (table["paired_win_rate"] >= min_paired_win_rate).all()
        and table["paired_win_rate_ci_low"].notna().all()
        and (table["paired_win_rate_ci_low"] > 0.5).all()
    )
