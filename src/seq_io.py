"""Binder sequence extraction, FASTA emission, and sequence-diversity stats.

Extracted from `switch_pipeline.py` without modification; none of these functions
captured enclosing state, so the move is behaviour-preserving by construction.

`binder_seq` carries the domain constraint: MPNN emits one ":"-joined sequence
per input-PDB chain, in the chain order of the PDB, and the binder is not
necessarily last. See its docstring; assuming otherwise produced a scoring
error.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd


def require_single_chain_sequence(value, context: str) -> str:
    """Assert a sequence is binder-only (no ":"-joined target chains).

    Fails fast: if an upstream schema change ever starts returning target chains
    too, silently scoring the target instead of the binder is far worse than a crash.
    """
    parts = str(value).split(":")
    if len(parts) != 1 or not parts[0]:
        raise ValueError(
            f"{context} must contain one binder-only sequence; "
            f"got {len(parts)} colon-separated chains"
        )
    return parts[0]


def binder_seq(full_seq: str, binder_chain_letter: str) -> str:
    """Pull the binder chain out of a ":"-joined multi-chain MPNN sequence.

    LigandMPNN emits one ":"-joined sequence per input-PDB chain, in the PDB's
    chain order (alphabetical from A). The binder is not always the last chain:
    the holo pose is target(A):binder(B) so the binder is last, but the apo pose is
    binder(A):PCNA(B) so the binder is FIRST. Blindly taking parts[-1] silently
    grabbed PCNA for the apo side (a 249-residue folded protein), inflating apo
    monomer pLDDT and corrupting both the joint-min ranking and the apo decoy RMSD.

    Index by the known chain letter instead (A->0, B->1, ...). Assumes
    single-chain targets, which holds for PD-L1 and PCNA.
    """
    parts = full_seq.split(":")
    if len(parts) == 1:
        return parts[0]
    idx = ord(binder_chain_letter) - ord("A")
    return parts[idx]


def write_binder_fastas(df: pd.DataFrame, seq_col: str, out_dir: str,
                        binder_chain_letter: str) -> list[str]:
    """One binder-only FASTA per row, named by `poses_description`."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for _, row in df.iterrows():
        seq = binder_seq(row[seq_col], binder_chain_letter)
        fasta_path = os.path.join(out_dir, f"{row['poses_description']}.fasta")
        with open(fasta_path, "w") as fh:
            fh.write(f">{row['poses_description']}\n{seq}\n")
        paths.append(fasta_path)
    return paths


def write_binder_only_fastas(seq_df: pd.DataFrame, seq_col: str,
                             out_dir: str) -> tuple[list[str], list[str]]:
    """Like `write_binder_fastas`, but for columns that are already binder-only.

    Returns (paths, keys). Uses `require_single_chain_sequence` rather than
    `binder_seq`, so a column that unexpectedly carries target chains raises
    instead of being silently indexed into.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths, keys = [], []
    for _, row in seq_df.iterrows():
        binder = require_single_chain_sequence(row[seq_col], seq_col)
        key = str(row["poses_description"])
        fasta_path = os.path.join(out_dir, f"{key}.fasta")
        with open(fasta_path, "w") as fh:
            fh.write(f">{key}\n{binder}\n")
        paths.append(fasta_path)
        keys.append(key)
    return paths, keys


def compute_seq_diversity(df: pd.DataFrame, seq_col: str) -> dict:
    """Pairwise sequence identity and per-sequence amino-acid complexity."""
    seqs = df[seq_col].dropna().tolist()
    binder_seqs = [require_single_chain_sequence(s, seq_col) for s in seqs]

    n = len(binder_seqs)
    if n == 0:
        return {"n_seqs": 0}

    unique_aas = [len(set(s)) for s in binder_seqs]
    lengths = [len(s) for s in binder_seqs]

    pairwise_ids = []
    for i, j in combinations(range(n), 2):
        s1, s2 = binder_seqs[i], binder_seqs[j]
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            continue
        matches = sum(a == b for a, b in zip(s1[:min_len], s2[:min_len]))
        pairwise_ids.append(matches / min_len)

    return {
        "n_seqs": n,
        "mean_length": float(np.mean(lengths)),
        "mean_unique_aas": float(np.mean(unique_aas)),
        "mean_pairwise_identity": float(np.mean(pairwise_ids)) if pairwise_ids else float("nan"),
        "min_pairwise_identity": float(np.min(pairwise_ids)) if pairwise_ids else float("nan"),
        "max_pairwise_identity": float(np.max(pairwise_ids)) if pairwise_ids else float("nan"),
    }
