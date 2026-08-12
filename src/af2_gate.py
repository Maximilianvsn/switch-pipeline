"""AF2 initial-guess two-state gate: the discriminative selector.

`AF2GateEnv` carries the output directory, the binder chain letters and the AF2
configuration explicitly.

AF2 initial-guess separates real designs from composition-matched scrambles
(pLDDT AUC 0.99 on the 2026-07-14 validation set, 0.92 at production scale),
which the Boltz-2 confidence metrics do not on this problem. Selection therefore
rests on this gate rather than on the downstream Boltz-2 scoring, and any change
to the chain assignment or request construction here invalidates that basis.

The chain convention differs between the two states:

    holo backbone   target = chain A, binder = `binder_chain`
    apo backbone    target = chain B, binder = `apo_binder_chain`
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from af2_runner import run_af2_ig, build_state_requests

METRICS = ("plddt", "i_pae", "i_ptm")


@dataclass
class AF2GateEnv:
    """Everything the gate used to capture from `main()`'s scope."""
    outputs: str
    binder_chain: str
    apo_binder_chain: str
    af2_cfg: dict
    params_dir: str


def af2_gate_score(env: AF2GateEnv, seq_df: pd.DataFrame, seq_col: str,
                   subdir: str, save_structures: bool = False) -> pd.DataFrame:
    """Score every design's binder sequence with AF2 initial-guess in both states.

    Returns a frame keyed on `poses_description` with
    `af2_{holo,apo}_{plddt,i_pae,i_ptm}`. Runs via the sharded sbatch array in the
    BindCraft env (`af2_runner`).

    `save_structures=True` also persists each prediction's structure (consensus tier consensus
    tier) and adds `af2_{holo,apo}_pdb` columns — at ~zero extra GPU cost, because
    `save_pdb` writes coordinates that the same `predict()` call already computed, with
    no re-inference. Set it only for candidates that will actually reach Boltz
    scoring, so the later Boltz-vs-AF2 RMSD has something to compare against
    without ever running AF2 twice.
    """
    gate_dir = os.path.join(env.outputs, subdir)
    holo_req = build_state_requests(
        seq_df, "poses_description", "s1_rfd3_holo_location", "A", env.binder_chain,
        seq_col, os.path.join(gate_dir, "holo_bb"), "holo__")
    apo_req = build_state_requests(
        seq_df, "poses_description", "state2_pdb", "B", env.apo_binder_chain,
        seq_col, os.path.join(gate_dir, "apo_bb"), "apo__")
    reqs = pd.concat([holo_req, apo_req], ignore_index=True)
    if reqs.empty:
        return pd.DataFrame(columns=["poses_description"])

    save_dir = os.path.join(gate_dir, "structures") if save_structures else None
    scores = run_af2_ig(reqs, os.path.join(gate_dir, "af2"), env.af2_cfg,
                        env.params_dir, save_pdb_dir=save_dir)
    scores = scores.merge(reqs[["id", "_orig_id"]], on="id", how="left")

    rows: dict = {}
    for _, r in scores.iterrows():
        orig_id = r["_orig_id"]
        state = "holo" if str(r["id"]).startswith("holo__") else "apo"
        rows.setdefault(orig_id, {"poses_description": orig_id})
        for metric in METRICS:
            rows[orig_id][f"af2_{state}_{metric}"] = r.get(metric)
        if save_structures:
            rows[orig_id][f"af2_{state}_pdb"] = r.get("pdb_path", "")
    return pd.DataFrame(list(rows.values()))
