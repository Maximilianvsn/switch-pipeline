"""Attribution of consensus-tier outcomes through two reference comparisons.

The pipeline records a single consensus distance,
`af2_rmsd_boltz_vs_af2_{holo,apo}`, between the Boltz and AF2 binder structures.
That distance alone does not identify which predictor diverged, leaving a large
disagreement unattributable and an empty tier uninterpretable.

This module adds the two reference comparisons against the RFdiffusion3 design
backbone that both predictors were asked to reproduce:

    af2_vs_design     AF2 against the backbone it was initialised from
    boltz_vs_design   Boltz against the same backbone, unseeded
    af2_vs_boltz      the consensus distance recorded by the pipeline

The two are asymmetric. AF2 is run in initial-guess mode, seeded with the design
backbone, so `af2_vs_design` is optimistically biased and partly measures how
little AF2 moved. `boltz_vs_design` carries no such seed and is the independent
designability signal, so the consensus tier principally tests whether Boltz
recovers the design without upstream pressure to do so.

All RMSDs are CA-only over the binder chain, Kabsch-superimposed, so a frame
difference between predictors cannot masquerade as disagreement.

    python src/consensus_diagnostics.py outputs/<run>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# (state, Boltz-confidence column, design-backbone path column)
STATES = [
    ("holo", "s6a_boltz_holo_plddt_mean", "s1_rfd3_holo_location"),
    ("apo", "s6b_boltz_apo_plddt_mean", "state2_pdb"),
]
CONSENSUS_RMSD_THRESHOLD = 2.0


def _ca_by_chain(path: str) -> dict[str, np.ndarray]:
    chains: dict[str, list] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chains.setdefault(line[21], []).append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return {k: np.asarray(v) for k, v in chains.items()}


def _kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    pc, qc = p - p.mean(0), q - q.mean(0)
    v, _, w = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(v @ w))
    r = v @ np.diag([1.0, 1.0, d]) @ w
    return float(np.sqrt(((pc @ r - qc) ** 2).sum(1).mean()))


def diagnose(out_dir: str) -> pd.DataFrame:
    ranked = os.path.join(out_dir, "final_all_ranked.csv")
    if not os.path.isfile(ranked):
        return pd.DataFrame()
    df = pd.read_csv(ranked)
    ws4 = os.path.join(out_dir, "ws4_consensus_pdb")
    rows = []
    for _, r in df.iterrows():
        key = r["poses_description"]
        for state, plddt_col, design_col in STATES:
            af2_p = os.path.join(ws4, state, f"{key}_af2.pdb")
            blz_p = os.path.join(ws4, state, f"{key}_boltz.pdb")
            design = r.get(design_col)
            if not (os.path.isfile(af2_p) and os.path.isfile(blz_p)
                    and isinstance(design, str) and os.path.isfile(design)):
                continue
            try:
                a = _ca_by_chain(af2_p).get("A")
                b = _ca_by_chain(blz_p).get("A")
                if a is None or b is None or len(a) != len(b) or not len(a):
                    continue
                # The binder in the design pose is the chain matching its length.
                cand = [v for v in _ca_by_chain(design).values() if len(v) == len(a)]
                if not cand:
                    continue
                d = cand[0]
            except Exception:
                continue
            rows.append({
                "poses_description": key, "state": state, "n_ca": len(a),
                "af2_vs_design": _kabsch_rmsd(a, d),
                "boltz_vs_design": _kabsch_rmsd(b, d),
                "af2_vs_boltz": _kabsch_rmsd(a, b),
                "boltz_plddt": pd.to_numeric(r.get(plddt_col), errors="coerce"),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["consensus_pass"] = out["af2_vs_boltz"] < CONSENSUS_RMSD_THRESHOLD
    out.to_csv(os.path.join(out_dir, "consensus_diagnostics.csv"), index=False)
    return out


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    """Per-state attribution of the consensus outcome."""
    rows = []
    for state, g in d.groupby("state"):
        conf = g[g["boltz_plddt"] > 0.70]
        rows.append({
            "state": state,
            "n": len(g),
            "af2_vs_design_mean": g["af2_vs_design"].mean(),
            "boltz_vs_design_mean": g["boltz_vs_design"].mean(),
            "af2_vs_boltz_mean": g["af2_vs_boltz"].mean(),
            "af2_recovers_design_lt2A": int((g["af2_vs_design"] < 2).sum()),
            "boltz_recovers_design_lt2A": int((g["boltz_vs_design"] < 2).sum()),
            "consensus_pass": int(g["consensus_pass"].sum()),
            "boltz_confident_n": len(conf),
            "boltz_confident_vs_design_mean": conf["boltz_vs_design"].mean() if len(conf) else np.nan,
            # If this is strongly negative, disagreement tracks Boltz UNCERTAINTY
            # rather than a genuine competing fold.
            "r_boltz_plddt_vs_design_rmsd": g["boltz_plddt"].corr(g["boltz_vs_design"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out_dir = sys.argv[1]
    d = diagnose(out_dir)
    if d.empty:
        sys.exit(f"no comparable consensus structures in {out_dir} "
                 "(needs final_all_ranked.csv + ws4_consensus_pdb/)")
    s = summarise(d)
    s.to_csv(os.path.join(out_dir, "consensus_diagnostics_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.round(2).to_string(index=False))
    print(f"\nwrote consensus_diagnostics.csv + consensus_diagnostics_summary.csv "
          f"to {out_dir}/")


if __name__ == "__main__":
    main()
