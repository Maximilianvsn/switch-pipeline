"""
AF2 initial-guess scoring: the orthogonal, discriminative gate.

Given a design backbone (as an "initial guess" template) and one or more binder
sequences threaded onto it, predict each with AlphaFold2 (via colabdesign, run
in the BindCraft env) and return interface confidence: pLDDT, i_pae, i_ptm.

Rationale (validated 2026-07-08): the Boltz-2 ipTM, ligand-ipTM and affinity
heads are non-discriminative on this problem, a residue-shuffled sequence of
identical composition scoring as well as a real design (AUC ~ 0.5). AF2
initial-guess is orthogonal to Boltz-2 (Evoformer with MSAs against
diffusion-based co-folding) and does discriminate: on a real-versus-scramble
test, AF2 pLDDT separated designs from scrambles with AUC 0.99 and i_pae with
AUC 0.70. AF2 therefore serves as the inexpensive gate, approximately 2.7 s per
prediction, that selects which designs receive Boltz-2 scoring.

Protocol (Bennett et al. 2023, initial guess): a de novo binder carries no
evolutionary signal, so single-sequence AF2 cannot fold it, returning pLDDT
around 35 for real and scrambled sequences alike, which makes the comparison
uninformative. The design's own backbone must be supplied as the template, using
the colabdesign binder protocol with
  rm_binder=False (retain the binder backbone as the initial guess),
  rm_binder_seq=True, rm_binder_sc=True (mask its identity, so that the
  threaded sequence is what is evaluated).

Must be run in the BindCraft environment with its lib directory on
LD_LIBRARY_PATH, which supplies the GLIBCXX_3.4.29 required by numpy and absent
from the system /lib64:
    export LD_LIBRARY_PATH=$BINDCRAFT_ENV/lib:$LD_LIBRARY_PATH
    $BINDCRAFT_ENV/bin/python af2_ig.py \
        --manifest gate_manifest.csv --out af2_scores.csv \
        --data_dir .../localcolabfold/colabfold [--num_recycles 3]

Manifest CSV columns (one row per (design, sequence) to score):
    id            unique id for this prediction (written back on the row)
    backbone      path to the design backbone PDB (target + binder chains;
                  HETATM records removed)
    target_chain  chain id of the target in `backbone`   (e.g. "A")
    binder_chain  chain id of the binder in `backbone`    (e.g. "B")
    seq           binder sequence to thread on (len must match the binder chain)

Output CSV columns: id, plddt, i_pae, i_ptm. colabdesign normalises plddt and
i_pae to 0-1, with i_pae*31 approximately in Angstroms. A per-row failure yields
NaN and does not fail the batch. Backbones are grouped so that the AF2 parameters
and template are prepared once per backbone and reused across its sequences,
amortising the setup cost.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def _score(manifest: pd.DataFrame, data_dir: str, num_recycles: int, out_path: str,
           save_pdb_dir: str | None = None) -> pd.DataFrame:
    from colabdesign import mk_afdesign_model, clear_mem

    if save_pdb_dir:
        os.makedirs(save_pdb_dir, exist_ok=True)

    rows = []
    failures = []

    def failure_row(row):
        return {
            "id": row["id"], "request_hash": row["request_hash"],
            "plddt": np.nan, "i_pae": np.nan, "i_ptm": np.nan, "pdb_path": "",
        }

    # Group by the backbone + chain spec so the template/params are prepped once
    # per backbone (setup dominates; prediction itself is ~2.7 s).
    for (backbone, tchain, bchain), grp in manifest.groupby(["backbone", "target_chain", "binder_chain"]):
        try:
            clear_mem()
            model = mk_afdesign_model(
                protocol="binder", use_multimer=True, use_templates=True, data_dir=data_dir,
            )
            model.prep_inputs(
                pdb_filename=backbone, target_chain=str(tchain), binder_chain=str(bchain),
                rm_binder=False, rm_binder_seq=True, rm_binder_sc=True,
            )
        except Exception as e:
            print(f"PREP-ERR {backbone}: {type(e).__name__}: {e}", flush=True)
            failures.append(f"PREP {backbone}: {type(e).__name__}: {e}")
            for _, r in grp.iterrows():
                rows.append(failure_row(r))
            continue

        for _, r in grp.iterrows():
            try:
                model.predict(seq=str(r["seq"]), num_recycles=num_recycles, models=[0], verbose=False)
                log = model.aux["log"]
                pdb_path = ""
                if save_pdb_dir:
                    # consensus tier: persist the structure this
                    # prediction ALREADY computed, at ~zero extra GPU cost
                    # (save_pdb writes out coordinates already in memory, no
                    # re-inference) -- lets switch_pipeline.py later RMSD
                    # this against Boltz's structure for the same design/
                    # state without a second AF2 pass, since the sequence
                    # scored here is the same one Boltz scores at Step 6.
                    pdb_path = os.path.join(save_pdb_dir, f"{r['id']}.pdb")
                    model.save_pdb(pdb_path)
                rows.append({
                    "id": r["id"],
                    "request_hash": r["request_hash"],
                    "plddt": float(log.get("plddt", np.nan)),
                    "i_pae": float(log.get("i_pae", np.nan)),
                    "i_ptm": float(log.get("i_ptm", np.nan)),
                    "pdb_path": pdb_path,
                })
            except Exception as e:
                print(f"PRED-ERR {r['id']}: {type(e).__name__}: {e}", flush=True)
                failures.append(f"PRED {r['id']}: {type(e).__name__}: {e}")
                rows.append(failure_row(r))
            pd.DataFrame(rows).to_csv(out_path, index=False)  # incremental
        print(f"  backbone {os.path.basename(str(backbone))}: {len(grp)} seqs scored", flush=True)

    result = pd.DataFrame(rows)
    if failures:
        result.to_csv(out_path, index=False)
        raise RuntimeError(
            f"AF2 scoring failed for {len(failures)} request group(s)/prediction(s); "
            f"first error: {failures[0]}"
        )
    return result


def main():
    ap = argparse.ArgumentParser(description="AF2 initial-guess interface scoring (colabdesign)")
    ap.add_argument("--manifest", required=True, help="CSV: id,backbone,target_chain,binder_chain,seq")
    ap.add_argument("--out", required=True, help="output scores CSV: id,plddt,i_pae,i_ptm")
    ap.add_argument("--data_dir", required=True,
                    help="dir containing params/ (e.g. .../localcolabfold/colabfold)")
    ap.add_argument("--num_recycles", type=int, default=3)
    ap.add_argument("--save_pdb_dir", default=None,
                    help="if given, save each prediction's structure here (consensus tier)")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    required = {"id", "request_hash", "backbone", "target_chain", "binder_chain", "seq"}
    missing = required - set(man.columns)
    if missing:
        sys.exit(f"manifest missing columns: {missing}")

    df = _score(man, args.data_dir, args.num_recycles, args.out, save_pdb_dir=args.save_pdb_dir)
    df.to_csv(args.out, index=False)
    ok = int(df["plddt"].notna().sum()) if len(df) else 0
    print(f"AF2-IG done: {ok}/{len(df)} scored -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
