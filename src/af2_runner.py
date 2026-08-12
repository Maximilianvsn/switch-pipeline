"""
Pipeline-side orchestration for the AF2 initial-guess gate.

af2_ig.py performs the scoring but must run in the BindCraft environment (JAX
AF2 with colabdesign) with its lib directory on LD_LIBRARY_PATH, which differs
from the `protflow` environment of the pipeline. Rather than an in-process call,
this module submits af2_ig.py as an sbatch array, one task per shard of
predictions, waits for the array to complete and collects the per-shard score
CSV files.

Notes:
- At approximately 2.7 s per prediction, a shard of 60 predictions is a
  three-minute task; the array parallelises across GPU nodes at bounded
  concurrency.
- Prediction, shard and scheduler failures are fatal. A rejection must not be
  derived from missing or NaN execution output.
- ProtFlow is not used here, the environment being different; the approach
  mirrors the subprocess and sbatch pattern used for ProteinMPNN-MSD.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protein_chain_lengths(path: str) -> tuple[list[str], dict[str, int]]:
    """Return protein-chain order and observed standard-residue counts."""
    from Bio.PDB import MMCIFParser, PDBParser
    parser = (MMCIFParser(QUIET=True)
              if str(path).endswith((".cif", ".cif.gz"))
              else PDBParser(QUIET=True))
    structure = parser.get_structure("af2_request", path)
    chain_ids = []
    lengths = {}
    for chain in structure[0].get_chains():
        n_residues = sum(residue.id[0] == " " for residue in chain.get_residues())
        if n_residues:
            chain_ids.append(chain.id)
            lengths[chain.id] = n_residues
    return chain_ids, lengths


def _sequence_for_chain(full_sequence: str, chain_id: str, chain_ids: list[str]) -> str:
    """Select a chain's sequence using the prepared PDB's explicit chain order."""
    parts = str(full_sequence).split(":")
    if len(parts) == 1:
        return parts[0]
    if len(parts) != len(chain_ids):
        raise ValueError(
            f"Sequence has {len(parts)} colon-separated chains but prepared backbone "
            f"has {len(chain_ids)} protein chains {chain_ids}"
        )
    if chain_id not in chain_ids:
        raise ValueError(f"Binder chain {chain_id!r} is absent from prepared backbone chains {chain_ids}")
    return parts[chain_ids.index(chain_id)]


def _request_hash(row: pd.Series, af2_cfg: dict, params_dir: str) -> str:
    """Fingerprint every input that can change an AF2 prediction."""
    payload = {
        "id": str(row["id"]),
        "backbone_sha256": _sha256_file(str(row["backbone"])),
        "target_chain": str(row["target_chain"]),
        "binder_chain": str(row["binder_chain"]),
        "sequence": str(row["seq"]),
        "num_recycles": int(af2_cfg.get("num_recycles", 3)),
        "models": [0],
        "params_dir": os.path.realpath(params_dir),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _write_shards(requests: pd.DataFrame, shard_dir: str, shard_size: int) -> int:
    os.makedirs(shard_dir, exist_ok=True)
    n = len(requests)
    n_shards = max(1, -(-n // shard_size))  # ceil
    for i in range(n_shards):
        sub = requests.iloc[i * shard_size:(i + 1) * shard_size]
        sub.to_csv(os.path.join(shard_dir, f"shard_{i:03d}.csv"), index=False)
    return n_shards


def _slurm_job_name(work_dir: str) -> str:
    path = Path(work_dir)
    parts = path.parts
    try:
        output_index = parts.index("outputs")
        run_name = parts[output_index + 1]
    except (ValueError, IndexError):
        run_name = path.parent.name or "pipeline"
    stage_name = path.parent.name or path.name
    return f"{run_name}__{stage_name}__af2"[:128]


# AF2-IG shard cost, measured on gpu-single A100s. The driver is not whether
# structures are saved -- it is how many predictions share a backbone, because
# per-backbone setup (strip chains, load the guess, build features) is paid once
# and then amortized over that backbone's predictions.
#
#   shard_seconds ~= n_backbones * SETUP + n_predictions * INFERENCE
#                 == shard_size * (SETUP / preds_per_backbone + INFERENCE)
#
# Validated against all three passes in this pipeline:
#   s5_5 gate     16-32 preds/backbone -> ~3.5 s/pred; 60-pred shard ~3.5 min
#                 (matches the measured "~3 min per 60 predictions")
#   s2_5 desig    2 preds/backbone     -> ~15 s/pred;  300-pred shard ~76 min
#   struct pass   1 pred/backbone      -> ~28 s/pred;  60-pred shard ~28 min
#                 (matches the measured "~20-25 min")
#
# Getting this wrong is expensive in exactly one direction: a shard that overruns
# is a TIMEOUT that discards its whole batch. It cost eight 5-hour sweep runs when
# the rate was keyed on save_pdb_dir instead of on amortization.
_SETUP_SECONDS_PER_BACKBONE = 25.0
_INFERENCE_SECONDS_PER_PRED = 2.7
_WALLTIME_HEADROOM = 3.0
_WALLTIME_SETUP_S = 600          # model load / node warmup, once per shard
_WALLTIME_MIN_S = 1200
_WALLTIME_MAX_S = 24 * 3600


def shard_walltime(shard_size: int, preds_per_backbone: float = 1.0) -> str:
    """HH:MM:SS budget for one shard of `shard_size` predictions.

    `preds_per_backbone` is how many predictions in the shard share a backbone,
    i.e. how far the per-backbone setup cost is amortized. Pass the real ratio
    (n_requests / n_unique_backbones); 1.0 is the safe pessimum.
    """
    ppb = max(1.0, float(preds_per_backbone))
    per_pred = _SETUP_SECONDS_PER_BACKBONE / ppb + _INFERENCE_SECONDS_PER_PRED
    budget = int(shard_size * per_pred * _WALLTIME_HEADROOM) + _WALLTIME_SETUP_S
    budget = max(_WALLTIME_MIN_S, min(budget, _WALLTIME_MAX_S))
    h, rem = divmod(budget, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _submit_array(work_dir: str, n_shards: int, af2_cfg: dict, params_dir: str,
                  max_concurrent: int = 20, save_pdb_dir: str | None = None,
                  shard_size: int = 60, preds_per_backbone: float = 1.0) -> str:
    bc = af2_cfg["bindcraft_env"]
    nr = int(af2_cfg.get("num_recycles", 3))
    logs = os.path.join(work_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "out"), exist_ok=True)
    save_flag = f"--save_pdb_dir {save_pdb_dir}" if save_pdb_dir else ""
    if save_pdb_dir:
        os.makedirs(save_pdb_dir, exist_ok=True)
    walltime = shard_walltime(shard_size, preds_per_backbone)
    sbatch = os.path.join(work_dir, "run_af2_array.sh")
    with open(sbatch, "w") as f:
        # Mem sized from real SLURM TRES accounting (host mem, not GPU VRAM): a
        # running shard used ~2.9GB regardless of shard size (same loaded model
        # scored repeatedly, not reloaded per prediction), against the 32GB
        # previously requested here.
        f.write(f"""#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --time={walltime}
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --job-name={_slurm_job_name(work_dir)}
#SBATCH --array=0-{n_shards - 1}%{max_concurrent}
#SBATCH --output={logs}/shard_%a.log
BC={bc}
export LD_LIBRARY_PATH=$BC/lib:$LD_LIBRARY_PATH
S=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
$BC/bin/python {_SCRIPTS}/af2_ig.py \\
  --manifest {work_dir}/shards/shard_$S.csv \\
  --out {work_dir}/out/shard_$S.csv \\
  --data_dir {params_dir} --num_recycles {nr} {save_flag}
""")
    out = subprocess.run(["sbatch", "--parsable", sbatch], capture_output=True, text=True, check=True)
    return out.stdout.strip().split(";")[0]


# SLURM states that mean "still in flight" — never a failure, however the job
# happens to be reported by squeue at that instant.
NON_TERMINAL_STATES = {
    "PENDING", "RUNNING", "SUSPENDED", "COMPLETING", "CONFIGURING", "RESIZING",
    "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD", "SIGNALING",
    "STAGE_OUT", "STOPPED",
}


def _wait(job_id: str, poll: int = 30, timeout: int = 24 * 3600) -> None:
    # 24h ceiling (was 6h): a large AF2 array (e.g. the 86-shard MSD gate) can take
    # >6h to drain purely from gpu-single queue contention, not because anything is
    # wrong. The orchestrator has a multi-day walltime, and run_af2_ig is resumable
    # (completed shards are reused), so waiting longer is safe and avoids abandoning
    # an array that is still making progress.
    start = time.time()
    while time.time() - start < timeout:
        q = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
        if not q.stdout.strip():
            for _ in range(6):
                accounting = subprocess.run(
                    ["sacct", "-j", job_id, "--format=JobIDRaw,State,ExitCode", "-n", "-P"],
                    capture_output=True, text=True,
                )
                rows = [line.split("|") for line in accounting.stdout.splitlines() if line.strip()]
                rows = [row for row in rows if len(row) >= 3 and row[0].startswith(job_id)]
                if rows:
                    states = [row[1].strip().split("+")[0] for row in rows]
                    # squeue can transiently return nothing for a job that is still
                    # alive (scheduler hiccup, or the gap between squeue dropping it
                    # and sacct marking it terminal). Treating a NON-TERMINAL state
                    # as failure killed a 34-minute run whose AF2 array then went on
                    # to COMPLETE normally. Only terminal-and-not-COMPLETED is a
                    # real failure; anything still in flight means keep waiting.
                    if any(st in NON_TERMINAL_STATES for st in states):
                        break                      # fall back to the squeue poll loop
                    failed = [row for row, st in zip(rows, states) if st != "COMPLETED"]
                    if failed:
                        details = "; ".join("|".join(row[:3]) for row in failed[:12])
                        raise RuntimeError(f"AF2 array {job_id} failed in Slurm accounting: {details}")
                    return
                time.sleep(2)
            else:
                raise RuntimeError(f"AF2 array {job_id} left squeue but produced no sacct record")
            time.sleep(poll)
            continue
        time.sleep(poll)
    subprocess.run(["scancel", job_id], capture_output=True, text=True)
    raise TimeoutError(f"AF2 array {job_id} did not finish within {timeout}s")


def prep_backbone(src: str, keep_chains: tuple[str, str], out_pdb: str) -> str:
    """Strip a design structure to just its two protein chains (target + binder),
    dropping any ligand/HETATM — AF2 is protein-only. Cached: skips if out_pdb
    already exists. Accepts .pdb or .cif. Returns out_pdb."""
    if os.path.isfile(out_pdb):
        return out_pdb
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
    parser = MMCIFParser(QUIET=True) if str(src).endswith(".cif") else PDBParser(QUIET=True)
    st = parser.get_structure("x", src)

    class _Sel(Select):
        def accept_chain(self, c):
            return c.id in keep_chains
        def accept_residue(self, r):
            return r.id[0] == " "  # standard residues only

    os.makedirs(os.path.dirname(out_pdb), exist_ok=True)
    io = PDBIO(); io.set_structure(st); io.save(out_pdb, _Sel())
    return out_pdb


def build_state_requests(df: pd.DataFrame, id_col: str, backbone_col: str,
                         target_chain: str, binder_chain: str, seq_col: str,
                         prep_dir: str, id_prefix: str) -> pd.DataFrame:
    """Build an AF2-IG requests frame for one state (holo or apo).

    Prepares (once, cached) each design's backbone stripped to target+binder,
    and pairs it with the design's binder sequence. `id` = f"{id_prefix}{id_col}"
    so holo/apo requests never collide and can be split back apart on return.
    Backbone prep is keyed on the backbone path so designs sharing a backbone
    reuse one stripped file.
    """
    os.makedirs(prep_dir, exist_ok=True)
    prepped: dict[str, str] = {}
    rows = []
    for _, r in df.iterrows():
        bb = r[backbone_col]
        if not isinstance(bb, str) or not os.path.isfile(bb):
            raise FileNotFoundError(f"AF2 request backbone does not exist for {r[id_col]}: {bb}")
        if bb not in prepped:
            stem = os.path.splitext(os.path.basename(bb))[0]
            cache_key = hashlib.sha256(
                f"{os.path.realpath(bb)}|{target_chain}|{binder_chain}".encode()
            ).hexdigest()[:12]
            prepped[bb] = prep_backbone(
                bb, (target_chain, binder_chain),
                os.path.join(prep_dir, f"{stem}_{cache_key}.pdb"),
            )
        chain_ids, chain_lengths = _protein_chain_lengths(prepped[bb])
        binder_sequence = _sequence_for_chain(r[seq_col], binder_chain, chain_ids)
        expected_length = chain_lengths.get(binder_chain)
        if expected_length is None:
            raise ValueError(
                f"AF2 request {r[id_col]} declares binder chain {binder_chain}, "
                f"but prepared backbone contains {chain_ids}"
            )
        if len(binder_sequence) != expected_length:
            raise ValueError(
                f"AF2 request {r[id_col]} binder sequence length {len(binder_sequence)} "
                f"does not match chain {binder_chain} backbone length {expected_length}"
            )
        rows.append({
            "id": f"{id_prefix}{r[id_col]}",
            "_orig_id": r[id_col],
            "backbone": prepped[bb],
            "target_chain": target_chain,
            "binder_chain": binder_chain,
            "seq": binder_sequence,
        })
    requests = pd.DataFrame(rows)
    if requests["id"].duplicated().any():
        raise RuntimeError("AF2 request identifiers must be unique")
    return requests


def run_af2_ig(requests: pd.DataFrame, work_dir: str, af2_cfg: dict, params_dir: str,
               max_concurrent: int = 20, poll: int = 30, save_pdb_dir: str | None = None) -> pd.DataFrame:
    """Score every row of `requests` with AF2 initial-guess.

    requests: DataFrame with columns id, backbone, target_chain, binder_chain, seq.
    Returns a DataFrame[id, plddt, i_pae, i_ptm, pdb_path] aligned to requests['id']
    (NaN or empty for any prediction that failed, or whose shard produced no output).

    save_pdb_dir: if given, each prediction's structure is persisted there (consensus tier
    consensus tier) at ~zero extra GPU cost (writes coordinates already computed,
    no re-inference) -- `pdb_path` is empty ("") when not requested.

    Resumable: if all shard outputs already exist for this work_dir and cover
    every id in `requests`, the sbatch array is skipped and results are
    collected directly. Coverage (not just shard COUNT) is checked: if a
    work_dir is reused across two calls whose id sets differ but happen to
    total the same number of requests (e.g. a design got renamed to avoid an
    id collision upstream, but the request count stayed the same), a
    count-only check would silently serve stale, wrong-id scores as NaN for
    every id absent from the existing shards.
    """
    if requests.empty:
        return pd.DataFrame(columns=["id", "plddt", "i_pae", "i_ptm", "pdb_path"])

    requests = requests.copy()
    requests["request_hash"] = requests.apply(
        lambda row: _request_hash(row, af2_cfg, params_dir), axis=1
    )
    ids = requests[["id", "request_hash"]].drop_duplicates()

    os.makedirs(work_dir, exist_ok=True)
    requests.to_csv(os.path.join(work_dir, "manifest.csv"), index=False)
    # How far per-backbone setup amortizes in THIS call. Setup-dominated passes
    # (state-2 designability at ~2 preds/backbone, the structure pass at 1)
    # cost several times more per prediction than the sequence gate at 16-32.
    n_bb = requests["backbone"].nunique() if "backbone" in requests.columns else len(requests)
    preds_per_backbone = (len(requests) / n_bb) if n_bb else 1.0

    shard_size = int(af2_cfg.get("shard_size", 60))
    # Cap the shard so its walltime stays schedulable. A big shard is only cheap
    # when setup amortizes; at 1-2 preds/backbone the same shard_size produces a
    # multi-hour request that both backfills badly and TIMEOUTs if the estimate is
    # off. Scale the cap by the amortization actually available.
    target_s = float(af2_cfg.get("shard_target_seconds", 2700))
    per_pred = _SETUP_SECONDS_PER_BACKBONE / max(1.0, preds_per_backbone) + _INFERENCE_SECONDS_PER_PRED
    affordable = max(20, int(target_s / per_pred))
    if affordable < shard_size:
        print(f"  AF2 gate: shard_size {shard_size} -> {affordable} "
              f"({preds_per_backbone:.1f} preds/backbone, setup-dominated)")
        shard_size = affordable
    n_shards = _write_shards(requests, os.path.join(work_dir, "shards"), shard_size)

    out_dir = os.path.join(work_dir, "out")

    def _collect() -> pd.DataFrame:
        parts = [pd.read_csv(p) for p in sorted(glob(os.path.join(out_dir, "shard_*.csv")))
                 if os.path.getsize(p) > 0]
        sc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["id", "request_hash", "plddt", "i_pae", "i_ptm", "pdb_path"]
        )
        if "request_hash" not in sc:
            sc["request_hash"] = ""
        return sc.drop_duplicates(subset=["id", "request_hash"], keep="last")

    have = len(glob(os.path.join(out_dir, "shard_*.csv")))
    scores = _collect() if have else pd.DataFrame(columns=["id", "request_hash", "plddt", "i_pae", "i_ptm", "pdb_path"])
    expected_keys = set(map(tuple, ids[["id", "request_hash"]].itertuples(index=False, name=None)))
    observed_keys = set(map(tuple, scores[["id", "request_hash"]].itertuples(index=False, name=None)))
    missing = expected_keys - observed_keys
    if have < n_shards or missing:
        if missing and have >= n_shards:
            print(f"  AF2 gate: {out_dir} has {have} shard file(s) matching the expected count "
                  f"but is missing/stale for {len(missing)}/{len(ids)} request fingerprints "
                  f"-- resubmitting")
        job_id = _submit_array(work_dir, n_shards, af2_cfg, params_dir, max_concurrent,
                               save_pdb_dir=save_pdb_dir, shard_size=shard_size,
                               preds_per_backbone=preds_per_backbone)
        print(f"  AF2 gate: submitted array {job_id} ({n_shards} shards x ~{shard_size} preds, "
              f"walltime {shard_walltime(shard_size, preds_per_backbone)}/shard)")
        _wait(job_id, poll=poll)
        scores = _collect()

    merged = ids.merge(scores, on=["id", "request_hash"], how="left", validate="one_to_one")
    metric_columns = ["plddt", "i_pae", "i_ptm"]
    valid = merged[metric_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    n_ok = int(valid.sum())
    print(f"  AF2 gate: collected {n_ok}/{len(merged)} scores")
    if not valid.all():
        failed_ids = merged.loc[~valid, "id"].astype(str).tolist()
        raise RuntimeError(
            f"AF2 execution produced invalid/missing metrics for {len(failed_ids)}/{len(merged)} "
            f"requests; examples={failed_ids[:10]}"
        )
    return merged
