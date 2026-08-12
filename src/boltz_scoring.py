"""Boltz-2 scoring: target-MSA precompute, per-state scoring, cheap switch proxy.

`BoltzEnv` carries the output and input directories, the two GPU jobstarters, the
Boltz runner and the target-MSA cache.

A single jobstarter is used for all states. SLURM TRES accounting on a
production run recorded every state scored here at `gres/gpuutil=0`,
`gres/gpumem=0` and 1-10 GB RSS, so all states belong in the lighter resource
envelope.

## Invariants

1. **Fresh side objects per state.** Each state is scored on its own `Poses`
   object built from binder-only FASTAs, never by chaining `boltz.run()` on a
   shared object. Under chaining, each apo or control prediction carried over the
   preceding state's target chain: Boltz re-reads the prior step's output
   structure and `BoltzParams.generate_yaml_files` appends the new target, so the
   apo YAML became [binder, holo_target, apo_target] and the apo ipTM was
   computed on a contaminated three-chain complex.

2. **Collapse to one row per key before merging.** Boltz can emit more than one
   model row per input pose. Without the collapse, merges become many-to-many and
   compound multiplicatively; 4 rows expanded to 262,144 in one observed case.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from glob import glob

import numpy as np
import pandas as pd
import yaml

from protflow.poses import Poses
from protflow.tools.boltz import BoltzParams

import seq_io
import structure_io


@dataclass
class BoltzEnv:
    """Everything the scorers used to capture from `main()`'s scope.

    `msa_map` is mutated in place by `precompute_target_msa` and read by
    `score_one_state`, so it is shared state by design — one cache per run.
    """
    outputs: str
    inputs: str
    boltz: object
    jobstarter: object
    final_samples: int = 1
    msa_map: dict = field(default_factory=dict)


# Target MSA precompute

def precompute_target_msa(env: BoltzEnv, target_seq: str, label: str) -> None:
    """Generate once and cache a Boltz MSA .csv for a target sequence.

    The natural target (PD-L1 / PCNA) has a deep, informative MSA that materially
    improves its predicted fold, and its sequence is identical across every design
    — so compute it ONCE and reuse. The de novo binder has no evolutionary
    homologs and is scored single-sequence, which is the standard choice for
    designed binders.

    Any failure is non-fatal: `msa_map` then omits this sequence and
    `score_one_state` falls back to on-the-fly server MSAs.
    """
    if target_seq in env.msa_map:
        return
    cache_dir = os.path.join(env.inputs, "msa_cache")
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.md5(target_seq.encode()).hexdigest()[:10]
    cached = os.path.join(cache_dir, f"{label}_{digest}.csv")
    if os.path.isfile(cached) and os.path.getsize(cached) > 0:
        env.msa_map[target_seq] = cached
        print(f"  target MSA for {label}: reusing cached {os.path.basename(cached)}")
        return
    try:
        yaml_dir = os.path.join(env.outputs, "msa_precompute", label)
        os.makedirs(yaml_dir, exist_ok=True)
        yaml_path = os.path.join(yaml_dir, f"{label}.yaml")
        # A pre-made Boltz YAML whose single protein chain has NO 'msa' key.
        # boltz.run passes an existing .yaml through untouched, and a chain
        # WITHOUT an msa field gets fetched from the server under
        # --use_msa_server. A chain written as 'msa: empty' (what a plain FASTA
        # pose becomes) SUPPRESSES the fetch instead, which is why an earlier
        # FASTA-based precompute produced an empty msa dir.
        seq_clean = "".join(str(target_seq).split())
        with open(yaml_path, "w") as fh:
            yaml.safe_dump(
                {"sequences": [{"protein": {"id": "A", "sequence": seq_clean}}]},
                fh, default_flow_style=False, sort_keys=False,
            )
        side = Poses(poses=[yaml_path], work_dir=env.outputs, jobstarter=env.jobstarter)
        env.boltz.run(poses=side, prefix=f"msa_precompute_{label}",
                      jobstarter=env.jobstarter,
                      options="--diffusion_samples 1 --no_kernels --use_msa_server")
        out_root = os.path.join(env.outputs, f"msa_precompute_{label}")
        hits = sorted(glob(os.path.join(out_root, "**", "msa", "*_0.csv"), recursive=True))
        if not hits:
            hits = sorted(glob(os.path.join(out_root, "**", "msa", "*.csv"), recursive=True))
        if not hits:
            print(f"  WARNING: no MSA .csv produced for {label}; falling back to server MSAs")
            return
        shutil.copyfile(hits[0], cached)
        env.msa_map[target_seq] = cached
        print(f"  target MSA for {label}: cached {os.path.basename(cached)} "
              f"(from {os.path.relpath(hits[0], env.outputs)})")
    except Exception as exc:                                   # non-fatal by design
        print(f"  WARNING: target MSA precompute failed for {label} "
              f"({type(exc).__name__}: {exc}); falling back to server MSAs")


# Generic side-object Boltz run

def run_side_boltz(env: BoltzEnv, file_list: list[str], prefix: str,
                   design_ids, params=None) -> pd.DataFrame:
    """Run Boltz on a fresh side `Poses`, returning one row per `_design_id`.

    `design_ids` is assigned POSITIONALLY onto the side frame, which is only
    correct because `Poses` preserves the order of an explicit input list (verified
    empirically against this ProtFlow version) and every caller builds `file_list`
    by iterating the same frame. That invariant is undocumented upstream, so it is
    asserted here rather than trusted: a silent misalignment would attach every
    score to the wrong design, which no downstream check would catch.
    """
    design_ids = list(design_ids)
    if len(file_list) != len(design_ids):
        raise ValueError(
            f"run_side_boltz({prefix}): {len(file_list)} input files vs "
            f"{len(design_ids)} design ids — positional _design_id assignment "
            "would misalign scores with designs")
    side = Poses(poses=list(file_list), work_dir=env.outputs, jobstarter=env.jobstarter)
    if list(side.df["poses"]) != list(file_list):
        raise RuntimeError(
            f"run_side_boltz({prefix}): Poses reordered the input list; "
            "positional _design_id assignment is no longer safe")
    side.df["_design_id"] = design_ids
    kwargs = dict(poses=side, prefix=prefix, jobstarter=env.jobstarter,
                  options="--diffusion_samples 1 --no_kernels")
    if params is not None:
        kwargs["params"] = params
    side = env.boltz.run(**kwargs)
    # Collapse to exactly one row per _design_id (best confidence) BEFORE
    # returning — see invariant 2 in the module docstring.
    df = side.df
    conf = f"{prefix}_confidence_score"
    if conf in df.columns:
        df = df.sort_values(conf, ascending=False)
    return df.drop_duplicates(subset="_design_id", keep="first")


# Per-state scoring

def score_one_state(env: BoltzEnv, fastas, keys, prefix: str, target_seq: str,
                    samples: int) -> pd.DataFrame:
    """Score one binder+target state; returns per-state columns keyed on `_state_key`."""
    side = Poses(poses=list(fastas), work_dir=env.outputs, jobstarter=env.jobstarter)
    side.df["_state_key"] = list(keys)
    side.df["_pre_boltz_id"] = side.df["poses_description"]

    params = BoltzParams()
    target_msa = env.msa_map.get(target_seq)
    if target_msa:
        params.add_protein(sequence=target_seq, id="B", msa=target_msa)
        msa_opt = ""                       # binder -> msa: empty, no server round trip
    else:
        params.add_protein(sequence=target_seq, id="B", msa=False)
        msa_opt = " --use_msa_server"

    side = env.boltz.run(poses=side, prefix=prefix, jobstarter=env.jobstarter,
                         params=params,
                         options=f"--diffusion_samples {samples}{msa_opt}")
    side = structure_io.collapse_to_best_model(side, metric_col=f"{prefix}_iptm")
    df = side.df

    # Binder-only pLDDT (chain A in Boltz's complex output), not the whole-complex
    # mean — the well-folded natural target would otherwise dominate the average
    # and inflate the pLDDT gate for a badly-folded binder. NaN-safe.
    if f"{prefix}_location" in df.columns and f"{prefix}_plddt_location" in df.columns:
        df[f"{prefix}_plddt_mean"] = df.apply(
            lambda r: structure_io.get_binder_plddt(
                r[f"{prefix}_location"], r[f"{prefix}_plddt_location"], "A"), axis=1)
    elif f"{prefix}_plddt_location" in df.columns:
        df[f"{prefix}_plddt_mean"] = df[f"{prefix}_plddt_location"].apply(
            structure_io.load_plddt_mean)

    # Interface PAE (binder chain A <-> target chain B), Bennett-style
    # pae_interaction. Computed for every scored state so ranking can gate on it
    # directly rather than leaving it an eval-only diagnostic. NaN-safe, so a state
    # missing its PAE npz is not gated.
    if f"{prefix}_location" in df.columns and f"{prefix}_pae_location" in df.columns:
        df[f"{prefix}_ipae"] = df.apply(
            lambda r: structure_io.get_interface_pae(
                r[f"{prefix}_location"], r[f"{prefix}_pae_location"], "A", "B"), axis=1)

    keep = ["_state_key"] + [c for c in df.columns if c.startswith(prefix)]
    return df[keep].drop_duplicates(subset="_state_key", keep="first")


def score_states(env: BoltzEnv, seq_df: pd.DataFrame, seq_col: str,
                 state_defs: list[tuple[str, str]], fasta_subdir: str,
                 samples: int | None = None) -> pd.DataFrame:
    """Score each (prefix, target_seq) state on independent binder-only side objects.

    Returns a frame keyed on 'poses_description' with all per-state Boltz columns
    merged 1:1. Formerly `score_four_state`; renamed because the ligand states are
    gone and a protein-only run scores exactly two.
    """
    # Validate the shape up front. These were 4-tuples in the ligand era
    # (prefix, seq, with_ligand, with_affinity). Validating up front matters
    # because a stale 4-tuple otherwise fails deep inside the loop, after the
    # first state has already been scored on the GPU.
    for i, spec in enumerate(state_defs):
        if not (isinstance(spec, (tuple, list)) and len(spec) == 2):
            raise ValueError(
                f"score_states: state_defs[{i}] must be (prefix, target_seq); got "
                f"{len(spec) if hasattr(spec, '__len__') else type(spec).__name__} "
                f"elements. The legacy 4-tuple (prefix, seq, with_ligand, "
                f"with_affinity) is no longer supported.")
    samples = samples if samples is not None else env.final_samples
    fastas, keys = seq_io.write_binder_only_fastas(
        seq_df, seq_col, os.path.join(env.outputs, fasta_subdir))
    out = pd.DataFrame({"_state_key": keys})
    for prefix, target_seq in state_defs:
        print(f"  scoring state '{prefix}' ({len(keys)} designs, no chain carry-over)")
        out = out.merge(
            score_one_state(env, fastas, keys, prefix, target_seq, samples),
            on="_state_key", how="left")
    return out.rename(columns={"_state_key": "poses_description"})


# No-MSA switch proxy

def cheap_switch_scores(env: BoltzEnv, seq_df: pd.DataFrame, seq_col: str,
                        prefix: str, subdir: str,
                        holo_seq: str, apo_seq: str) -> pd.DataFrame:
    """No-MSA holo+apo complex ipTM per binder-only sequence.

    Returns a frame keyed on 'poses_description' with {prefix}_holo_iptm,
    {prefix}_apo_iptm and {prefix}_switch_proxy. The proxy is the HARMONIC mean,
    matching the `switch_harmonic` the pipeline finally ranks by — a sum would let
    the per-backbone top-K selection discard a balanced two-state design in favour
    of a lopsided high-sum one before it ever reached expensive scoring.
    """
    def _score(target_seq: str, state: str) -> pd.DataFrame:
        # Write a 2-chain FASTA (binder:target) and let the Boltz YAML conversion
        # assign msa:empty to both protein chains. Adding the target via
        # add_protein(msa=False) instead leaves it with no MSA field and Boltz
        # aborts with "Missing MSA's ... and --use_msa_server flag not set".
        out_dir = os.path.join(env.outputs, subdir, f"{state}_fastas")
        os.makedirs(out_dir, exist_ok=True)
        paths, keys = [], []
        for _, row in seq_df.iterrows():
            binder = seq_io.require_single_chain_sequence(row[seq_col], seq_col)
            key = row["poses_description"]
            fasta_path = os.path.join(out_dir, f"{key}.fasta")
            with open(fasta_path, "w") as fh:
                fh.write(f">{key}\n{binder}:{target_seq}\n")
            paths.append(fasta_path)
            keys.append(key)

        side = Poses(poses=paths, work_dir=env.outputs, jobstarter=env.jobstarter)
        side.df["_cheap_key"] = keys
        side = env.boltz.run(poses=side, prefix=f"{prefix}_{state}",
                             jobstarter=env.jobstarter,
                             options="--diffusion_samples 1 --no_kernels")
        df = side.df
        conf = f"{prefix}_{state}_confidence_score"
        if conf in df.columns:
            df = df.sort_values(conf, ascending=False)
        df = df.drop_duplicates(subset="_cheap_key", keep="first")
        return df[["_cheap_key", f"{prefix}_{state}_iptm"]]

    holo_scores = _score(holo_seq, "holo")
    apo_scores = _score(apo_seq, "apo")
    merged = holo_scores.merge(apo_scores, on="_cheap_key", how="outer")
    holo_iptm = merged[f"{prefix}_holo_iptm"]
    apo_iptm = merged[f"{prefix}_apo_iptm"]
    merged[f"{prefix}_switch_proxy"] = np.where(
        (holo_iptm + apo_iptm) > 0,
        2 * holo_iptm * apo_iptm / (holo_iptm + apo_iptm), 0.0)
    return merged.rename(columns={"_cheap_key": "poses_description"})
