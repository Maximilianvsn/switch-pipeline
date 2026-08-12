"""Run setup: configuration, paths, jobstarters and tool runners in one context.

Run setup produces approximately 45 values required by later stages.
`PipelineContext` makes that set explicit and typed, so that a stage function
takes a single argument rather than reading names from an enclosing scope.

Fields retain upper-case names where they correspond directly to configuration
constants, so that a field can be traced to the key that sets it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from protflow.jobstarters import SbatchArrayJobstarter
from protflow.tools.rfdiffusion3 import RFdiffusion3
from protflow.tools.ligandmpnn import LigandMPNN
from protflow.tools.boltz import Boltz
from protflow.tools.dynamicmpnn import DynamicMPNN

import af2_gate
import boltz_scoring
import funnel as funnel_mod
import pipeline_config
import switch_gating


# Bumped on any change that must invalidate a resumed run's cached provenance.
PIPELINE_VERSION = "2026.07.24"


@dataclass
class PipelineContext:
    """Every value the pipeline stages need, resolved once at startup."""
    ADAPTIVE_MAX_NSEQ: Any = None
    ADAPTIVE_TARGET: Any = None
    AF2_ENABLED: Any = None
    AF2_GATE_ONLY: Any = None
    AF2_PARAMS_DIR: Any = None
    APO_BATCH: Any = None
    DECOY_TARGETS: Any = None
    DIFFUSION_BATCH_SIZE: Any = None
    DMPNN_NSEQ: Any = None
    DMPNN_OPTIONS: Any = None
    HOLO_N_BATCHES: Any = None
    INPUTS: Any = None
    LMPNN_NSEQ: Any = None
    LMPNN_TOP_K: Any = None
    MAX_BACKBONES: Any = None
    MPNN_MSD_NSEQ: Any = None
    OUTPUTS: Any = None
    POST_AF2_PER_BACKBONE: Any = None
    POST_AF2_TOP_K: Any = None
    POST_DMPNN_TOP_K: Any = None
    PROXY_MPNN_MODEL: Any = None
    SELFCONS_IPTM_THRESHOLD: Any = None
    SELFCONS_PLDDT_THRESHOLD: Any = None
    SELFCONS_RMSD_DECOY_THRESHOLD: Any = None
    SPECIFICITY_ENABLED: Any = None
    SPECIFICITY_MARGIN_THRESHOLD: Any = None
    SPECIFICITY_MAX_CANDIDATES: Any = None
    WS: Any = None
    af2_cfg: Any = None
    binder_chain: Any = None
    apo: Any = None
    args: Any = None
    boltz_env: Any = None
    cfg: Any = None
    cpu_jst: Any = None
    cpu_jst_fast: Any = None
    dynamicmpnn: Any = None
    funnel: Any = None
    gpu_jst: Any = None
    holo: Any = None
    holo_pdb_path: Any = None
    ligandmpnn: Any = None
    rfd3: Any = None
    sp: Any = None

def build_context() -> PipelineContext:
    """Parse args, load+validate config, resolve paths, build jobstarters and
    tool runners, and open the funnel log. Returns the populated context.
    """
    args = pipeline_config.parse_args()
    if args.smoke and args.geometry_calibration:
        raise ValueError("--smoke and --geometry-calibration are mutually exclusive")

    WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUTS = os.path.join(WS, "inputs")

    default_run_name = (
        "geometry_calibration" if args.geometry_calibration
        else "smoke" if args.smoke
        else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    run_name = args.run_name or default_run_name
    OUTPUTS = os.path.join(WS, "outputs", run_name)
    os.makedirs(OUTPUTS, exist_ok=True)
    print(f"Run name: {run_name}")
    print(f"Outputs:  {OUTPUTS}")

    cfg = pipeline_config.load_config(pipeline_config.resolve_path(args.config, WS), WS)
    pipeline_config.validate_config(cfg)

    # ProtFlow caches are keyed by run directory. Refuse to silently reuse cached
    # structures or scores across a configuration change, or across an intentional
    # cache-invalidating code change (signalled by bumping PIPELINE_VERSION).
    # Keying on config + version rather than on source hashes means an ordinary
    # bugfix does not block a resume.
    run_mode = (
        "geometry_calibration" if args.geometry_calibration
        else "smoke" if args.smoke else "production"
    )
    provenance = {
        "mode": run_mode,
        "config_path": os.path.abspath(pipeline_config.resolve_path(args.config, WS)),
        "config": cfg,
        "pipeline_version": PIPELINE_VERSION,
    }
    provenance_path = os.path.join(OUTPUTS, "run_provenance.json")
    if os.path.isfile(provenance_path):
        with open(provenance_path) as provenance_handle:
            existing_provenance = json.load(provenance_handle)
        if existing_provenance != provenance:
            raise RuntimeError(
                f"Run directory {OUTPUTS} was created by a different config or code revision. "
                "Use a new --run-name to avoid stale ProtFlow/AF2 cache reuse."
            )
    else:
        with open(provenance_path, "w") as provenance_handle:
            json.dump(provenance, provenance_handle, indent=2, sort_keys=True)

    holo = cfg["holo_target"]
    apo = cfg["apo_target"]
    PROXY_MPNN_MODEL = "protein_mpnn"

    if args.geometry_calibration:
        sp = cfg.get("geometry_calibration", {})
    elif args.smoke:
        sp = cfg.get("smoke", {})
    else:
        sp = cfg.get("sampling", {})
    fp = cfg.get("smoke", {}) if args.smoke else cfg.get("filtering", {})

    HOLO_BATCH = sp.get("holo_batch", 200)
    APO_BATCH = sp.get("apo_batch", 5)
    # diffusion_batch_size is a per-GPU-call batch, memory-bound, not a total
    # count. RFD3's total per input pose = n_batches x diffusion_batch_size.
    # Cramming holo_batch=200 into diffusion_batch_size tried to diffuse 200
    # structures in one batch and exhausted the memory of a 44 GiB A100. Keep the per-call batch
    # small and reach HOLO_BATCH via n_batches (sequential, one GPU job).
    DIFFUSION_BATCH_SIZE = sp.get("diffusion_batch_size", 10)
    DIFFUSION_BATCH_SIZE = max(1, min(DIFFUSION_BATCH_SIZE, HOLO_BATCH))
    HOLO_N_BATCHES = max(1, -(-HOLO_BATCH // DIFFUSION_BATCH_SIZE))  # ceil div
    LMPNN_NSEQ = sp.get("lmpnn_nseq", 50)
    DMPNN_NSEQ = sp.get("dmpnn_nseq", 20)
    MPNN_MSD_NSEQ = sp.get("mpnn_msd_nseq", 20)
    # Adaptive resampling: if a backbone's initial DMPNN_NSEQ designs yield fewer
    # than ADAPTIVE_TARGET AF2-gate passes (relaxed or strict), top up that
    # backbone with more DynamicMPNN sequences (cheap, CPU) up to
    # ADAPTIVE_MAX_NSEQ total, so the fixed expensive Boltz budget downstream is
    # filled with real survivors rather than padded with under-sampled duds.
    # A target of 0 disables the top-up, giving a single fixed nseq.
    ADAPTIVE_TARGET = sp.get("adaptive_target", 0)
    ADAPTIVE_MAX_NSEQ = sp.get("adaptive_max_nseq", DMPNN_NSEQ)
    # DynamicMPNN sampling temperature (Hydra key model.temperature; the tool's
    # built-in default is 0.1). Higher = more sequence diversity per design =
    # more exploration to find one sequence that co-satisfies both states. Passed
    # through the runner's `options` Hydra passthrough; None leaves the tool
    # default untouched (no override emitted).
    DMPNN_TEMPERATURE = sp.get("dmpnn_temperature", None)
    DMPNN_OPTIONS = f"model.temperature={DMPNN_TEMPERATURE}" if DMPNN_TEMPERATURE is not None else ""
    FINAL_BOLTZ_SAMPLES = sp.get("final_boltz_samples", 3)
    LMPNN_TOP_K = fp.get("lmpnn_top_k", 1)
    SELFCONS_PLDDT_THRESHOLD = fp.get("selfcons_plddt_threshold", 0)
    SELFCONS_IPTM_THRESHOLD = fp.get("selfcons_iptm_threshold", 0)
    SELFCONS_RMSD_DECOY_THRESHOLD = fp.get("selfcons_rmsd_decoy_threshold", 0)
    MAX_BACKBONES = fp.get("max_backbones", 40)
    POST_DMPNN_TOP_K = fp.get("post_dmpnn_top_k", 0)
    POST_AF2_TOP_K = fp.get("post_af2_top_k", 180)
    POST_AF2_PER_BACKBONE = fp.get("post_af2_per_backbone", 0)

    # AF2 initial-guess gate (the discriminative, orthogonal selection)
    af2_cfg = cfg.get("af2", {})
    AF2_ENABLED = bool(af2_cfg.get("enabled", False))
    # gate_only: stop after the AF2 gate (skip the expensive four-state Boltz
    # scoring / scramble / MSD). Lets a smoke test exercise the full generate->
    # gate path cheaply without launching the ~180-design Boltz stage. Selectable
    # per-config (smoke sets it true).
    # gate_only can be set in the af2 section (production) or in the smoke/
    # sampling block (so --smoke stops after the gate without a separate flag).
    AF2_GATE_ONLY = bool(af2_cfg.get("gate_only", False)) or bool(sp.get("gate_only", False))
    AF2_PARAMS_DIR = pipeline_config.resolve_path(af2_cfg.get("params_dir", ""), WS) if af2_cfg else ""

    # Specificity / negative-target control
    spec_cfg = cfg.get("specificity", {})
    SPECIFICITY_ENABLED = bool(spec_cfg.get("enabled", False))
    SPECIFICITY_MARGIN_THRESHOLD = spec_cfg.get("margin_threshold", 0.15)
    SPECIFICITY_MAX_CANDIDATES = spec_cfg.get("max_candidates", 30)
    DECOY_TARGETS = spec_cfg.get("decoy_targets", []) or []
    for _dt in DECOY_TARGETS:
        if "sequence" in _dt:
            _dt["sequence"] = "".join(str(_dt["sequence"]).split())

    # Display labels (cosmetic only — no pipeline logic keys off these).
    # Written to labels.json so protein_only_evaluation.py can pick them up and
    # relabel plots/summaries for whatever target pair this config targets.
    lbl = cfg.get("labels", {})
    holo_name = lbl.get("holo_name", "holo")
    apo_name = lbl.get("apo_name", "apo")
    # Two-state protein-only labels. The ON/OFF distinction was ligand-conditional;
    # without a cofactor each target simply names one state.
    state_labels = {
        "holo_on": f"{holo_name} (state A)",
        "holo_off": f"{holo_name} (state A)",
        "apo_on": f"{apo_name} (state B)",
        "apo_off": f"{apo_name} (state B)",
        "holo_name": holo_name,
        "apo_name": apo_name,
        "ligand_name": None,
        "holo_hotspots": "",
        "holo_contig_start": None,
    }
    with open(os.path.join(OUTPUTS, "labels.json"), "w") as f:
        json.dump(state_labels, f, indent=2)

    funnel = funnel_mod.FunnelTracker(OUTPUTS)

    # Build holo input PDB (protein + ligand, ligand docked to hotspots)
    holo_pdb_path = holo["pdb"]

    # JobStarters
    # max_cores controls how many array tasks run concurrently. gpu-single
    # has 61 independent nodes, permitting broad parallelisation: each task
    # still only requests 1 GPU via --gres=gpu:1. Without this, every
    # prediction in a step (e.g. 10-30+ Boltz calls) runs sequentially on
    # a single GPU, which costs a ~2h wall-clock for a step whose
    # individual jobs each took ~10-15 min.
    # Sized from real SLURM TRES accounting on a completed run (each ProtFlow
    # array task = exactly one design, confirmed via jobstarters.py's
    # `sbatch -a 1-len(cmds)` -- no per-task batching to account for here):
    # ligand+affinity Boltz calls used ~12.7GB/~2min max; RFdiffusion3 used
    # ~8.1GB/~1:45 at smoke's diffusion_batch_size=2 (production's batch=10
    # generates more structures per task, so memory/time here keep more
    # headroom than the Boltz-only numbers alone would justify). Down from
    # 4h/32GB, which no observed task came within 4x of using.
    # 01:00:00 was too tight for RFD3 and SILENTLY TRUNCATED a production run:
    # prod_separate_20260802 asked for 800 backbones/state and got 510 and 220,
    # because ProtFlow splits n_batches across only ~2 array tasks, so one task
    # carries ~400 designs and lands right at the hour (job 14116045_1: TIMEOUT
    # at 01:00:02). RFD3 is killed mid-generation rather than failing, so the
    # run continues on a short backbone set and nothing in the funnel says why.
    # 4h gives ~4x headroom on the measured throughput; the cost is only queue
    # priority, whereas the cost of the old value was a quietly under-powered run.
    gpu_jst = SbatchArrayJobstarter(
        max_cores=20,
        gpus=True,
        options="--partition=gpu-single --time=04:00:00 --mem=24G --gres=gpu:1",
    )
    # No-MSA, single-diffusion-sample Boltz calls (Step 3.5B) skip the MSA
    # server round trip and the MSA-memory footprint entirely, so they run
    # in a couple minutes rather than the 10-15 min/job the full pipeline
    # was sized for. Requesting the same 4h/32GB envelope for these anyway
    # does not make them faster and claims more resource than required;
    # which on a fairshare scheduler costs queue priority for no benefit.
    # Also used for no-ligand/no-affinity four-state Boltz calls (apo states,
    # no-ligand controls, decoy/scramble mirrors of both) -- measured via
    # gres/gpuutil+gres/gpumem TRES accounting at literally 0 on a real run
    # (against 30-64% for ligand and affinity states), confirming the same
    # class of cheap call, just routed through gpu_jst by default before.
    gpu_jst_fast = SbatchArrayJobstarter(
        max_cores=20,
        gpus=True,
        options="--partition=gpu-single --time=00:30:00 --mem=16G --gres=gpu:1",
    )
    cpu_jst = SbatchArrayJobstarter(
        max_cores=20,
        gpus=False,
        options="--partition=cpu-single --time=02:00:00 --mem-per-cpu=8G",
    )
    # LigandMPNN/DynamicMPNN generate ALL of their nseq sequences for a
    # backbone within one array task (unlike Boltz's one-task-per-design
    # pattern), and production's nseq is 6-25x smoke's -- real usage measured
    # at smoke scale (<1.1GB, <30s) does not safely extrapolate to production
    # for those tools, so cpu_jst above is left untouched for them.
    # RMSD calls are different: make_backbone_rmsd/BackboneRMSD.run() submits
    # one array task per structural comparison regardless of upstream config
    # (same `sbatch -a 1-len(cmds)` pattern as Boltz), so its real footprint
    # (<150MB, <31s measured across consensus + late-structure self-consistency)
    # generalizes safely to production's much larger RMSD call volume.
    cpu_jst_fast = SbatchArrayJobstarter(
        max_cores=20,
        gpus=False,
        options="--partition=cpu-single --time=00:15:00 --mem-per-cpu=1G",
    )

    # Runners
    rfd3 = RFdiffusion3()
    ligandmpnn = LigandMPNN()
    boltz = Boltz()

    # Configurable like the other three tool envs (proteinmpnn_msd.python,
    # af2.bindcraft_env), so a differently-named DynamicMPNN env needs no code
    # change.
    dmpnn_cfg = cfg.get("dynamicmpnn", {})
    dmpnn_env = dmpnn_cfg.get("conda_env", "dynamicmpnn")
    dynamicmpnn = DynamicMPNN(
        pre_cmd=f'eval "$(conda shell.bash hook)" && conda activate {dmpnn_env}',
        checkpoint=dmpnn_cfg.get("checkpoint"),
    )

    # Cache of precomputed target MSAs (target_sequence -> Boltz MSA .csv path),
    # populated once before the final scoring stages (Step 6). Empty until then,
    # so the no-MSA triage stages (3.5B, 5.5) are unaffected. Referenced as a
    # closure by _score_one_state above.
    boltz_env = boltz_scoring.BoltzEnv(
        outputs=OUTPUTS, inputs=INPUTS, boltz=boltz, jobstarter=gpu_jst_fast,
        final_samples=FINAL_BOLTZ_SAMPLES,
    )


    # Needed by the designability pre-filter (right after Step 1) as well as
    # Step 3 onward, so defined once here rather than inside Step 2.
    binder_chain = holo.get("binder_chain", "B")

    return PipelineContext(
        ADAPTIVE_MAX_NSEQ=ADAPTIVE_MAX_NSEQ,
        ADAPTIVE_TARGET=ADAPTIVE_TARGET,
        AF2_ENABLED=AF2_ENABLED,
        AF2_GATE_ONLY=AF2_GATE_ONLY,
        AF2_PARAMS_DIR=AF2_PARAMS_DIR,
        APO_BATCH=APO_BATCH,
        DECOY_TARGETS=DECOY_TARGETS,
        DIFFUSION_BATCH_SIZE=DIFFUSION_BATCH_SIZE,
        DMPNN_NSEQ=DMPNN_NSEQ,
        DMPNN_OPTIONS=DMPNN_OPTIONS,
        HOLO_N_BATCHES=HOLO_N_BATCHES,
        INPUTS=INPUTS,
        LMPNN_NSEQ=LMPNN_NSEQ,
        LMPNN_TOP_K=LMPNN_TOP_K,
        MAX_BACKBONES=MAX_BACKBONES,
        MPNN_MSD_NSEQ=MPNN_MSD_NSEQ,
        OUTPUTS=OUTPUTS,
        POST_AF2_PER_BACKBONE=POST_AF2_PER_BACKBONE,
        POST_AF2_TOP_K=POST_AF2_TOP_K,
        POST_DMPNN_TOP_K=POST_DMPNN_TOP_K,
        PROXY_MPNN_MODEL=PROXY_MPNN_MODEL,
        SELFCONS_IPTM_THRESHOLD=SELFCONS_IPTM_THRESHOLD,
        SELFCONS_PLDDT_THRESHOLD=SELFCONS_PLDDT_THRESHOLD,
        SELFCONS_RMSD_DECOY_THRESHOLD=SELFCONS_RMSD_DECOY_THRESHOLD,
        SPECIFICITY_ENABLED=SPECIFICITY_ENABLED,
        SPECIFICITY_MARGIN_THRESHOLD=SPECIFICITY_MARGIN_THRESHOLD,
        SPECIFICITY_MAX_CANDIDATES=SPECIFICITY_MAX_CANDIDATES,
        WS=WS,
        af2_cfg=af2_cfg,
        binder_chain=binder_chain,
        apo=apo,
        args=args,
        boltz_env=boltz_env,
        cfg=cfg,
        cpu_jst=cpu_jst,
        cpu_jst_fast=cpu_jst_fast,
        dynamicmpnn=dynamicmpnn,
        funnel=funnel,
        gpu_jst=gpu_jst,
        holo=holo,
        holo_pdb_path=holo_pdb_path,
        ligandmpnn=ligandmpnn,
        rfd3=rfd3,
        sp=sp,
    )
