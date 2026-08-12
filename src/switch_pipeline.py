"""Two-state de novo protein binder design pipeline.

Generates a state-1 binder backbone, derives state 2 from it by partial
diffusion, designs one sequence tied across both states, and selects candidates
against paired composition-matched controls. All target-specific parameters are
read from a YAML configuration, so the pipeline applies unchanged to any pair of
protein targets.

Usage:
    conda activate protflow
    python src/switch_pipeline.py --config configs/pdl1_pcna_protein_only.yaml --run-name prod_v1 [--smoke]

Resuming:
    Re-running under an existing --run-name resumes from the cached ProtFlow
    scorefiles; a new --run-name starts an independent run.
"""
import os
import re
import sys
import json
import hashlib
import argparse
import subprocess
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from protflow.poses import Poses
from protflow.jobstarters import SbatchArrayJobstarter
from protflow.tools.rfdiffusion3 import RFdiffusion3, RFD3Params
from protflow.tools.ligandmpnn import LigandMPNN
from protflow.tools.boltz import Boltz, BoltzParams
from protflow.tools.dynamicmpnn import DynamicMPNN
from protflow.metrics.rmsd import BackboneRMSD
from protflow import load_config_path, require_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from af2_runner import run_af2_ig, build_state_requests
import switch_gating
import seq_io
import funnel as funnel_mod
import structure_io
import pipeline_config
import pipeline_context
import stages
import boltz_scoring
import af2_gate
import state_pairing
import paired_nulls
import method_comparison
import boltz_interface_evaluation

# Re-exported for the runbooks, which reference the version through this module.
# The run-provenance guard keys on this value together with the configuration,
# so it is incremented only when a change invalidates cached ProtFlow or AF2
# artefacts, for example a change to how a scored quantity is computed.
from pipeline_context import PIPELINE_VERSION


# Helpers


def run_proteinmpnn_msd(
    poses_df: pd.DataFrame,
    holo_col: str,
    apo_col: str,
    binder_chain_holo: str,
    binder_chain_apo: str,
    work_dir: str,
    python_path: str,
    script_path: str,
    parse_script: str,
    weights_dir: str,
    num_seqs: int = 20,
) -> pd.DataFrame:
    """Run ProteinMPNN multi-state design with tied positions.

    For each design pair, the binder chain of both states is extracted into a
    single two-chain PDB (chain A, holo binder; chain B, apo binder), and
    ProteinMPNN is run with homooligomer=1 to tie all positions across the two
    chains, constraining the design to one sequence compatible with both
    backbone conformations.
    """
    os.makedirs(work_dir, exist_ok=True)
    seq_dir = os.path.join(work_dir, "seqs")
    os.makedirs(seq_dir, exist_ok=True)

    rows = []
    for _, row in poses_df.iterrows():
        design_name = row["poses_description"]
        design_dir = os.path.join(work_dir, design_name)
        pdb_dir = os.path.join(design_dir, "pdbs")
        os.makedirs(pdb_dir, exist_ok=True)

        holo_pdb = os.path.abspath(row[holo_col])
        apo_pdb = os.path.abspath(row[apo_col])

        # Extract binder chains as A and B into one combined PDB
        holo_binder_pdb = os.path.join(design_dir, "holo_binder_A.pdb")
        apo_binder_pdb = os.path.join(design_dir, "apo_binder_B.pdb")
        combined_pdb = os.path.join(pdb_dir, f"{design_name}_msd.pdb")

        if not os.path.exists(combined_pdb):
            structure_io._extract_binder_to_chain(holo_pdb, binder_chain_holo, "A", holo_binder_pdb)
            structure_io._extract_binder_to_chain(apo_pdb, binder_chain_apo, "B", apo_binder_pdb)

            with open(holo_binder_pdb) as f:
                lines_a = [l for l in f if not l.startswith("END")]
            with open(apo_binder_pdb) as f:
                lines_b = list(f)
            with open(combined_pdb, "w") as f:
                f.writelines(lines_a)
                f.writelines(lines_b)

        parsed_jsonl = os.path.join(design_dir, "parsed.jsonl")
        tied_jsonl = os.path.join(design_dir, "tied.jsonl")
        out_dir = os.path.join(design_dir, "output")

        # Parse combined PDB
        subprocess.run([
            python_path, parse_script,
            "--input_path", pdb_dir,
            "--output_path", parsed_jsonl,
        ], check=True, capture_output=True)

        # Tied positions in homooligomer mode: residues are tied 1:1 across
        # chains A and B.
        tied_dict = {}
        with open(parsed_jsonl) as f:
            import json as _json
            for line in f:
                entry = _json.loads(line)
                name = entry["name"]
                seq_a = entry.get("seq_chain_A", "")
                n_res = len(seq_a)
                tied_list = [{"A": [i + 1], "B": [i + 1]} for i in range(n_res)]
                tied_dict[name] = tied_list
        with open(tied_jsonl, "w") as f:
            f.write(_json.dumps(tied_dict) + "\n")

        # ProteinMPNN designs chains A and B under the tied positions.
        subprocess.run([
            python_path, script_path,
            "--jsonl_path", parsed_jsonl,
            "--tied_positions_jsonl", tied_jsonl,
            "--out_folder", out_dir,
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", "0.1",
            "--seed", "42",
            "--batch_size", "1",
            "--path_to_model_weights", weights_dir,
        ], check=True, capture_output=True)

        # Collect output FASTAs
        seq_idx = 0
        for fa in sorted(glob(os.path.join(out_dir, "seqs", "*.fa"))):
            with open(fa) as fh:
                lines = fh.readlines()
            i = 0
            first_record = True
            while i < len(lines):
                if not lines[i].startswith(">"):
                    i += 1
                    continue
                header = lines[i].strip().lstrip(">")
                i += 1
                seq_parts = []
                while i < len(lines) and not lines[i].startswith(">"):
                    seq_parts.append(lines[i].strip())
                    i += 1
                sequence = "".join(seq_parts)
                if first_record:
                    # The first record is the input reference sequence.
                    first_record = False
                    continue
                if not sequence:
                    continue
                # Take only chain A sequence (first half, tied to B)
                binder_seq = sequence.split("/")[0] if "/" in sequence else sequence
                seq_idx += 1
                # ProteinMPNN writes sampling metadata into the FASTA header
                # ("T=0.1, sample=1, score=1.11, ..."); the spaces, equals signs
                # and commas are not valid in a pose identifier, filename or
                # Boltz description. A sequential identifier is assigned instead
                # and the score is parsed out separately.
                name = f"{design_name}_msd{seq_idx:03d}"
                m = re.search(r"\bscore=([0-9.]+)", header)
                mpnn_score = float(m.group(1)) if m else float("nan")
                fa_out = os.path.join(seq_dir, f"{name}.fa")
                with open(fa_out, "w") as fw:
                    fw.write(f">{name}\n{binder_seq}\n")
                rows.append({
                    "description": name,
                    "location": fa_out,
                    "sequence": binder_seq,
                    "method": "proteinmpnn_msd",
                    "mpnn_score": mpnn_score,
                    # Lineage to the originating backbone pair, required to join
                    # against the DynamicMPNN results for the per-backbone method
                    # comparison (see protein_only_evaluation.py).
                    "backbone": row.get("s1_rfd3_holo_description", design_name),
                })

    if not rows:
        raise FileNotFoundError(f"No ProteinMPNN-MSD outputs in {work_dir}")
    return pd.DataFrame(rows)


def main():
    ctx = pipeline_context.build_context()

    # Unpacked into locals for the inline stages below. Stages extracted from
    # here take `ctx` directly and omit this unpacking.
    ADAPTIVE_MAX_NSEQ, ADAPTIVE_TARGET, AF2_ENABLED, AF2_GATE_ONLY = ctx.ADAPTIVE_MAX_NSEQ, ctx.ADAPTIVE_TARGET, ctx.AF2_ENABLED, ctx.AF2_GATE_ONLY
    AF2_PARAMS_DIR, APO_BATCH, DECOY_TARGETS, DIFFUSION_BATCH_SIZE = ctx.AF2_PARAMS_DIR, ctx.APO_BATCH, ctx.DECOY_TARGETS, ctx.DIFFUSION_BATCH_SIZE
    DMPNN_NSEQ, DMPNN_OPTIONS, HOLO_N_BATCHES, INPUTS = ctx.DMPNN_NSEQ, ctx.DMPNN_OPTIONS, ctx.HOLO_N_BATCHES, ctx.INPUTS
    LMPNN_NSEQ, LMPNN_TOP_K, MAX_BACKBONES, MPNN_MSD_NSEQ = ctx.LMPNN_NSEQ, ctx.LMPNN_TOP_K, ctx.MAX_BACKBONES, ctx.MPNN_MSD_NSEQ
    OUTPUTS, POST_AF2_PER_BACKBONE, POST_AF2_TOP_K, POST_DMPNN_TOP_K = ctx.OUTPUTS, ctx.POST_AF2_PER_BACKBONE, ctx.POST_AF2_TOP_K, ctx.POST_DMPNN_TOP_K
    PROXY_MPNN_MODEL, SELFCONS_IPTM_THRESHOLD, SELFCONS_PLDDT_THRESHOLD, SELFCONS_RMSD_DECOY_THRESHOLD = ctx.PROXY_MPNN_MODEL, ctx.SELFCONS_IPTM_THRESHOLD, ctx.SELFCONS_PLDDT_THRESHOLD, ctx.SELFCONS_RMSD_DECOY_THRESHOLD
    SPECIFICITY_ENABLED, SPECIFICITY_MARGIN_THRESHOLD, SPECIFICITY_MAX_CANDIDATES, WS = ctx.SPECIFICITY_ENABLED, ctx.SPECIFICITY_MARGIN_THRESHOLD, ctx.SPECIFICITY_MAX_CANDIDATES, ctx.WS
    af2_cfg, apo, args, boltz_env = ctx.af2_cfg, ctx.apo, ctx.args, ctx.boltz_env
    cfg, cpu_jst, cpu_jst_fast, dynamicmpnn = ctx.cfg, ctx.cpu_jst, ctx.cpu_jst_fast, ctx.dynamicmpnn
    funnel, gpu_jst, holo, holo_pdb_path = ctx.funnel, ctx.gpu_jst, ctx.holo, ctx.holo_pdb_path
    ligandmpnn, rfd3, sp = ctx.ligandmpnn, ctx.rfd3, ctx.sp
    binder_chain = ctx.binder_chain

    print("\n[Stage 1] RFdiffusion3 holo diffusion")

    poses = Poses(
        poses=[holo_pdb_path],
        work_dir=OUTPUTS,
        jobstarter=gpu_jst,
    )

    params_holo = RFD3Params(poses=poses)
    rfd3_kwargs = dict(
        contig=holo["contig"],
        select_hotspots=holo["hotspots"],
    )
    params_holo.set_input_specs(**rfd3_kwargs)

    # n_batches x diffusion_batch_size gives HOLO_BATCH backbones in total,
    # generated in memory-bounded sub-batches. Both are runner keyword
    # arguments; passing them in the options string duplicates the argument.
    poses = rfd3.run(
        poses=poses,
        prefix="s1_rfd3_holo",
        params=params_holo,
        n_batches=HOLO_N_BATCHES,
        diffusion_batch_size=DIFFUSION_BATCH_SIZE,
        options="skip_existing=True",
    )
    funnel.log("s1_rfd3_holo", len(poses.df),
               f"holo backbones ({HOLO_N_BATCHES} x {DIFFUSION_BATCH_SIZE})")

    # AF2 designability pre-filter. Undesignable backbones are removed before
    # the apo partial diffusion of stage 2 and all downstream stages. Design
    # success is a per-backbone property that additional sequence sampling does
    # not recover, so backbone count is the primary diversity lever; this filter
    # is what makes a large HOLO_BATCH affordable. LigandMPNN proxy sequences are
    # scored by AF2 initial-guess on the un-expanded holo backbones, and only
    # backbones whose best proxy design clears min_plddt are retained.
    desig_cfg = af2_cfg.get("designability", {})
    survivor_stems = None  # None => generate_apo_inputs processes every backbone (unfiltered)
    if AF2_ENABLED and desig_cfg.get("enabled", False):
        print("\n[Stage 1.5] AF2 designability pre-filter, holo backbones")

        n_seqs_d = int(sp.get("designability_n_seqs", desig_cfg.get("n_seqs", 2)))
        pre_af2_top_k_d = int(
            sp.get("designability_pre_af2_top_k", desig_cfg.get("pre_af2_top_k", n_seqs_d))
        )
        pre_af2_top_k_d = max(1, min(pre_af2_top_k_d, n_seqs_d))
        min_plddt_d = float(
            sp.get("designability_min_plddt", desig_cfg.get("min_plddt", 0.70))
        )
        max_i_pae_d = float(
            sp.get("designability_max_i_pae", desig_cfg.get("max_i_pae", 1.0))
        )
        min_i_ptm_d = float(
            sp.get("designability_min_i_ptm", desig_cfg.get("min_i_ptm", 0.0))
        )

        desig_poses = Poses(poses=poses.df["s1_rfd3_holo_location"].tolist(),
                            work_dir=OUTPUTS, jobstarter=cpu_jst)
        # Recorded before the ProteinMPNN expansion so that identity and
        # backbone paths propagate to every proxy sequence.
        desig_poses.df["_desig_backbone_pdb"] = poses.df["s1_rfd3_holo_location"].values
        desig_poses.df["_desig_backbone_id"] = poses.df["s1_rfd3_holo_description"].values

        desig_poses = ligandmpnn.run(
            poses=desig_poses, prefix="s1_5_desig_lmpnn", jobstarter=cpu_jst,
            nseq=n_seqs_d, model_type=PROXY_MPNN_MODEL,
            options=f"--chains_to_design {binder_chain} ",
        )
        # ProteinMPNN is inexpensive relative to AF2: sequences are
        # over-generated, and only a fixed-size, highest-confidence shortlist per
        # backbone is predicted.
        proxy_conf_col = "s1_5_desig_lmpnn_overall_confidence"
        desig_poses.df = (
            desig_poses.df.sort_values(
                ["_desig_backbone_id", proxy_conf_col, "poses_description"],
                ascending=[True, False, True],
            )
            .groupby("_desig_backbone_id", group_keys=False)
            .head(pre_af2_top_k_d)
            .reset_index(drop=True)
        )
        desig_reqs = build_state_requests(
            desig_poses.df, "poses_description", "_desig_backbone_pdb", "A", binder_chain,
            "s1_5_desig_lmpnn_sequence", os.path.join(OUTPUTS, "s1_5_desig_af2", "bb"), "desig__",
        )
        desig_scores = run_af2_ig(desig_reqs, os.path.join(OUTPUTS, "s1_5_desig_af2", "af2"),
                                  af2_cfg, AF2_PARAMS_DIR)
        desig_scores = desig_scores.merge(desig_reqs[["id", "_orig_id"]], on="id", how="left")
        desig_poses.df = desig_poses.df.merge(
            desig_scores.rename(columns={"_orig_id": "poses_description"})[
                ["poses_description", "plddt", "i_pae", "i_ptm"]
            ],
            on="poses_description",
            how="left",
        )
        desig_poses.df["_individual_designability_score"] = (
            pd.to_numeric(desig_poses.df["plddt"], errors="coerce")
            * np.sqrt(
                pd.to_numeric(desig_poses.df["i_ptm"], errors="coerce").clip(lower=0)
                * np.exp(
                    -3.1
                    * pd.to_numeric(desig_poses.df["i_pae"], errors="coerce")
                )
            )
        )
        best_state1_rows = (
            desig_poses.df.sort_values(
                [
                    "_desig_backbone_id",
                    "_individual_designability_score",
                    "plddt",
                    "i_pae",
                    "poses_description",
                ],
                ascending=[True, False, False, True, True],
                na_position="last",
            )
            .groupby("_desig_backbone_id", group_keys=False)
            .head(1)
            .set_index("_desig_backbone_id")
        )
        state1_id = poses.df["s1_rfd3_holo_description"]
        for source, destination in (
            ("plddt", "state1_designability_plddt"),
            ("i_pae", "state1_designability_i_pae"),
            ("i_ptm", "state1_designability_i_ptm"),
            ("_individual_designability_score", "state1_designability_score"),
        ):
            poses.df[destination] = state1_id.map(best_state1_rows[source])
        state1_pass = (
            (best_state1_rows["plddt"] > min_plddt_d)
            & (best_state1_rows["i_pae"] < max_i_pae_d)
            & (best_state1_rows["i_ptm"] > min_i_ptm_d)
        )
        survivor_stems = set(best_state1_rows.index[state1_pass])

        n_before = poses.df["s1_rfd3_holo_description"].nunique()
        poses.df = poses.df[poses.df["s1_rfd3_holo_description"].isin(survivor_stems)].reset_index(drop=True)
        funnel.log(
            "s1_5_designability", len(poses.df),
            f"ProteinMPNN {n_seqs_d}/backbone -> AF2 top {pre_af2_top_k_d}; "
            f"pLDDT>{min_plddt_d}, iPAE<{max_i_pae_d}, ipTM>{min_i_ptm_d}: "
            f"kept {len(survivor_stems)}/{n_before} backbones",
        )
        if poses.df.empty:
            raise RuntimeError(
                "AF2 designability pre-filter rejected all holo backbones. "
                "min_plddt may be too strict, or the RFdiffusion3 backbones may not "
                "be designable; inspect the s1_5_desig_af2/ scores before lowering "
                "the threshold.")

    print("\n[Stage 2] RFdiffusion3 apo partial diffusion")

    sys.path.insert(0, os.path.join(WS, "scripts"))
    from write_apo_inputs import generate_apo_inputs

    holo_out_dir = os.path.join(OUTPUTS, "s1_rfd3_holo")
    apo_input_dir = os.path.join(OUTPUTS, "staging", "rfd3_apo_inputs")

    if args.geometry_calibration:
        calibration_cfg = cfg.get("geometry_calibration", {})
        partial_t_values = args.partial_t_values or calibration_cfg.get(
            "partial_t_values", [2.0, 5.0, 10.0, 15.0]
        )
        partial_t_values = [float(value) for value in partial_t_values]
        if len(partial_t_values) < 2:
            raise ValueError("Geometry calibration requires at least two partial_t values")
        if len(partial_t_values) != len(set(partial_t_values)):
            raise ValueError("Geometry calibration partial_t values must be unique")
        if any(value <= 0.0 or value > 15.0 for value in partial_t_values):
            raise ValueError("Geometry calibration partial_t values must be in (0, 15] Angstroms")

        print(
            "  Calibration mode: matched state-1 backbones, production geometry "
            f"thresholds, partial_t={partial_t_values}"
        )
        geometry_cfg = cfg.get("geometry_gate", {})
        calibration_frames = []
        base_state1 = poses.df.copy()
        for partial_t in partial_t_values:
            tag = f"{partial_t:g}".replace(".", "p")
            prefix = f"s2_rfd3_apo_pt{tag}"
            level_input_dir = os.path.join(
                OUTPUTS, "staging", f"rfd3_apo_inputs_pt{tag}"
            )
            print(f"\n[Calibration] partial_t={partial_t:g} Angstrom")
            level_jsons = generate_apo_inputs(
                holo_dir=holo_out_dir,
                pcna_pdb=apo["pdb"],
                out_dir=level_input_dir,
                pcna_hotspots=apo.get("hotspots", ""),
                partial_t=partial_t,
                binder_chain=binder_chain,
                include_stems=survivor_stems,
            )
            combined_pdbs = []
            level_specs = {}
            for json_path in level_jsons:
                with open(json_path) as handle:
                    spec = json.load(handle)
                for name, level_spec in spec.items():
                    combined_pdbs.append(level_spec["input"])
                    level_specs[name] = level_spec

            level_apo_poses = Poses(
                poses=combined_pdbs,
                work_dir=OUTPUTS,
                jobstarter=gpu_jst,
            )
            level_params = RFD3Params(
                poses=level_apo_poses, spec_from_dict=level_specs
            )
            level_apo_poses = rfd3.run(
                poses=level_apo_poses,
                prefix=prefix,
                params=level_params,
                n_batches=1,
                diffusion_batch_size=max(1, APO_BATCH),
                options="skip_existing=True",
            )
            state_pairing.validate_staging_manifest(
                os.path.join(level_input_dir, "lineage_manifest.json"),
                base_state1["s1_rfd3_holo_description"],
            )
            paired = state_pairing.pair_state_outputs(
                base_state1,
                level_apo_poses.df,
                state2_description_col=f"{prefix}_description",
                state2_location_col=f"{prefix}_location",
                expected_variants=max(1, APO_BATCH),
            )
            paired = state_pairing.add_geometry_metrics(
                paired,
                state1_location_col="s1_rfd3_holo_location",
                state1_binder_chain=binder_chain,
                state2_binder_chain="A",
                interface_cutoff=float(geometry_cfg.get("interface_cutoff", 5.0)),
                clash_cutoff=float(geometry_cfg.get("clash_cutoff", 2.5)),
            )
            state_pairing.validate_generation_sanity(
                paired,
                state2_binder_chain="A",
                minimum_state_change=float(
                    geometry_cfg.get("generation_sanity_min_binder_ca_rmsd", 0.25)
                ),
            )
            paired["partial_t"] = partial_t
            paired["geometry_pass"] = state_pairing.geometry_pass_mask(
                paired, geometry_cfg
            )
            paired.to_csv(
                os.path.join(OUTPUTS, f"s2_state_pair_geometry_pt{tag}.csv"),
                index=False,
            )
            calibration_frames.append(paired)
            n_pass = int(paired["geometry_pass"].sum())
            funnel.log(
                f"geometry_calibration_pt{tag}", n_pass,
                f"production geometry gate; kept {n_pass}/{len(paired)} pairs",
            )

        combined_calibration = pd.concat(calibration_frames, ignore_index=True)
        from partial_t_calibration import write_calibration_artifacts

        recommendation = write_calibration_artifacts(
            combined_calibration,
            OUTPUTS,
            geometry_cfg,
            target_rmsd=float(
                af2_cfg.get("state2_designability", {}).get("target_rmsd", 3.0)
            ),
        )
        funnel.log(
            "geometry_calibration_complete", len(combined_calibration),
            f"matched sweep across {len(partial_t_values)} partial_t values",
        )
        print("\nPartial_t calibration complete.")
        print(f"  {recommendation['reason']}")
        print(f"  Table: {os.path.join(OUTPUTS, 'partial_t_geometry_summary.csv')}")
        print(f"  Plot:  {os.path.join(OUTPUTS, 'partial_t_geometry_comparison.png')}")
        print(f"  Report: {os.path.join(OUTPUTS, 'PARTIAL_T_CALIBRATION.md')}")
        return

    apo_jsons = generate_apo_inputs(
        holo_dir=holo_out_dir,
        pcna_pdb=apo["pdb"],
        out_dir=apo_input_dir,
        pcna_hotspots=apo.get("hotspots", ""),
        partial_t=apo.get("partial_t", 2.0),
        binder_chain=binder_chain,
        include_stems=survivor_stems,
    )

    combined_pdbs = []
    apo_specs = {}
    for jpath in apo_jsons:
        with open(jpath) as f:
            spec = json.load(f)
        for name, s in spec.items():
            combined_pdbs.append(s["input"])
            apo_specs[name] = s

    apo_poses = Poses(
        poses=combined_pdbs,
        work_dir=OUTPUTS,
        jobstarter=gpu_jst,
    )

    params_apo = RFD3Params(poses=apo_poses, spec_from_dict=apo_specs)

    # The apo stage runs one job per holo backbone, parallelised across the
    # array, so a small per-pose batch suffices. Passed as a keyword argument
    # rather than in the options string, as in the holo stage above.
    apo_poses = rfd3.run(
        poses=apo_poses,
        prefix="s2_rfd3_apo",
        params=params_apo,
        n_batches=1,
        diffusion_batch_size=max(1, APO_BATCH),
        options="skip_existing=True",
    )

    state_pairing.validate_staging_manifest(
        os.path.join(apo_input_dir, "lineage_manifest.json"),
        poses.df["s1_rfd3_holo_description"],
    )
    poses.df = state_pairing.pair_state_outputs(
        poses.df, apo_poses.df, expected_variants=max(1, APO_BATCH)
    )
    geometry_cfg = cfg.get("geometry_gate", {})
    if args.smoke and cfg.get("smoke_geometry_gate") is not None:
        # A smoke run is an integration test rather than a selection run, and is
        # permitted to exercise the downstream stages even when its small
        # trajectory sample contains no production-quality state pair. Production
        # runs always apply geometry_gate unmodified.
        geometry_cfg = {**geometry_cfg, **cfg["smoke_geometry_gate"]}
        print(
            "  Smoke run: relaxed pre-sequence geometry thresholds in effect; "
            "downstream scores are integration diagnostics."
        )
    poses.df = state_pairing.add_geometry_metrics(
        poses.df, state1_location_col="s1_rfd3_holo_location",
        state1_binder_chain=binder_chain, state2_binder_chain="A",
        interface_cutoff=float(geometry_cfg.get("interface_cutoff", 5.0)),
        clash_cutoff=float(geometry_cfg.get("clash_cutoff", 2.5)),
    )
    state_pairing.validate_generation_sanity(
        poses.df,
        state2_binder_chain="A",
        minimum_state_change=float(
            cfg.get("geometry_gate", {}).get("generation_sanity_min_binder_ca_rmsd", 0.25)
        ),
    )
    poses.df["geometry_pass"] = state_pairing.geometry_pass_mask(poses.df, geometry_cfg)
    poses.df.to_csv(os.path.join(OUTPUTS, "s2_state_pair_geometry.csv"), index=False)
    if geometry_cfg.get("enabled", False):
        n_before_geometry = len(poses.df)
        poses.df = poses.df[poses.df["geometry_pass"]].reset_index(drop=True)
        if poses.df.empty:
            raise RuntimeError(
                "All state pairs failed the pre-sequence geometry gate; inspect "
                "s2_state_pair_geometry.csv and recalibrate partial_t or placement."
            )
        funnel.log("s2_geometry_gate", len(poses.df),
                   f"same-interface, bounded-RMSD, mutually-exclusive pairs; kept {len(poses.df)}/{n_before_geometry}")
    funnel.log(
        "s2_rfd3_apo", len(apo_poses.df),
        f"state-2 candidates generated and lineage-expanded ({APO_BATCH}/state-1)",
    )

    poses = stages.step2_5_select_state2(ctx, poses)
    run_shared_tail(ctx, poses)


def run_shared_tail(ctx, poses):
    """Sequence design through evaluation: stages 3 onward.

    Separated so that alternative backbone-generation front ends reuse this
    scoring, gating and reporting path unmodified, preserving comparability
    between variants.

    The interface is (ctx, poses). All state is carried on PipelineContext;
    `poses` must provide the per-design columns produced by stages 1 to 2.5, in
    particular s1_rfd3_holo_location and state2_pdb.
    """
    ADAPTIVE_MAX_NSEQ, ADAPTIVE_TARGET, AF2_ENABLED, AF2_GATE_ONLY = ctx.ADAPTIVE_MAX_NSEQ, ctx.ADAPTIVE_TARGET, ctx.AF2_ENABLED, ctx.AF2_GATE_ONLY
    AF2_PARAMS_DIR, DECOY_TARGETS, DMPNN_NSEQ, DMPNN_OPTIONS = ctx.AF2_PARAMS_DIR, ctx.DECOY_TARGETS, ctx.DMPNN_NSEQ, ctx.DMPNN_OPTIONS
    LMPNN_NSEQ, MAX_BACKBONES, MPNN_MSD_NSEQ, OUTPUTS = ctx.LMPNN_NSEQ, ctx.MAX_BACKBONES, ctx.MPNN_MSD_NSEQ, ctx.OUTPUTS
    POST_AF2_PER_BACKBONE, POST_AF2_TOP_K, POST_DMPNN_TOP_K, PROXY_MPNN_MODEL = ctx.POST_AF2_PER_BACKBONE, ctx.POST_AF2_TOP_K, ctx.POST_DMPNN_TOP_K, ctx.PROXY_MPNN_MODEL
    SELFCONS_IPTM_THRESHOLD, SELFCONS_PLDDT_THRESHOLD, SELFCONS_RMSD_DECOY_THRESHOLD, SPECIFICITY_ENABLED = ctx.SELFCONS_IPTM_THRESHOLD, ctx.SELFCONS_PLDDT_THRESHOLD, ctx.SELFCONS_RMSD_DECOY_THRESHOLD, ctx.SPECIFICITY_ENABLED
    SPECIFICITY_MARGIN_THRESHOLD, SPECIFICITY_MAX_CANDIDATES, WS, af2_cfg = ctx.SPECIFICITY_MARGIN_THRESHOLD, ctx.SPECIFICITY_MAX_CANDIDATES, ctx.WS, ctx.af2_cfg
    apo, binder_chain, boltz_env, cfg = ctx.apo, ctx.binder_chain, ctx.boltz_env, ctx.cfg
    cpu_jst, cpu_jst_fast, dynamicmpnn, funnel = ctx.cpu_jst, ctx.cpu_jst_fast, ctx.dynamicmpnn, ctx.funnel
    holo, ligandmpnn, sp = ctx.holo, ctx.ligandmpnn, ctx.sp

    print("\n[Stage 3] MPNN proxy sequence design")

    lmpnn_opts = f"--chains_to_design {binder_chain} "

    poses = ligandmpnn.run(
        poses=poses,
        prefix="s3_ligandmpnn",
        jobstarter=cpu_jst,
        nseq=LMPNN_NSEQ,
        model_type=PROXY_MPNN_MODEL,
        options=lmpnn_opts,
    )
    funnel.log("s3_ligandmpnn", len(poses.df), f"{LMPNN_NSEQ} seqs/backbone")

    poses = stages.step3_5a_lmpnn_topk(ctx, poses)
    # Stage 3.5A applied to the apo side. Without it the apo backbone passes
    # triage unsequenced and unscored, and a backbone with an acceptable holo
    # side but a structurally poor apo side is only rejected at stage 5 or 6,
    # after the multi-state design and four-state scoring budget has been spent.
    print("\n[Stage 3.5A-apo] LigandMPNN design on apo backbone, top-1 per backbone")

    # Always "A", independently of the targets used: write_apo_inputs.py builds
    # every apo contig with the binder segment first, "{binder}1-N,{target...}",
    # and RFdiffusion3 reindexes output chains sequentially in contig order.
    # Verified against s2_rfd3_apo output.
    apo_binder_chain = "A"

    # Constructed here rather than alongside boltz_env: apo_binder_chain is not
    # known earlier, and binder_chain is set once the stage-1 backbones exist.
    af2_env = af2_gate.AF2GateEnv(
        outputs=OUTPUTS, binder_chain=binder_chain, apo_binder_chain=apo_binder_chain,
        af2_cfg=af2_cfg or {}, params_dir=AF2_PARAMS_DIR,
    )
    poses.df["_holo_pose"] = poses.df["poses"]

    # Stage 3.5A can retain more than one holo sequence variant per backbone, so
    # several rows of poses.df may share a state2_pdb. LigandMPNN requires unique
    # input pose paths; duplicates cause its per-job output collection to
    # collide and return short. Apo design therefore runs on a dedicated Poses
    # object built from the unique backbones, and the resulting apo sequence is
    # broadcast back onto every row sharing that backbone.
    apo_backbone_df = poses.df.drop_duplicates(subset="s1_rfd3_holo_description").reset_index(drop=True)
    apo_poses_for_lmpnn = Poses(
        poses=apo_backbone_df["state2_pdb"].tolist(),
        work_dir=OUTPUTS,
        jobstarter=cpu_jst,
    )
    # Backbone identity is tagged on the pre-expansion rows: ligandmpnn.run()
    # expands one row into nseq rows and duplicates pre-existing columns across
    # that expansion automatically, as "_pre_boltz_id" does ahead of each
    # boltz.run() call. Tagging after the run would mismatch in length.
    apo_poses_for_lmpnn.df["_apo_backbone"] = apo_backbone_df["s1_rfd3_holo_description"].values
    apo_poses_for_lmpnn = ligandmpnn.run(
        poses=apo_poses_for_lmpnn,
        prefix="s3_5a_apo_ligandmpnn",
        jobstarter=cpu_jst,
        nseq=LMPNN_NSEQ,
        model_type=PROXY_MPNN_MODEL,
        options=f"--chains_to_design {apo_binder_chain} ",
    )
    funnel.log("s3_5a_apo_ligandmpnn", len(apo_poses_for_lmpnn.df),
               f"{LMPNN_NSEQ} apo seqs/backbone ({len(apo_backbone_df)} unique backbones)")

    apo_poses_for_lmpnn.df["_apo_lmpnn_combined_score"] = (
        apo_poses_for_lmpnn.df["s3_5a_apo_ligandmpnn_overall_confidence"]
        + apo_poses_for_lmpnn.df.get("s3_5a_apo_ligandmpnn_ligand_confidence", 0)
    )
    apo_poses_for_lmpnn.filter_poses_by_rank(
        n=1,
        score_col="_apo_lmpnn_combined_score",
        group_col="_apo_backbone",
        ascending=False,
        prefix="s3_5a_apo_lmpnn_top1",
    )
    funnel.log("s3_5a_apo_lmpnn_top1", len(apo_poses_for_lmpnn.df), "top-1/backbone by apo LigandMPNN confidence")

    apo_seq_map = apo_poses_for_lmpnn.df.set_index("_apo_backbone")["s3_5a_apo_ligandmpnn_sequence"]
    apo_pose_map = apo_poses_for_lmpnn.df.set_index("_apo_backbone")["poses"]
    poses.df["s3_5a_apo_ligandmpnn_sequence"] = poses.df["s1_rfd3_holo_description"].map(apo_seq_map)
    poses.df["_apo_pose"] = poses.df["s1_rfd3_holo_description"].map(apo_pose_map)

    # Joint no-MSA self-consistency filter over both states. Boltz-2 is run in
    # msa=empty mode (--use_msa_server omitted), giving single-sequence
    # prediction without a network round trip; this is appropriate here because
    # both binder sequences are de novo and have no evolutionary homologues to
    # retrieve. Two quantities are computed per state: binder-alone monomer
    # pLDDT, measuring whether the sequence folds coherently in isolation from a
    # target-supported complex prediction, and complex ipTM. Ranking uses the
    # minimum of the two monomer pLDDTs, since both states must independently
    # support a foldable sequence. This is backbone triage; final sequences are
    # designed in stage 5.
    print("\n[Stage 3.5B] no-MSA Boltz-2 self-consistency filter, holo and apo")

    selfcons_work_dir = os.path.join(OUTPUTS, "s3_5b_selfcons")
    os.makedirs(selfcons_work_dir, exist_ok=True)

    # Each self-consistency sub-prediction runs on its own Poses object, leaving
    # the shared `poses` object's "poses" and poses_description untouched, and
    # the resulting columns are merged back on a stable "_design_id" key
    # captured beforehand. Boltz appends a "_model_N" suffix to the description
    # it is given, and this naming is not stable when a single shared object is
    # repointed between FASTA and PDB inputs across successive boltz.run()
    # calls; that pattern previously produced a "no overlap in merge" failure.
    # Independent side objects avoid it.
    poses.df["_design_id"] = poses.df["poses_description"]


    # Holo: binder-alone monomer foldability
    holo_fasta_paths = seq_io.write_binder_fastas(
        poses.df, "s3_ligandmpnn_sequence", os.path.join(selfcons_work_dir, "holo_binder_fastas"),
        binder_chain_letter=binder_chain,
    )
    holo_mono_df = boltz_scoring.run_side_boltz(boltz_env, holo_fasta_paths, "s3_5b_holo_mono", poses.df["_design_id"])
    if "s3_5b_holo_mono_plddt_location" in holo_mono_df.columns:
        holo_mono_df["s3_5b_holo_mono_plddt_mean"] = holo_mono_df["s3_5b_holo_mono_plddt_location"].apply(structure_io.load_plddt_mean)
    poses.df = poses.df.merge(
        holo_mono_df[["_design_id"] + [c for c in holo_mono_df.columns if c.startswith("s3_5b_holo_mono")]],
        on="_design_id", how="left",
    )

    # Holo: full complex (binder + holo target), no MSA
    holo_complex_df = boltz_scoring.run_side_boltz(boltz_env, poses.df["_holo_pose"].tolist(), "s3_5b_holo_complex", poses.df["_design_id"])
    poses.df = poses.df.merge(
        holo_complex_df[["_design_id"] + [c for c in holo_complex_df.columns if c.startswith("s3_5b_holo_complex")]],
        on="_design_id", how="left",
    )

    # Apo: binder-alone monomer foldability
    apo_fasta_paths = seq_io.write_binder_fastas(
        poses.df, "s3_5a_apo_ligandmpnn_sequence", os.path.join(selfcons_work_dir, "apo_binder_fastas"),
        binder_chain_letter=apo_binder_chain,
    )
    apo_mono_df = boltz_scoring.run_side_boltz(boltz_env, apo_fasta_paths, "s3_5b_apo_mono", poses.df["_design_id"])
    if "s3_5b_apo_mono_plddt_location" in apo_mono_df.columns:
        apo_mono_df["s3_5b_apo_mono_plddt_mean"] = apo_mono_df["s3_5b_apo_mono_plddt_location"].apply(structure_io.load_plddt_mean)
    poses.df = poses.df.merge(
        apo_mono_df[["_design_id"] + [c for c in apo_mono_df.columns if c.startswith("s3_5b_apo_mono")]],
        on="_design_id", how="left",
    )

    # Apo: full complex (binder + apo target)
    apo_complex_df = boltz_scoring.run_side_boltz(boltz_env, poses.df["_apo_pose"].tolist(), "s3_5b_apo_complex", poses.df["_design_id"])
    poses.df = poses.df.merge(
        apo_complex_df[["_design_id"] + [c for c in apo_complex_df.columns if c.startswith("s3_5b_apo_complex")]],
        on="_design_id", how="left",
    )
    funnel.log("s3_5b_selfcons_scored", len(poses.df), "no-MSA holo+apo monomer/complex self-consistency scored")

    # skipna=False: if either side's score is missing, for instance from a
    # partial cached scorefile left by an interrupted run, the joint score must
    # be NaN rather than falling back to the state that does have a value, which
    # would let a design with a failed self-consistency check rank on one state
    # alone.
    poses.df["s3_5b_joint_mono_plddt"] = poses.df[["s3_5b_holo_mono_plddt_mean", "s3_5b_apo_mono_plddt_mean"]].min(axis=1, skipna=False)

    # Decoy-normalised structural self-consistency, both states. Each predicted
    # monomer fold is compared by RMSD against the backbone it was designed on
    # (target) and against a randomly assigned different backbone from the same
    # batch (decoy). The monomer structures predicted above are reused, so no
    # further Boltz calls are required, only CPU-side geometry. A low
    # target/decoy ratio indicates that the fold tracks its own backbone rather
    # than being generically compact, which would score similarly against any
    # reference. This follows the decoy-normalised metric of the DynamicMPNN
    # publication, using an already-generated negative control in place of a
    # dedicated decoy prediction.
    unique_backbones = sorted(poses.df["s1_rfd3_holo_description"].unique())
    decoy_assignment = {}
    if len(unique_backbones) > 1:
        rng = np.random.RandomState(42)
        shuffled = list(unique_backbones)
        while True:
            rng.shuffle(shuffled)
            if all(a != b for a, b in zip(unique_backbones, shuffled)):
                break
        decoy_assignment = dict(zip(unique_backbones, shuffled))
    poses.df["_decoy_backbone"] = poses.df["s1_rfd3_holo_description"].map(decoy_assignment)

    if decoy_assignment:
        ref_dir = os.path.join(selfcons_work_dir, "binder_refs")
        os.makedirs(ref_dir, exist_ok=True)
        holo_ref_paths, apo_ref_paths = {}, {}
        for backbone in unique_backbones:
            holo_pdb = poses.df.loc[poses.df["s1_rfd3_holo_description"] == backbone, "s1_rfd3_holo_location"].iloc[0]
            apo_pdb = poses.df.loc[poses.df["s1_rfd3_holo_description"] == backbone, "state2_pdb"].iloc[0]
            holo_ref = os.path.join(ref_dir, f"{backbone}_holo_ref.pdb")
            apo_ref = os.path.join(ref_dir, f"{backbone}_apo_ref.pdb")
            structure_io._extract_binder_to_chain(holo_pdb, binder_chain, "A", holo_ref)
            structure_io._extract_binder_to_chain(apo_pdb, apo_binder_chain, "A", apo_ref)
            holo_ref_paths[backbone] = holo_ref
            apo_ref_paths[backbone] = apo_ref

        poses.df["_holo_target_ref"] = poses.df["s1_rfd3_holo_description"].map(holo_ref_paths)
        poses.df["_holo_decoy_ref"] = poses.df["_decoy_backbone"].map(holo_ref_paths)
        poses.df["_apo_target_ref"] = poses.df["s1_rfd3_holo_description"].map(apo_ref_paths)
        poses.df["_apo_decoy_ref"] = poses.df["_decoy_backbone"].map(apo_ref_paths)

        def _run_side_rmsd(file_list: list[str], prefix: str, ref_col: str) -> pd.DataFrame:
            pdb_paths = structure_io._cif_locations_to_pdb(file_list, os.path.join(selfcons_work_dir, f"{prefix}_as_pdb"))
            side = Poses(poses=pdb_paths, work_dir=OUTPUTS, jobstarter=cpu_jst_fast)
            side.df["_design_id"] = poses.df["_design_id"].values
            side.df["_ref"] = poses.df[ref_col].values
            side = structure_io.make_backbone_rmsd(atoms=["CA"], chains=["A"], jobstarter=cpu_jst_fast).run(poses=side, prefix=prefix, ref_col="_ref", jobstarter=cpu_jst_fast)
            # One row per key, as in _run_side_boltz, so the merge stays 1:1.
            return side.df[["_design_id", f"{prefix}_rmsd"]].drop_duplicates(subset="_design_id", keep="first")

        # how="left" is explicit: the default inner join would drop any poses.df
        # row whose _design_id did not return from BackboneRMSD, for instance
        # after a partial array-task failure, instead of surfacing it as NaN.
        # Consistent with the Boltz-score merges above.
        poses.df = poses.df.merge(_run_side_rmsd(poses.df["s3_5b_holo_mono_location"].tolist(), "s3_5b_holo_rmsd_target", "_holo_target_ref"), on="_design_id", how="left")
        poses.df = poses.df.merge(_run_side_rmsd(poses.df["s3_5b_holo_mono_location"].tolist(), "s3_5b_holo_rmsd_decoy", "_holo_decoy_ref"), on="_design_id", how="left")
        poses.df = poses.df.merge(_run_side_rmsd(poses.df["s3_5b_apo_mono_location"].tolist(), "s3_5b_apo_rmsd_target", "_apo_target_ref"), on="_design_id", how="left")
        poses.df = poses.df.merge(_run_side_rmsd(poses.df["s3_5b_apo_mono_location"].tolist(), "s3_5b_apo_rmsd_decoy", "_apo_decoy_ref"), on="_design_id", how="left")

        poses.df["s3_5b_holo_rmsd_decoy_ratio"] = poses.df["s3_5b_holo_rmsd_target_rmsd"] / poses.df["s3_5b_holo_rmsd_decoy_rmsd"]
        poses.df["s3_5b_apo_rmsd_decoy_ratio"] = poses.df["s3_5b_apo_rmsd_target_rmsd"] / poses.df["s3_5b_apo_rmsd_decoy_rmsd"]
        funnel.log("s3_5b_decoy_rmsd_scored", len(poses.df), "decoy-normalized structural self-consistency (holo + apo)")

    poses.df.to_csv(os.path.join(OUTPUTS, "s3_5b_selfcons_all_scored.csv"), index=False)

    n_before = len(poses.df)
    mask = pd.Series(True, index=poses.df.index)
    if SELFCONS_PLDDT_THRESHOLD > 0:
        mask &= poses.df["s3_5b_holo_mono_plddt_mean"] > SELFCONS_PLDDT_THRESHOLD
        mask &= poses.df["s3_5b_apo_mono_plddt_mean"] > SELFCONS_PLDDT_THRESHOLD
    if SELFCONS_IPTM_THRESHOLD > 0:
        mask &= poses.df["s3_5b_holo_complex_iptm"] > SELFCONS_IPTM_THRESHOLD
        mask &= poses.df["s3_5b_apo_complex_iptm"] > SELFCONS_IPTM_THRESHOLD
    if SELFCONS_RMSD_DECOY_THRESHOLD > 0 and "s3_5b_holo_rmsd_decoy_ratio" in poses.df.columns:
        mask &= poses.df["s3_5b_holo_rmsd_decoy_ratio"] < SELFCONS_RMSD_DECOY_THRESHOLD
        mask &= poses.df["s3_5b_apo_rmsd_decoy_ratio"] < SELFCONS_RMSD_DECOY_THRESHOLD
    poses.df = poses.df[mask].reset_index(drop=True)

    selfcons_criteria = [
        (
            f"holo+apo monomer pLDDT>{SELFCONS_PLDDT_THRESHOLD}"
            if SELFCONS_PLDDT_THRESHOLD > 0 else "monomer pLDDT gate disabled"
        ),
        (
            f"complex ipTM>{SELFCONS_IPTM_THRESHOLD}"
            if SELFCONS_IPTM_THRESHOLD > 0 else "complex ipTM gate disabled"
        ),
        (
            f"RMSD decoy-ratio<{SELFCONS_RMSD_DECOY_THRESHOLD}"
            if SELFCONS_RMSD_DECOY_THRESHOLD > 0 else "RMSD decoy-ratio gate disabled"
        ),
    ]
    funnel.log(
        "s3_5b_filtered", len(poses.df), f"from {n_before} ({'; '.join(selfcons_criteria)})"
    )

    if len(poses.df) == 0:
        print("No designs passed the self-consistency filter; stopping.")
        print("See s3_5b_selfcons_all_scored.csv for the per-design scores.")
        return

    # Collapse to the best-scoring sequence per backbone, ranked by the lower of
    # the holo and apo monomer pLDDTs. No-MSA complex ipTM is retained for
    # inspection but not used for ranking, MSA-free complex predictions being
    # noisier than the monomer foldability check. The number of backbones
    # forwarded to multi-state design and four-state scoring is then capped.
    poses.filter_poses_by_rank(
        n=1, score_col="s3_5b_joint_mono_plddt", group_col="s1_rfd3_holo_description",
        ascending=False, prefix="s3_5c_dedup_per_backbone",
    )
    poses.filter_poses_by_rank(
        n=MAX_BACKBONES, score_col="s3_5b_joint_mono_plddt", ascending=False,
        prefix="s3_5d_top_backbones",
    )
    funnel.log("s3_5_backbones_forwarded", len(poses.df),
               f"top {MAX_BACKBONES} backbones by joint (min) holo/apo monomer pLDDT")

    # poses.df["poses"] is untouched by stage 3.5B, all sub-predictions having
    # run on independent side objects: it still holds the complex-ready holo
    # structure from stage 3.5A-apo, as stages 5 and 6 require.

    print("\n[Stage 5] DynamicMPNN multi-state sequence design")

    poses.df["partner_A_pdb"] = holo["pdb"]
    poses.df["partner_B_pdb"] = apo["pdb"]

    # The one-row-per-backbone frame is retained before the num_seqs expansion,
    # so that the adaptive top-up in stage 5.5 can re-invoke DynamicMPNN on the
    # deficient backbones alone without regenerating already-scored designs.
    backbone_level_df = poses.df.copy()

    apo_binder_chain = "A"
    poses = dynamicmpnn.run(
        poses=poses,
        prefix="s5_dynamicmpnn",
        state1_col="s1_rfd3_holo_location",
        state1_chain=binder_chain,
        state2_col="state2_pdb",
        state2_chain=apo_binder_chain,
        num_seqs=DMPNN_NSEQ,
        options=DMPNN_OPTIONS,
        jobstarter=cpu_jst,
    )
    funnel.log("s5_dynamicmpnn", len(poses.df), f"{DMPNN_NSEQ} seqs/design")

    # Sequence diversity check
    seq_col = "s5_dynamicmpnn_sequence"
    if seq_col in poses.df.columns:
        div = seq_io.compute_seq_diversity(poses.df, seq_col)
        print("\n  Sequence diversity (DynamicMPNN):")
        print(f"    {div['n_seqs']} sequences, mean length {div['mean_length']:.0f}")
        print(f"    Mean unique AAs/seq: {div['mean_unique_aas']:.1f}")
        print(f"    Pairwise identity: {div['mean_pairwise_identity']:.1%} "
              f"(range {div['min_pairwise_identity']:.1%}–{div['max_pairwise_identity']:.1%})")
        if div["mean_pairwise_identity"] > 0.85:
            print("    Warning: high pairwise identity; DynamicMPNN sampling may have collapsed")
        div_path = os.path.join(OUTPUTS, "sequence_diversity.json")
        with open(div_path, "w") as f:
            json.dump({"dynamicmpnn": div}, f, indent=2)

    # No-MSA switch pre-scoring with per-backbone selection. DynamicMPNN designs
    # dmpnn_nseq sequences per backbone, and scoring all of them through the
    # four-state MSA Boltz-2 run of stage 6 dominates the compute cost. Every
    # sequence is therefore scored here at low cost (no-MSA holo and apo complex,
    # single sample), the scores are retained for the full-sample readout, and
    # only the top post_dmpnn_top_k per backbone are forwarded to stage 6. The
    # pre-scoring runs on isolated side Poses objects built from binder-only
    # FASTAs and does not touch poses.df["poses"], so it cannot perturb the stage
    # 6 Boltz chain, whose BoltzParams.generate_yaml_files appends to the current
    # pose YAML and would otherwise accumulate chains.

    # The four-state MSA Boltz-2 scorer is shared by stage 6 (DynamicMPNN),
    # stage 7 (ProteinMPNN-MSD) and the scrambled null control.
    #
    # Each state must be scored on a fresh Poses object built from binder-only
    # FASTAs, never by chaining boltz.run() on a shared object. Under chaining,
    # Boltz re-reads the preceding step's output structure and
    # BoltzParams.generate_yaml_files appends the new target, so each apo or
    # control prediction carried over the previous state's target chain: the apo
    # YAML became [binder, holo_target, apo_target] and its ipTM was computed on
    # a contaminated three-chain complex. Fresh side objects with keyed 1:1
    # merges avoid both this contamination and row expansion during merges.


    # AF2 initial-guess gate (scores both states on their design backbones)

    if AF2_ENABLED:
        print("\n[Stage 5.5] AF2 initial-guess gate, orthogonal discriminative selection")
        af2df = af2_gate.af2_gate_score(af2_env, poses.df, "s5_dynamicmpnn_sequence", "s5_5_af2_gate")
        poses.df = poses.df.merge(af2df, on="poses_description", how="left")

        # AF2 scramble null, supplying both the relative (relaxed) tier and the
        # per-run demonstration that the gate metric discriminates. No metric
        # gates unless it separates real designs from the composition-matched
        # null.
        n_scr = cfg.get("evaluation", {}).get("n_scramble_controls", 0)
        af2_null = None
        if n_scr and n_scr > 0:
            scr = paired_nulls.balanced_scrambles(
                poses.df, sequence_col="s5_dynamicmpnn_sequence",
                backbone_col="s1_rfd3_holo_description", n=int(n_scr), seed=42,
            )
            null_keys = scr[["poses_description", "_null_backbone", "_real_design_id"]].copy()
            af2_null = af2_gate.af2_gate_score(af2_env, scr, "_scr_seq", "s5_5_af2_gate_null")
            if len(af2_null):
                af2_null = af2_null.merge(null_keys, on="poses_description", how="left", validate="one_to_one")
                af2_null.to_csv(os.path.join(OUTPUTS, "af2_gate_null.csv"), index=False)

        poses.df = switch_gating.assign_af2_tiers(
            poses.df, "af2_holo_plddt", "af2_apo_plddt", "af2_holo_i_pae", "af2_apo_i_pae",
            null_df=af2_null, strict_abs=af2_cfg.get("strict_abs"),
            holo_iptm="af2_holo_i_ptm", apo_iptm="af2_apo_i_ptm",
        )
        null_gate_pass = False
        if af2_null is not None and len(af2_null):
            separation = paired_nulls.separation_table(
                poses.df, af2_null,
                {
                    "af2_holo_plddt": "higher", "af2_apo_plddt": "higher",
                    "af2_holo_i_pae": "lower", "af2_apo_i_pae": "lower",
                },
                real_backbone_col="s1_rfd3_holo_description",
            )
            separation.to_csv(os.path.join(OUTPUTS, "af2_null_separation.csv"), index=False)
            min_null_auc = float(sp.get("min_null_auc", cfg.get("evaluation", {}).get("min_null_auc", 0.70)))
            min_null_pairs = int(sp.get("min_null_pairs", cfg.get("evaluation", {}).get("min_null_pairs", 10)))
            null_gate_pass = paired_nulls.passes_stop_go(
                separation, min_auc=min_null_auc, min_pairs=min_null_pairs
            )
        poses.df["af2_null_discriminates"] = bool(null_gate_pass)
        require_null = bool(sp.get(
            "require_null_separation",
            cfg.get("evaluation", {}).get("require_null_separation", True),
        ))
        if require_null and not null_gate_pass:
            poses.df["af2_relaxed"] = False
            poses.df["af2_tier"] = np.where(poses.df["af2_strict"], "strict", "fail")
        poses.df.to_csv(os.path.join(OUTPUTS, "s5_5_af2_gate_all.csv"), index=False)

        if af2_null is not None and len(af2_null) >= 5:
            for m in ("af2_holo_plddt", "af2_apo_plddt"):
                if m in poses.df.columns and m in af2_null.columns:
                    sep = switch_gating.null_separation(poses.df[m], af2_null[m], "higher")
                    print(f"  null-separation {m}: z={sep['z']:.2f}  AUC={sep['auc']:.2f}  "
                          f"({'discriminates' if (sep['auc'] or 0) >= 0.7 else 'below threshold'})")
        n_strict = int(poses.df.get("af2_strict", pd.Series(dtype=bool)).sum())
        n_relaxed = int(poses.df.get("af2_relaxed", pd.Series(dtype=bool)).sum())
        funnel.log("s5_5_af2_gate", len(poses.df),
                   f"AF2-IG scored both states; strict={n_strict} relaxed={n_relaxed}; null_discriminates={null_gate_pass}")
        if require_null and not null_gate_pass:
            funnel.log("stopped_null_gate", 0,
                       "AF2 metrics did not separate real sequences from paired backbone nulls")
            print("Stopping: paired AF2 null discrimination failed; scoring was not launched.")
            return

        # Adaptive resampling top-up. A backbone with few or no AF2-gate passes
        # among its first DMPNN_NSEQ sequences receives additional sequences,
        # which are cheap to generate, rather than being judged on a thin sample.
        # Exactly one top-up round is performed, bounding the additional compute.
        bb_col = "s1_rfd3_holo_description"
        if ADAPTIVE_TARGET > 0 and ADAPTIVE_MAX_NSEQ > DMPNN_NSEQ and bb_col in poses.df.columns:
            n_pass = (poses.df["af2_tier"] != "fail").groupby(poses.df[bb_col]).sum()
            deficient = set(n_pass[n_pass < ADAPTIVE_TARGET].index)
            if deficient:
                n_extra = ADAPTIVE_MAX_NSEQ - DMPNN_NSEQ
                print(f"  Adaptive top-up: {len(deficient)} backbone(s) below target "
                      f"({ADAPTIVE_TARGET} AF2-gate passes); generating {n_extra} further "
                      f"DynamicMPNN sequences each")
                topup_input = backbone_level_df[backbone_level_df[bb_col].isin(deficient)].reset_index(drop=True)
                topup_poses = Poses(poses=topup_input["poses"].tolist(), work_dir=OUTPUTS, jobstarter=cpu_jst)
                for c in topup_input.columns:
                    if c not in topup_poses.df.columns:
                        topup_poses.df[c] = topup_input[c].values
                topup_poses = dynamicmpnn.run(
                    poses=topup_poses, prefix="s5_dynamicmpnn_topup",
                    state1_col="s1_rfd3_holo_location", state1_chain=binder_chain,
                    state2_col="state2_pdb", state2_chain=apo_binder_chain,
                    num_seqs=n_extra, options=DMPNN_OPTIONS, jobstarter=cpu_jst,
                )
                # DynamicMPNN output numbering (sample_000, sample_001, ...)
                # restarts at zero on every call, and ProtFlow's collect_scores()
                # builds poses_description as
                # f"{input_poses_description}_{seq_id}". As this top-up reuses
                # the input backbone description of the original stage-5 call,
                # its output names collide exactly with that call's. Every
                # downstream merge on poses_description would then join
                # duplicate-keyed frames, squaring the row count and merging two
                # distinct designs under one name. Uniqueness is enforced here,
                # before any merge or AF2 scoring key depends on the column.
                topup_poses.df["poses_description"] = (
                    topup_poses.df["poses_description"].astype(str) + "_topup")
                # Normalised column name, so that the downstream stages, which
                # key on s5_dynamicmpnn_sequence, treat top-up designs identically.
                topup_poses.df["s5_dynamicmpnn_sequence"] = topup_poses.df["s5_dynamicmpnn_topup_sequence"]
                topup_poses.df["method"] = "dynamicmpnn"
                topup_af2 = af2_gate.af2_gate_score(af2_env, topup_poses.df, "s5_dynamicmpnn_sequence", "s5_5_af2_gate_topup")
                topup_poses.df = topup_poses.df.merge(topup_af2, on="poses_description", how="left")
                topup_poses.df = switch_gating.assign_af2_tiers(
                    topup_poses.df, "af2_holo_plddt", "af2_apo_plddt", "af2_holo_i_pae", "af2_apo_i_pae",
                    null_df=af2_null, strict_abs=af2_cfg.get("strict_abs"),
                    holo_iptm="af2_holo_i_ptm", apo_iptm="af2_apo_i_ptm",
                )
                poses.df = pd.concat([poses.df, topup_poses.df], ignore_index=True, sort=False)
                funnel.log("s5_5_adaptive_topup", len(poses.df),
                           f"+{len(topup_poses.df)} top-up designs for {len(deficient)} deficient backbones")

        if sp.get("forward_only_af2_passes", cfg.get("evaluation", {}).get("forward_only_af2_passes", True)):
            poses.df = poses.df[poses.df["af2_tier"] != "fail"].reset_index(drop=True)
            if poses.df.empty:
                funnel.log("stopped_no_af2_passes", 0, "No sequence passed the two-state AF2 gate")
                print("Stopping: no sequence passed the two-state AF2 gate; scoring was not launched.")
                return

        if POST_AF2_PER_BACKBONE and POST_AF2_PER_BACKBONE > 0:
            poses.df = (poses.df.sort_values("af2_switch_plddt", ascending=False)
                        .groupby("s1_rfd3_holo_description", group_keys=False)
                        .head(POST_AF2_PER_BACKBONE).reset_index(drop=True))

        # Forward the top POST_AF2_TOP_K designs by harmonic AF2 pLDDT to the
        # four-state Boltz-2 scoring. The cap is fixed, so upstream
        # over-generation cannot inflate the expensive budget.
        if POST_AF2_TOP_K and POST_AF2_TOP_K > 0 and len(poses.df) > POST_AF2_TOP_K:
            poses.df = (poses.df.sort_values("af2_switch_plddt", ascending=False)
                        .head(POST_AF2_TOP_K).reset_index(drop=True))
        funnel.log("s5_5_af2_forwarded", len(poses.df),
                   f"top {POST_AF2_TOP_K} by AF2 harmonic pLDDT -> expensive scoring")

        # Consensus tier: the AF2 structure is captured for the post-selection
        # designs that reach stage 6. This re-scores the same backbone and
        # sequence as above but only for the survivors, bounding the cost at
        # approximately 2*len(poses.df) predictions rather than the full
        # pre-selection pool. It adds the af2_{holo,apo}_pdb columns used after
        # stage 6 to compare the Boltz and AF2 structures of a given design.
        if len(poses.df):
            struct_af2 = af2_gate.af2_gate_score(af2_env, poses.df, "s5_dynamicmpnn_sequence",
                                         "s5_5_af2_structures", save_structures=True)
            pdb_cols = [c for c in struct_af2.columns if c.endswith("_pdb")]
            if pdb_cols:
                poses.df = poses.df.merge(struct_af2[["poses_description"] + pdb_cols],
                                          on="poses_description", how="left")
            funnel.log("s5_5_af2_structures", len(poses.df),
                       f"AF2 structures captured for {len(pdb_cols)} state(s) (consensus tier)")

    elif POST_DMPNN_TOP_K and POST_DMPNN_TOP_K > 0:
        print("\n[Stage 5.5] no-MSA switch pre-scoring and selection, proxy metric")

        cheap = boltz_scoring.cheap_switch_scores(boltz_env, poses.df, "s5_dynamicmpnn_sequence", "s5_5_cheap", "s5_5_cheap_prescore", holo["sequence"], apo["sequence"])
        poses.df = poses.df.merge(cheap, on="poses_description", how="left")
        poses.df.to_csv(os.path.join(OUTPUTS, "s5_5_cheap_prescore_all.csv"), index=False)
        funnel.log("s5_5_cheap_scored", len(poses.df), "no-MSA holo+apo switch proxy, all DynamicMPNN seqs")

        poses.filter_poses_by_rank(
            n=POST_DMPNN_TOP_K, score_col="s5_5_cheap_switch_proxy",
            group_col="s1_rfd3_holo_description", ascending=False,
            prefix="s5_5_post_dmpnn_topk",
        )
        funnel.log("s5_5_post_dmpnn_topk", len(poses.df),
                   f"top {POST_DMPNN_TOP_K}/backbone by no-MSA switch proxy -> Step 6")

    # gate_only: stop here, skipping four-state Boltz-2 scoring, the scramble
    # control and the MSD comparator, and emit the AF2-gate ranking as the run
    # output. Used by the smoke test to exercise the generation and gating path
    # without the Boltz-2 stage.
    if AF2_ENABLED and AF2_GATE_ONLY:
        print("\n[AF2 gate only] Boltz-2 stages skipped (stage 6 onward)")
        if "af2_switch_plddt" in poses.df.columns:
            ranked = poses.df.sort_values("af2_switch_plddt", ascending=False)
            cols = [c for c in ["poses_description", "af2_tier", "af2_switch_plddt",
                    "af2_worst_ipae", "af2_holo_plddt", "af2_apo_plddt",
                    "af2_holo_i_pae", "af2_apo_i_pae"] if c in ranked.columns]
            ranked.to_csv(os.path.join(OUTPUTS, "final_af2_ranked.csv"), index=False)
            print(ranked[cols].head(15).to_string(index=False))
            print(f"\nSaved AF2-gate ranking -> {os.path.join(OUTPUTS, 'final_af2_ranked.csv')}")
        funnel.log("af2_gate_only_exit", len(poses.df), "stopped after AF2 gate (gate_only)")
        from protein_only_evaluation import run_protein_only_evaluation
        run_protein_only_evaluation(OUTPUTS)
        # Parameter sweeps run in gate-only mode and are compared on hit counts
        # and per-backbone evidence, so the same hit artefacts a full run
        # produces are emitted here; otherwise hits_summary_by_arm.csv is absent
        # and the sweep metrics derived from it are empty.
        for _mod, _fn in (("tier_hits", "write_tier_hits"),
                          ("results_report", "write_results")):
            try:
                _m = __import__(_mod)
                getattr(_m, _fn)(OUTPUTS)
            except Exception as _e:
                print(f"  {_mod} skipped: {type(_e).__name__}: {_e}")
        return

    # ProteinMPNN-MSD multi-state design, the comparison method.
    msd_cfg = cfg.get("proteinmpnn_msd", {})
    msd_scores = None

    if msd_cfg.get("script"):
        print("\n[Stage 5b] ProteinMPNN-MSD multi-state sequence design")

        msd_work_dir = os.path.join(OUTPUTS, "s5b_mpnn_msd")
        msd_scores_file = os.path.join(msd_work_dir, "mpnn_msd_scores.csv")

        if os.path.isfile(msd_scores_file):
            print(f"  Loading cached MSD scores from {msd_scores_file}")
            msd_scores = pd.read_csv(msd_scores_file)
        else:
            # MSD branches from the same one-row-per-backbone frame used to
            # launch DynamicMPNN. The comparator is never conditioned on
            # DynamicMPNN's subsequent AF2 survival or top-K selection, which
            # would make an MSD-only advantage unobservable by construction.
            msd_input_df = backbone_level_df.drop_duplicates(
                subset="s1_rfd3_holo_description"
            ).reset_index(drop=True)
            msd_scores = run_proteinmpnn_msd(
                poses_df=msd_input_df,
                holo_col="s1_rfd3_holo_location",
                apo_col="state2_pdb",
                binder_chain_holo=binder_chain,
                binder_chain_apo=apo_binder_chain,
                work_dir=msd_work_dir,
                python_path=os.path.expanduser(msd_cfg["python"]),
                script_path=pipeline_config.resolve_path(msd_cfg["script"], WS),
                parse_script=pipeline_config.resolve_path(msd_cfg["helper_parse"], WS),
                weights_dir=pipeline_config.resolve_path(msd_cfg["weights"], WS),
                num_seqs=MPNN_MSD_NSEQ,
            )
            msd_scores.to_csv(msd_scores_file, index=False)

        # Method label for the DynamicMPNN designs.
        poses.df["method"] = "dynamicmpnn"

        funnel.log("s5b_mpnn_msd", len(msd_scores), f"{MPNN_MSD_NSEQ} seqs/design (comparison)")

        # MSD sequence diversity is merged into the sequence_diversity.json
        # written by stage 5, so that a report can load both methods' statistics
        # from a single file.
        if "sequence" in msd_scores.columns and len(msd_scores) > 1:
            msd_div = seq_io.compute_seq_diversity(msd_scores, "sequence")
            print("  Sequence diversity (ProteinMPNN-MSD):")
            print(f"    {msd_div['n_seqs']} sequences, mean pairwise id: {msd_div['mean_pairwise_identity']:.1%}")
            div_path = os.path.join(OUTPUTS, "sequence_diversity.json")
            div_all = {}
            if os.path.isfile(div_path):
                with open(div_path) as f:
                    div_all = json.load(f)
            div_all["proteinmpnn_msd"] = msd_div
            with open(div_path, "w") as f:
                json.dump(div_all, f, indent=2)

    poses = stages.step6_boltz_scoring(ctx, poses)
    # Scrambled-sequence negative control. The residues of N real DynamicMPNN
    # designs are shuffled, preserving amino-acid composition while destroying
    # sequence identity, and passed through the same four-state scoring. This
    # establishes the chance level of switch_score and delta_iptm: the
    # distribution obtained from a protein-like sequence never designed for these
    # targets. It expresses the ranking thresholds relative to the scrambled
    # null, distinguishing designs from predictor artefacts. Active only when
    # evaluation.n_scramble_controls > 0.
    n_scramble = cfg.get("evaluation", {}).get("n_scramble_controls", 0)
    if n_scramble and n_scramble > 0:
        print(f"\n[Stage 6.5] scrambled-sequence negative control, n={n_scramble}")
        scr = paired_nulls.balanced_scrambles(
            poses.df, sequence_col="s5_dynamicmpnn_sequence",
            backbone_col="s1_rfd3_holo_description", n=int(n_scramble), seed=42,
        )

        scr_defs = [
            ("scr_holo", holo["sequence"]),
            ("scr_apo", apo["sequence"]),
        ]
        scr_scored = boltz_scoring.score_states(boltz_env, scr, "_scr_seq", scr_defs, "scramble_fastas")
        scr = scr.merge(scr_scored, on="poses_description", how="left")
        scr["switch_score"] = scr["scr_holo_iptm"] + scr["scr_apo_iptm"]
        scr["state_balance"] = (scr["scr_holo_iptm"] - scr["scr_apo_iptm"]).abs()
        keep_cols = ["poses_description", "_null_backbone", "_real_design_id", "scr_holo_iptm", "scr_apo_iptm", "switch_score", "state_balance"]
        for metric_col in (
            "scr_holo_ipae", "scr_apo_ipae",
            "scr_holo_plddt_mean", "scr_apo_plddt_mean",
        ):
            if metric_col in scr.columns:
                keep_cols.append(metric_col)
        scr[keep_cols].to_csv(os.path.join(OUTPUTS, "scramble_null.csv"), index=False)
        funnel.log("scramble_null", len(scr), f"{n_scramble} scrambled-seq negative controls (four-state)")

    # Late structure self-consistency: whether the final complex prediction, made
    # with real MSAs on the multi-state DynamicMPNN sequence, resembles the
    # RFdiffusion3 backbone it was designed on. This complements the no-MSA check
    # of stage 3.5B, which screens LigandMPNN proxy sequences before the
    # expensive stages, whereas this validates the final candidates. The binder
    # chain in the Boltz complex output is "A", confirmed against the s6a and s6b
    # CIFs of a completed run. Separate Poses objects and keyed merges are used
    # (see _late_rmsd) so that poses.df is not disturbed.
    if "_holo_target_ref" in poses.df.columns and "_apo_target_ref" in poses.df.columns:
        print("\n[Self-consistency] final complex versus RFdiffusion3 backbone")
        backbone_rmsd = structure_io.make_backbone_rmsd(atoms=["CA"], chains=["A"], jobstarter=cpu_jst_fast)
        late_rmsd_pdb_dir = os.path.join(OUTPUTS, "s6_rmsd_to_design_pdb")

        # Results are merged back on a stable per-row key rather than by
        # position: the inner merge inside BackboneRMSD.run() drops, rather than
        # NaN-fills, any pose whose RMSD sub-job failed, which under positional
        # assignment would misalign every subsequent row. The key is the current
        # poses_description, unique per row at this point, not the "_design_id"
        # of stage 3.5B: DynamicMPNN expanded each backbone into dmpnn_nseq rows
        # sharing one _design_id, so joining on it would produce a cartesian
        # product.
        poses.df["_late_rmsd_key"] = poses.df["poses_description"]

        def _late_rmsd(cif_col: str, ref_col: str, prefix: str, subdir: str) -> None:
            pdb_paths = structure_io._cif_locations_to_pdb(poses.df[cif_col].tolist(), os.path.join(late_rmsd_pdb_dir, subdir))
            side = Poses(poses=pdb_paths, work_dir=OUTPUTS, jobstarter=cpu_jst_fast)
            side.df["_late_rmsd_key"] = poses.df["_late_rmsd_key"].values
            side.df["_ref"] = poses.df[ref_col].values
            side = backbone_rmsd.run(poses=side, prefix=prefix, ref_col="_ref", jobstarter=cpu_jst_fast)
            side_scores = (side.df[["_late_rmsd_key", f"{prefix}_rmsd"]]
                           .rename(columns={f"{prefix}_rmsd": prefix})
                           .drop_duplicates(subset="_late_rmsd_key", keep="first"))
            poses.df = poses.df.merge(side_scores, on="_late_rmsd_key", how="left")

        _late_rmsd("s6a_boltz_holo_location", "_holo_target_ref", "s6a_rmsd_to_design", "holo")
        _late_rmsd("s6b_boltz_apo_location", "_apo_target_ref", "s6b_rmsd_to_design", "apo")

        funnel.log("s6_rmsd_to_design", len(poses.df), "final complex vs. RFD3 backbone, holo + apo")
    else:
        print("  Late structure self-consistency skipped: _holo_target_ref and "
              "_apo_target_ref are unset, requiring the stage 3.5B decoy "
              "references, which need more than one unique backbone.")

    # Boltz-versus-AF2 structural consensus. Agreement between two independently
    # modelled predictions of the same sequence, Boltz-2 diffusion and the AF2
    # Evoformer, is an orthogonal validity signal distinct from either
    # predictor's own confidence. No further AF2 inference is required:
    # af2_{holo,apo}_pdb, captured in stages 5.5 and 7, already hold the AF2
    # structure for this sequence, and this block performs only a CPU-side RMSD.
    # The binder chain in the AF2 PDB is not assumed, colabdesign's output
    # chain-labelling convention being unverified; the chain is identified by
    # matching its residue count to the design's sequence length
    # (_extract_chain_by_length).
    if "af2_holo_pdb" in poses.df.columns and "af2_apo_pdb" in poses.df.columns:
        print("\n[Consensus] Boltz-2 versus AF2 structural agreement")
        ws4_backbone_rmsd = structure_io.make_backbone_rmsd(atoms=["CA"], chains=["A"], jobstarter=cpu_jst_fast)
        ws4_pdb_dir = os.path.join(OUTPUTS, "ws4_consensus_pdb")
        poses.df["_ws4_key"] = poses.df["poses_description"]
        poses.df["_ws4_binder_len"] = poses.df["s5_dynamicmpnn_sequence"].apply(
            lambda s: len(seq_io.require_single_chain_sequence(s, "s5_dynamicmpnn_sequence")))

        def _ws4_rmsd(boltz_cif_col: str, af2_pdb_col: str, prefix: str, subdir: str) -> None:
            sub_dir = os.path.join(ws4_pdb_dir, subdir)
            os.makedirs(sub_dir, exist_ok=True)
            boltz_pdbs, af2_pdbs, keys = [], [], []
            for _, r in poses.df.iterrows():
                cif, af2_pdb = r.get(boltz_cif_col), r.get(af2_pdb_col)
                if not (isinstance(cif, str) and os.path.isfile(cif)
                        and isinstance(af2_pdb, str) and os.path.isfile(af2_pdb)):
                    continue
                key = r["_ws4_key"]
                b_out = os.path.join(sub_dir, f"{key}_boltz.pdb")
                a_out = os.path.join(sub_dir, f"{key}_af2.pdb")
                try:
                    structure_io._extract_binder_to_chain(cif, "A", "A", b_out)  # Boltz binder chain
                    structure_io._extract_chain_by_length(af2_pdb, int(r["_ws4_binder_len"]), "A", a_out)
                except Exception as e:
                    print(f"  Consensus skipped for {key} ({subdir}): {type(e).__name__}: {e}")
                    continue
                boltz_pdbs.append(b_out); af2_pdbs.append(a_out); keys.append(key)
            if not boltz_pdbs:
                print(f"  Consensus {subdir}: no comparable structures, skipping")
                return
            side = Poses(poses=boltz_pdbs, work_dir=OUTPUTS, jobstarter=cpu_jst_fast)
            side.df["_ws4_key"] = keys
            side.df["_ref"] = af2_pdbs
            side = ws4_backbone_rmsd.run(poses=side, prefix=prefix, ref_col="_ref", jobstarter=cpu_jst_fast)
            side_scores = (side.df[["_ws4_key", f"{prefix}_rmsd"]]
                           .rename(columns={f"{prefix}_rmsd": prefix})
                           .drop_duplicates(subset="_ws4_key", keep="first"))
            poses.df = poses.df.merge(side_scores, on="_ws4_key", how="left")

        _ws4_rmsd("s6a_boltz_holo_location", "af2_holo_pdb", "af2_rmsd_boltz_vs_af2_holo", "holo")
        _ws4_rmsd("s6b_boltz_apo_location", "af2_apo_pdb", "af2_rmsd_boltz_vs_af2_apo", "apo")
        funnel.log("ws4_consensus_rmsd", len(poses.df), "Boltz-vs-AF2 structural RMSD, holo + apo")
    else:
        print("  Consensus skipped: af2_{holo,apo}_pdb unset (af2.enabled is false, "
              "or the run predates structure capture)")

    poses.df.to_csv(os.path.join(OUTPUTS, "s6_all_scored.csv"), index=False)

    print("\n[Ranking]")

    df = poses.df.copy()

    _h = df["s6a_boltz_holo_iptm"]
    _a = df["s6b_boltz_apo_iptm"]
    # switch_score, the additive sum, is retained as a display column only. A
    # switch requires both states to succeed, which a sum encodes poorly, since a
    # strong holo state masks a failed apo state. Ranking and gating use
    # switch_harmonic, the harmonic mean 2ha/(h+a), which collapses towards zero
    # if either state is weak, so that a balanced (0.75, 0.72) design ranks above
    # a lopsided (0.95, 0.55) one. switch_min is reported as the most
    # conservative conjunctive encoding.
    df["switch_score"] = _h + _a
    df["switch_harmonic"] = np.where((_h + _a) > 0, 2 * _h * _a / (_h + _a), 0.0)
    df["switch_min"] = np.minimum(_h, _a)
    df["state_balance"] = (_h - _a).abs()

    # The Boltz affinity property writes {prefix}_affinity_pred_value and
    # {prefix}_affinity_probability_binary rather than a plain {prefix}_affinity
    # column, verified against s6_all_scored.csv output.
    has_holo_affinity = "s6a_boltz_holo_affinity_pred_value" in df.columns

    # Tiered ranking. The thresholds are defined once here, written to
    # thresholds.json, and consumed both below and by protein_only_evaluation.py,
    # so that the figures cannot drift from the values the pipeline ranked by.
    # Boltz pLDDT is on a 0-1 scale, verified against .npz output spanning
    # approximately 0.23-0.97, so the pLDDT gate is 0.70 rather than 70.
    # Interface-PAE gates follow Bennett et al. (2023): pae_interaction below
    # approximately 10 A marks a confident designed interface and discriminates
    # binders more directly than ipTM. They are applied only where the PAE
    # column is present, NaN-safely below.
    CONSENSUS_RMSD_THRESHOLD = 2.0
    strict_abs = dict(switch_gating.STRICT_ABS)
    strict_abs.update(af2_cfg.get("strict_abs") or {})
    null_specs = {
        "af2_holo_plddt": "higher", "af2_apo_plddt": "higher",
        "af2_holo_i_pae": "lower", "af2_apo_i_pae": "lower",
    }
    observed_null_thresholds = (
        switch_gating.null_thresholds(af2_null, null_specs)
        if af2_null is not None and len(af2_null) else {}
    )
    # Records what the veto did in this run rather than a constant: a run with
    # require_null_separation false previously recorded True here without
    # enforcing it, leaving the tier files uninterpretable after the fact.
    _null_required = bool(sp.get(
        "require_null_separation",
        cfg.get("evaluation", {}).get("require_null_separation", True),
    ))
    recorded_thresholds = {
        "pipeline_mode": "protein_only_two_state",
        "strict": {
            "both_states_required": True,
            "af2_plddt_gt": float(strict_abs["plddt"]),
            "af2_i_pae_lt": float(strict_abs["i_pae"]),
            "af2_i_ptm_gt": float(strict_abs["i_ptm"]),
            "paired_null_discrimination_required": _null_required,
        },
        "relaxed": {
            "both_states_required": True,
            "null_plddt_quantile": 0.95,
            "null_i_pae_quantile": 0.05,
            "observed_thresholds": observed_null_thresholds,
            "paired_null_discrimination_required": _null_required,
        },
        "paired_null": {
            "minimum_auc": float(cfg.get("evaluation", {}).get("min_null_auc", 0.70)),
            "minimum_backbone_pairs": int(cfg.get("evaluation", {}).get("min_null_pairs", 20)),
            "minimum_paired_win_rate": 0.70,
            "bootstrap_ci_low_gt": 0.50,
        },
        "consensus": {
            "boltz_vs_af2_rmsd_lt_angstrom": CONSENSUS_RMSD_THRESHOLD,
            "both_states_required": True,
        },
    }
    with open(os.path.join(OUTPUTS, "thresholds.json"), "w") as f:
        json.dump(recorded_thresholds, f, indent=2)
    if "af2_strict" in df.columns:
        # The final tiers are anchored to the null-validated AF2 gate; Boltz
        # metrics remain orthogonal diagnostics rather than gates.
        #
        # The paired-null verdict vetoes the tiers only when the run requests it.
        # The conjunction was previously unconditional, which made
        # evaluation.require_null_separation self-contradictory: setting it false
        # allowed the run to continue past the stage-5.5 stop and pay for Boltz
        # scoring, yet still wrote empty final_{strict,relaxed,consensus}.csv
        # files. The flag disabled the stop but not the veto, so a nogate run
        # reported none of the hits it had found. It is resolved as in stage 5.5,
        # smoke override first, so that both sites read one setting.
        null_supported = df.get("af2_null_discriminates", pd.Series(False, index=df.index)).fillna(False)
        require_null_tiers = bool(sp.get(
            "require_null_separation",
            cfg.get("evaluation", {}).get("require_null_separation", True),
        ))
        strict_mask = df["af2_strict"].fillna(False)
        relaxed_mask = df["af2_relaxed"].fillna(False) | df["af2_strict"].fillna(False)
        if require_null_tiers:
            strict_mask &= null_supported
            relaxed_mask &= null_supported

    # Consensus tier: the strongest in-silico validity signal available here. It
    # requires two orthogonal predictors, Boltz-2 diffusion and the AF2
    # Evoformer, to agree both in confidence, through each predictor's own
    # relaxed gate, and in structure, their predictions for the same final
    # sequence lying within CONSENSUS_RMSD_THRESHOLD. NaN in the RMSD columns,
    # from a failed extraction or af2.enabled false, excludes a design from the
    # tier, since NaN < threshold is False in pandas. This is the opposite
    # convention to the PAE gates above, because consensus requires positive
    # evidence of structural agreement rather than the absence of disagreement.
    consensus_mask = relaxed_mask.copy()
    if "af2_relaxed" in df.columns and "af2_strict" in df.columns:
        consensus_mask &= (
            df["af2_relaxed"].fillna(False) | df["af2_strict"].fillna(False)
        )
    if "af2_rmsd_boltz_vs_af2_holo" in df.columns:
        consensus_mask &= df["af2_rmsd_boltz_vs_af2_holo"] < CONSENSUS_RMSD_THRESHOLD
    if "af2_rmsd_boltz_vs_af2_apo" in df.columns:
        consensus_mask &= df["af2_rmsd_boltz_vs_af2_apo"] < CONSENSUS_RMSD_THRESHOLD
    df["consensus_tier"] = consensus_mask

    # Ranking uses the harmonic switch score regardless of whether the
    # four-state controls ran; the additive switch_score and selectivity_score
    # are display columns.
    rank_col = "af2_switch_plddt" if "af2_switch_plddt" in df.columns else "switch_harmonic"

    display_cols = [
        "poses_description",
        "s6a_boltz_holo_iptm",
        "s6b_boltz_apo_iptm",
        "state_balance",
        "switch_harmonic",
        "switch_min",
        "switch_score",
    ]
    if "s6a_boltz_holo_ipae" in df.columns:
        display_cols.append("s6a_boltz_holo_ipae")
    if "s6b_boltz_apo_ipae" in df.columns:
        display_cols.append("s6b_boltz_apo_ipae")
    if "s6a_boltz_holo_plddt_mean" in df.columns:
        display_cols.append("s6a_boltz_holo_plddt_mean")
    if "s6b_boltz_apo_plddt_mean" in df.columns:
        display_cols.append("s6b_boltz_apo_plddt_mean")
    if has_holo_affinity:
        display_cols.append("s6a_boltz_holo_affinity_pred_value")
    if "s6a_rmsd_to_design" in df.columns:
        display_cols += ["s6a_rmsd_to_design", "s6b_rmsd_to_design"]
    if "af2_rmsd_boltz_vs_af2_holo" in df.columns:
        display_cols += ["af2_rmsd_boltz_vs_af2_holo", "af2_rmsd_boltz_vs_af2_apo", "consensus_tier"]

    strict = df[strict_mask].sort_values(rank_col, ascending=False)
    relaxed = df[relaxed_mask].sort_values(rank_col, ascending=False)
    consensus = df[consensus_mask].sort_values(rank_col, ascending=False)
    all_ranked = df.sort_values(rank_col, ascending=False)

    strict.to_csv(os.path.join(OUTPUTS, "final_strict.csv"), index=False)
    relaxed.to_csv(os.path.join(OUTPUTS, "final_relaxed.csv"), index=False)
    consensus.to_csv(os.path.join(OUTPUTS, "final_consensus.csv"), index=False)
    all_ranked.to_csv(os.path.join(OUTPUTS, "final_all_ranked.csv"), index=False)

    funnel.log("rank_strict", len(strict),
               "AF2 strict in both states AND paired-null discrimination; ranked by AF2 harmonic pLDDT")
    funnel.log("rank_relaxed", len(relaxed),
               "AF2 relaxed-or-strict in both states AND paired-null discrimination; ranked by AF2 harmonic pLDDT")
    funnel.log("rank_consensus", len(consensus),
               f"final AF2 tier and Boltz-vs-AF2 RMSD<{CONSENSUS_RMSD_THRESHOLD}A in both states")

    print(f"\nStrict candidates: {len(strict)}")
    if len(strict) > 0:
        print(strict[display_cols].head(10).to_string())
    else:
        print("No designs met strict thresholds.")
        print(f"\nRelaxed candidates: {len(relaxed)}")
        if len(relaxed) > 0:
            print(relaxed[display_cols].head(10).to_string())
        else:
            print("No designs met relaxed thresholds either.")
            print(f"\nAll designs ranked by {rank_col}:")
            print(all_ranked[display_cols].head(10).to_string())

    print(f"\nOutputs saved to {OUTPUTS}/")
    print(f"  final_strict.csv      ({len(strict)} designs)")
    print(f"  final_relaxed.csv     ({len(relaxed)} designs)")
    print(f"  final_consensus.csv   ({len(consensus)} designs; Boltz-AF2 agreement)")
    print(f"  final_all_ranked.csv  ({len(all_ranked)} designs)")
    print("  s3_5b_selfcons_all_scored.csv (pre-filter self-consistency snapshot)")
    print("  s6_all_scored.csv     (pre-ranking snapshot)")
    print("  funnel_summary.csv    (per-stage counts)")

    # Specificity control against negative targets. No upstream stage tests that
    # a binder fails where it should. Each relaxed-tier finalist is scored
    # against an unrelated decoy protein, using the same scoring path in a single
    # state, and holo_iptm is required to exceed decoy_iptm by a margin. The cost
    # is bounded to relaxed-tier candidates, capped at
    # specificity.max_candidates, falling back to the top of all_ranked when the
    # relaxed tier is empty so that smoke and early runs still exercise the path.
    if SPECIFICITY_ENABLED and DECOY_TARGETS:
        print(f"\n[Stage 8] specificity control against {len(DECOY_TARGETS)} decoy target(s)")
        spec_candidates = relaxed if len(relaxed) > 0 else all_ranked
        spec_candidates = spec_candidates.head(int(SPECIFICITY_MAX_CANDIDATES)).copy()
        if len(spec_candidates) == 0:
            print("  No candidates to score: relaxed and all_ranked are both empty")
        else:
            decoy_defs = [
                (f"s8_decoy_{_dt['name']}", _dt["sequence"])
                for _dt in DECOY_TARGETS
            ]
            decoy_scored = boltz_scoring.score_states(boltz_env, 
                spec_candidates, "s5_dynamicmpnn_sequence", decoy_defs, "s8_decoy_fastas")
            spec_candidates = spec_candidates.merge(decoy_scored, on="poses_description", how="left")
            decoy_iptm_cols = [f"s8_decoy_{_dt['name']}_iptm" for _dt in DECOY_TARGETS]
            present = [c for c in decoy_iptm_cols if c in spec_candidates.columns]
            if present:
                spec_candidates["decoy_iptm_max"] = spec_candidates[present].max(axis=1)
                spec_candidates["specificity_margin"] = (
                    spec_candidates["s6a_boltz_holo_iptm"] - spec_candidates["decoy_iptm_max"])
                spec_candidates["specificity_pass"] = (
                    spec_candidates["specificity_margin"] > SPECIFICITY_MARGIN_THRESHOLD)
                spec_out_cols = (["poses_description", "s6a_boltz_holo_iptm"] + present
                                  + ["decoy_iptm_max", "specificity_margin", "specificity_pass"])
                spec_path = os.path.join(OUTPUTS, "specificity_report.csv")
                spec_candidates[spec_out_cols].sort_values(
                    "specificity_margin", ascending=False).to_csv(spec_path, index=False)
                n_pass = int(spec_candidates["specificity_pass"].sum())
                funnel.log("s8_specificity", len(spec_candidates),
                           f"{n_pass}/{len(spec_candidates)} beat margin>{SPECIFICITY_MARGIN_THRESHOLD} "
                           f"vs {len(DECOY_TARGETS)} decoy(s) (soft gate, display-only)")
                print(f"  specificity_margin > {SPECIFICITY_MARGIN_THRESHOLD}: "
                      f"{n_pass}/{len(spec_candidates)}")
                print(f"  Saved to {spec_path}")
                # The margin and flag are merged back into the final CSV files so
                # that they appear alongside the primary ranking.
                merge_back = spec_candidates[["poses_description", "specificity_margin",
                                               "specificity_pass"]]
                for _name, _frame in (("final_strict.csv", strict), ("final_relaxed.csv", relaxed),
                                       ("final_all_ranked.csv", all_ranked)):
                    _merged = _frame.merge(merge_back, on="poses_description", how="left")
                    _merged.to_csv(os.path.join(OUTPUTS, _name), index=False)
            else:
                print("  Warning: no decoy ipTM columns were scored; specificity_margin omitted")
    elif SPECIFICITY_ENABLED:
        print("\nspecificity.enabled is true but no decoy_targets are configured; "
              "the specificity control is skipped.")

    stages.step7_msd_compare(ctx, af2_env, msd_scores, backbone_level_df)
    stages.run_evaluation_stage(ctx)


if __name__ == "__main__":
    main()
