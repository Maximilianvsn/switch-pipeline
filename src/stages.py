"""Pipeline stages, one function per stage.

Every stage has the signature `(ctx, poses) -> poses` and reads its
configuration from the `PipelineContext` rather than from the enclosing scope of
`main()`.

The stages collected here produce no local value that a later stage reads: they
mutate `poses` and write output files only, so extracting them cannot alter data
flow. Stages that pass values onward remain in `main()` until their outputs are
threaded explicitly.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from protflow.poses import Poses

from af2_runner import run_af2_ig, build_state_requests
import af2_gate
import boltz_interface_evaluation
import boltz_scoring
import method_comparison
import paired_nulls
import switch_gating
import state_pairing


def step3_5a_lmpnn_topk(ctx, poses):
    """Retain the top-K LigandMPNN sequences per backbone."""
    LMPNN_TOP_K, funnel = ctx.LMPNN_TOP_K, ctx.funnel

    # Sequence-level filter. No backbone is rejected: the highest-scoring proxy
    # sequences per backbone are selected on LigandMPNN's own confidence, which
    # has already been computed, so no additional prediction is required.
    print("\n[Stage 3.5A] MPNN proxy confidence filter, top-K per backbone")

    poses.df["_lmpnn_combined_score"] = (
        poses.df["s3_ligandmpnn_overall_confidence"]
        + poses.df.get("s3_ligandmpnn_ligand_confidence", 0)
    )
    poses.filter_poses_by_rank(
        n=LMPNN_TOP_K,
        score_col="_lmpnn_combined_score",
        group_col="s1_rfd3_holo_description",
        ascending=False,
        prefix="s3_5a_lmpnn_topk",
    )
    funnel.log("s3_5a_lmpnn_topk", len(poses.df), f"top {LMPNN_TOP_K}/backbone by LigandMPNN confidence")

    return poses


def step6_boltz_scoring(ctx, poses):
    """Score both states with Boltz-2, each on its own fresh side object."""
    apo, boltz_env, funnel, holo = ctx.apo, ctx.boltz_env, ctx.funnel, ctx.holo

    # Four-state Boltz-2 scoring, each state predicted independently: s6a holo,
    # s6b apo. Every state is scored on a fresh binder-only side object (see
    # score_states) so that no state carries over another's target chain.
    print("\n[Stage 6] four-state Boltz-2 scoring and affinity")

    # The target MSAs are precomputed once for both targets, so that every state
    # below, together with the scramble null and the MSD comparator, reuses them
    # while the binder is scored single-sequence. Cached on resume.
    print("  Precomputing target MSAs, reused across all final-scoring states")
    boltz_scoring.precompute_target_msa(boltz_env, holo["sequence"], "holo_target")
    boltz_scoring.precompute_target_msa(boltz_env, apo["sequence"], "apo_target")

    s6_state_defs = [
        ("s6a_boltz_holo", holo["sequence"]),
        ("s6b_boltz_apo", apo["sequence"]),
    ]
    s6_scored = boltz_scoring.score_states(boltz_env, poses.df, "s5_dynamicmpnn_sequence", s6_state_defs, "s6_binder_fastas")
    poses.df = poses.df.merge(s6_scored, on="poses_description", how="left")
    funnel.log("s6_four_state", len(poses.df),
               f"{len(s6_state_defs)}-state scored independently (no chain carry-over)")

    return poses


def step2_5_select_state2(ctx, poses):
    """Score state-2 candidates for designability, one selected per state-1 backbone.

    This stage is the narrowest point of the funnel: a small fraction of
    geometry-passing candidates are designable, and most surviving backbones do
    so through a single candidate.
    """
    AF2_ENABLED, AF2_PARAMS_DIR, APO_BATCH, OUTPUTS = ctx.AF2_ENABLED, ctx.AF2_PARAMS_DIR, ctx.APO_BATCH, ctx.OUTPUTS
    af2_cfg, cpu_jst, funnel, ligandmpnn = ctx.af2_cfg, ctx.cpu_jst, ctx.funnel, ctx.ligandmpnn
    sp = ctx.sp

    # Independent state-2 designability and candidate selection. Each
    # geometrically eligible state-2 backbone is optimised independently with
    # ProteinMPNN proxy sequences, a fixed top-K is evaluated by AF2
    # initial-guess, and exactly one state-2 candidate is selected per state-1
    # backbone. The proxy sequences are then discarded: DynamicMPNN and MSD
    # receive only the selected backbone pair and solve the shared-sequence
    # problem independently.
    s2_desig_cfg = af2_cfg.get("state2_designability", {})
    if APO_BATCH > 1 and not (
        AF2_ENABLED and s2_desig_cfg.get("enabled", False)
    ):
        raise RuntimeError(
            "apo_batch > 1 requires af2.state2_designability.enabled=true so "
            "multiple state-2 candidates can be collapsed without ambiguity"
        )

    if AF2_ENABLED and s2_desig_cfg.get("enabled", False):
        print("\n[Stage 2.5] ProteinMPNN to AF2 designability, state-2 candidates")

        s2_n_seqs = int(
            sp.get("state2_designability_n_seqs", s2_desig_cfg.get("n_seqs", 8))
        )
        s2_pre_af2_top_k = int(
            sp.get(
                "state2_designability_pre_af2_top_k",
                s2_desig_cfg.get("pre_af2_top_k", 4),
            )
        )
        s2_pre_af2_top_k = max(1, min(s2_pre_af2_top_k, s2_n_seqs))
        s2_min_plddt = float(
            sp.get(
                "state2_designability_min_plddt",
                s2_desig_cfg.get("min_plddt", 0.70),
            )
        )
        s2_max_i_pae = float(
            sp.get(
                "state2_designability_max_i_pae",
                s2_desig_cfg.get("max_i_pae", 1.0),
            )
        )
        s2_min_i_ptm = float(
            sp.get(
                "state2_designability_min_i_ptm",
                s2_desig_cfg.get("min_i_ptm", 0.0),
            )
        )
        s2_target_rmsd = float(s2_desig_cfg.get("target_rmsd", 3.0))

        s2_design_poses = Poses(
            poses=poses.df["state2_pdb"].tolist(),
            work_dir=OUTPUTS,
            jobstarter=cpu_jst,
        )
        s2_design_poses.df["_s2_pair_id"] = poses.df["state_pair_id"].values
        s2_design_poses.df["_s2_state1_id"] = poses.df[
            "s1_rfd3_holo_description"
        ].values
        s2_design_poses.df["_s2_backbone_pdb"] = poses.df["state2_pdb"].values
        s2_design_poses = ligandmpnn.run(
            poses=s2_design_poses,
            prefix="s2_5_desig_lmpnn",
            jobstarter=cpu_jst,
            nseq=s2_n_seqs,
            model_type="protein_mpnn",
            options="--chains_to_design A ",
        )
        s2_proxy_conf_col = "s2_5_desig_lmpnn_overall_confidence"
        s2_design_poses.df = (
            s2_design_poses.df.sort_values(
                ["_s2_pair_id", s2_proxy_conf_col, "poses_description"],
                ascending=[True, False, True],
            )
            .groupby("_s2_pair_id", group_keys=False)
            .head(s2_pre_af2_top_k)
            .reset_index(drop=True)
        )

        s2_desig_reqs = build_state_requests(
            s2_design_poses.df,
            "poses_description",
            "_s2_backbone_pdb",
            "B",
            "A",
            "s2_5_desig_lmpnn_sequence",
            os.path.join(OUTPUTS, "s2_5_desig_af2", "bb"),
            "s2desig__",
        )
        s2_desig_scores = run_af2_ig(
            s2_desig_reqs,
            os.path.join(OUTPUTS, "s2_5_desig_af2", "af2"),
            af2_cfg,
            AF2_PARAMS_DIR,
        )
        s2_desig_scores = s2_desig_scores.merge(
            s2_desig_reqs[["id", "_orig_id"]], on="id", how="left"
        )
        s2_design_poses.df = s2_design_poses.df.merge(
            s2_desig_scores.rename(
                columns={"_orig_id": "poses_description"}
            )[["poses_description", "plddt", "i_pae", "i_ptm"]],
            on="poses_description",
            how="left",
        )
        s2_design_poses.df["_individual_designability_score"] = (
            pd.to_numeric(s2_design_poses.df["plddt"], errors="coerce")
            * np.sqrt(
                pd.to_numeric(
                    s2_design_poses.df["i_ptm"], errors="coerce"
                ).clip(lower=0)
                * np.exp(
                    -3.1
                    * pd.to_numeric(
                        s2_design_poses.df["i_pae"], errors="coerce"
                    )
                )
            )
        )
        best_proxy_rows = (
            s2_design_poses.df.sort_values(
                [
                    "_s2_pair_id",
                    "_individual_designability_score",
                    "plddt",
                    "i_pae",
                    "poses_description",
                ],
                ascending=[True, False, False, True, True],
                na_position="last",
            )
            .groupby("_s2_pair_id", group_keys=False)
            .head(1)
            .rename(
                columns={
                    "plddt": "state2_designability_plddt",
                    "i_pae": "state2_designability_i_pae",
                    "i_ptm": "state2_designability_i_ptm",
                    "_individual_designability_score": "state2_designability_score",
                    "s2_5_desig_lmpnn_sequence": "state2_designability_proxy_sequence",
                }
            )
        )
        poses.df = poses.df.merge(
            best_proxy_rows[
                [
                    "_s2_pair_id",
                    "state2_designability_plddt",
                    "state2_designability_i_pae",
                    "state2_designability_i_ptm",
                    "state2_designability_score",
                    "state2_designability_proxy_sequence",
                ]
            ].rename(columns={"_s2_pair_id": "state_pair_id"}),
            on="state_pair_id",
            how="left",
            validate="one_to_one",
        )

        state1_plddt = pd.to_numeric(
            poses.df.get("state1_designability_plddt"), errors="coerce"
        )
        state2_plddt = pd.to_numeric(
            poses.df["state2_designability_plddt"], errors="coerce"
        )
        state1_score = pd.to_numeric(
            poses.df.get("state1_designability_score"), errors="coerce"
        )
        state2_score = pd.to_numeric(
            poses.df["state2_designability_score"], errors="coerce"
        )
        poses.df["pair_individual_designability_plddt"] = np.where(
            (state1_plddt > 0) & (state2_plddt > 0),
            2.0 * state1_plddt * state2_plddt / (state1_plddt + state2_plddt),
            np.nan,
        )
        poses.df["pair_individual_designability_score"] = np.where(
            (state1_score > 0) & (state2_score > 0),
            2.0 * state1_score * state2_score / (state1_score + state2_score),
            np.nan,
        )
        state2_i_pae = pd.to_numeric(
            poses.df["state2_designability_i_pae"], errors="coerce"
        )
        state2_i_ptm = pd.to_numeric(
            poses.df["state2_designability_i_ptm"], errors="coerce"
        )
        poses.df["state2_designability_pass"] = (
            (state2_plddt > s2_min_plddt)
            & (state2_i_pae < s2_max_i_pae)
            & (state2_i_ptm > s2_min_i_ptm)
        )
        eligible_s2 = poses.df[poses.df["state2_designability_pass"]].reset_index(
            drop=True
        )
        if eligible_s2.empty:
            poses.df.assign(state2_selected=False).to_csv(
                os.path.join(OUTPUTS, "s2_5_state2_designability.csv"), index=False
            )
            raise RuntimeError(
                "AF2 designability rejected every geometrically valid state-2 "
                "candidate; inspect s2_5_state2_designability.csv"
            )

        ranked_s2 = state_pairing.select_best_state2_candidate(
            eligible_s2,
            score_col="state2_designability_score",
            target_rmsd=s2_target_rmsd,
        )
        selected_pair_ids = set(
            ranked_s2.loc[ranked_s2["state2_selected"], "state_pair_id"]
        )
        audit_s2 = poses.df.copy()
        audit_s2["state2_selected"] = audit_s2["state_pair_id"].isin(selected_pair_ids)
        audit_s2 = audit_s2.merge(
            ranked_s2[["state_pair_id", "state2_candidate_rank"]],
            on="state_pair_id",
            how="left",
            validate="one_to_one",
        )
        audit_s2.to_csv(
            os.path.join(OUTPUTS, "s2_5_state2_designability.csv"), index=False
        )
        n_state1_before = poses.df["s1_rfd3_holo_description"].nunique()
        poses.df = ranked_s2[ranked_s2["state2_selected"]].reset_index(drop=True)
        funnel.log(
            "s2_5_state2_designability",
            len(poses.df),
            f"ProteinMPNN {s2_n_seqs}/candidate -> AF2 top "
            f"{s2_pre_af2_top_k}; pLDDT>{s2_min_plddt}, "
            f"iPAE<{s2_max_i_pae}, ipTM>{s2_min_i_ptm}; selected 1 state-2 "
            f"for {len(poses.df)}/{n_state1_before} state-1 backbones",
        )

    return poses


def step7_msd_compare(ctx, af2_env, msd_scores, backbone_level_df):
    """Score the ProteinMPNN-MSD arm on the same footing as DynamicMPNN.

    The MSD sequences and the backbone-level frame produced by stage 5b are
    passed as explicit arguments rather than read from the enclosing scope.
    """
    AF2_ENABLED, OUTPUTS, POST_AF2_PER_BACKBONE, POST_AF2_TOP_K = ctx.AF2_ENABLED, ctx.OUTPUTS, ctx.POST_AF2_PER_BACKBONE, ctx.POST_AF2_TOP_K
    POST_DMPNN_TOP_K, af2_cfg, apo, boltz_env = ctx.POST_DMPNN_TOP_K, ctx.af2_cfg, ctx.apo, ctx.boltz_env
    cfg, funnel, gpu_jst, holo = ctx.cfg, ctx.funnel, ctx.gpu_jst, ctx.holo
    sp = ctx.sp

    # Scoring of the ProteinMPNN-MSD designs, enabled by configuration and off by
    # default. Setting evaluation.score_msd true and resubmitting under the same
    # --run-name loads stages 1 to 6 from the existing ProtFlow scorefiles and
    # the stage-5b sequences from the cached mpnn_msd_scores.csv, so that only
    # this stage is computed.
    score_msd = cfg.get("evaluation", {}).get("score_msd", False)

    if score_msd and msd_scores is not None:
        print("\n[Stage 7] Boltz-2 scoring of ProteinMPNN-MSD designs")

        msd_poses = Poses(
            poses=msd_scores["location"].tolist(),
            work_dir=OUTPUTS,
            jobstarter=gpu_jst,
        )
        # Backbone lineage and binder sequence are carried through for
        # pre-selection, four-state scoring and the method comparison.
        msd_poses.df = msd_poses.df.merge(
            msd_scores[["description", "backbone", "sequence"]].rename(
                columns={"description": "poses_description", "sequence": "_msd_seq"}),
            on="poses_description", how="left",
        )

        # The MSD arm passes through the same AF2 initial-guess gate as stage
        # 5.5, with the same scoring call, the same null and the same
        # switch_gating tiers, so that the comparison is matched. An earlier
        # version used the no-MSA Boltz proxy (cheap_switch_scores), which
        # real-versus-scramble testing showed to be non-discriminative
        # (AUC ~ 0.5), so MSD candidates were forwarded at random while the
        # DynamicMPNN candidates were AF2-selected.
        #
        # af2_gate.af2_gate_score requires s1_rfd3_holo_location and state2_pdb
        # columns on the frame it is given. msd_poses.df carries only the
        # originating backbone's description string, in "backbone", set in the
        # row dictionary of run_proteinmpnn_msd. The paths are therefore taken
        # from backbone_level_df, the one-row-per-backbone frame recorded
        # immediately before the stage-5 DynamicMPNN call, which holds every
        # backbone forwarded to sequence design, unfiltered by the later
        # AF2-gate top-K cut. poses.df is not a safe source here, as it may
        # already have dropped an MSD design's originating backbone.
        if AF2_ENABLED:
            print("  Pre-selection: AF2 initial-guess gate on MSD sequences, as in stage 5.5")
            bb_paths = backbone_level_df.drop_duplicates(subset="s1_rfd3_holo_description").set_index(
                "s1_rfd3_holo_description")[["s1_rfd3_holo_location", "state2_pdb"]]
            msd_poses.df = msd_poses.df.merge(
                bb_paths, left_on="backbone", right_index=True, how="left")
            n_missing_bb = int(msd_poses.df["s1_rfd3_holo_location"].isna().sum())
            if n_missing_bb:
                print(f"  Warning: {n_missing_bb} MSD design(s) reference a backbone absent from "
                      f"backbone_level_df; these score NaN and fail the gate")
            af2_msd = af2_gate.af2_gate_score(af2_env, msd_poses.df, "_msd_seq", "s7_5_af2_gate")
            msd_poses.df = msd_poses.df.merge(af2_msd, on="poses_description", how="left")

            # Method-matched null: MSD is calibrated against scrambles of its own
            # sequences at the same per-backbone multiplicity. The DynamicMPNN
            # composition distribution is not a valid null for MSD.
            msd_af2_null = None
            msd_null_gate_pass = False
            msd_n_scr = cfg.get("evaluation", {}).get("n_scramble_controls", 0)
            msd_min_null_auc = float(sp.get(
                "min_null_auc", cfg.get("evaluation", {}).get("min_null_auc", 0.70)
            ))
            msd_min_null_pairs = int(sp.get(
                "min_null_pairs", cfg.get("evaluation", {}).get("min_null_pairs", 10)
            ))
            msd_require_null = bool(sp.get(
                "require_null_separation",
                cfg.get("evaluation", {}).get("require_null_separation", True),
            ))
            if msd_n_scr and msd_n_scr > 0:
                msd_scr = paired_nulls.balanced_scrambles(
                    msd_poses.df, sequence_col="_msd_seq", backbone_col="backbone",
                    n=int(msd_n_scr), seed=43,
                )
                msd_null_keys = msd_scr[[
                    "poses_description", "_null_backbone", "_real_design_id"
                ]].copy()
                msd_af2_null = af2_gate.af2_gate_score(af2_env, 
                    msd_scr, "_scr_seq", "s7_5_af2_gate_null"
                )
                msd_af2_null = msd_af2_null.merge(
                    msd_null_keys, on="poses_description", how="left", validate="one_to_one"
                )
                msd_af2_null.to_csv(
                    os.path.join(OUTPUTS, "msd_af2_gate_null.csv"), index=False
                )

            msd_poses.df = switch_gating.assign_af2_tiers(
                msd_poses.df, "af2_holo_plddt", "af2_apo_plddt", "af2_holo_i_pae", "af2_apo_i_pae",
                null_df=msd_af2_null, strict_abs=af2_cfg.get("strict_abs"),
                holo_iptm="af2_holo_i_ptm", apo_iptm="af2_apo_i_ptm",
            )
            if msd_af2_null is not None and len(msd_af2_null):
                msd_separation = paired_nulls.separation_table(
                    msd_poses.df, msd_af2_null,
                    {
                        "af2_holo_plddt": "higher", "af2_apo_plddt": "higher",
                        "af2_holo_i_pae": "lower", "af2_apo_i_pae": "lower",
                    },
                    real_backbone_col="backbone",
                )
                msd_separation.to_csv(
                    os.path.join(OUTPUTS, "msd_af2_null_separation.csv"), index=False
                )
                msd_null_gate_pass = paired_nulls.passes_stop_go(
                    msd_separation, min_auc=msd_min_null_auc, min_pairs=msd_min_null_pairs
                )
            msd_poses.df["af2_null_discriminates"] = bool(msd_null_gate_pass)
            if msd_require_null and not msd_null_gate_pass:
                msd_poses.df["af2_relaxed"] = False
                msd_poses.df["af2_tier"] = np.where(
                    msd_poses.df["af2_strict"], "strict", "fail"
                )
            msd_poses.df.to_csv(os.path.join(OUTPUTS, "s7_5_msd_af2_gate_all.csv"), index=False)
            n_strict_msd = int(msd_poses.df.get("af2_strict", pd.Series(dtype=bool)).sum())
            n_relaxed_msd = int(msd_poses.df.get("af2_relaxed", pd.Series(dtype=bool)).sum())
            funnel.log("s7_5_msd_af2_gate", len(msd_poses.df),
                       f"AF2-IG scored MSD both states; strict={n_strict_msd} relaxed={n_relaxed_msd}")
            if sp.get("forward_only_af2_passes", cfg.get("evaluation", {}).get("forward_only_af2_passes", True)):
                msd_poses.df = msd_poses.df[msd_poses.df["af2_tier"] != "fail"].reset_index(drop=True)
                if msd_poses.df.empty:
                    raise RuntimeError("No ProteinMPNN-MSD sequence passed the same two-state AF2 gate; comparator scoring stopped.")
            if POST_AF2_PER_BACKBONE and POST_AF2_PER_BACKBONE > 0:
                msd_poses.df = (msd_poses.df.sort_values("af2_switch_plddt", ascending=False)
                                .groupby("backbone", group_keys=False)
                                .head(POST_AF2_PER_BACKBONE).reset_index(drop=True))
            if POST_AF2_TOP_K and POST_AF2_TOP_K > 0 and len(msd_poses.df) > POST_AF2_TOP_K:
                msd_poses.df = (msd_poses.df.sort_values("af2_switch_plddt", ascending=False)
                                .head(POST_AF2_TOP_K).reset_index(drop=True))
            funnel.log("s7_5_msd_af2_forwarded", len(msd_poses.df),
                       f"top {POST_AF2_TOP_K} MSD by AF2 harmonic pLDDT -> expensive scoring")

        elif POST_DMPNN_TOP_K and POST_DMPNN_TOP_K > 0:
            print("  Pre-selection: no-MSA switch proxy on MSD sequences (proxy metric)")
            cheap_msd = boltz_scoring.cheap_switch_scores(boltz_env, msd_poses.df, "_msd_seq", "s7_5_cheap", "s7_5_cheap_prescore", holo["sequence"], apo["sequence"])
            msd_poses.df = msd_poses.df.merge(cheap_msd, on="poses_description", how="left")
            msd_poses.df.to_csv(os.path.join(OUTPUTS, "s7_5_msd_cheap_prescore_all.csv"), index=False)
            msd_poses.filter_poses_by_rank(
                n=POST_DMPNN_TOP_K, score_col="s7_5_cheap_switch_proxy",
                group_col="backbone", ascending=False, prefix="s7_5_msd_topk",
            )
            funnel.log("s7_5_msd_topk", len(msd_poses.df),
                       f"top {POST_DMPNN_TOP_K}/backbone MSD by no-MSA switch proxy")

        # Four-state scoring identical to stage 6, using the same helper and the
        # same states, so that DynamicMPNN and MSD are compared on matched terms,
        # including the off-diagonal selectivity controls.
        s7_state_defs = [
            ("s7a_msd_boltz_holo", holo["sequence"]),
            ("s7b_msd_boltz_apo", apo["sequence"]),
        ]
        s7_scored = boltz_scoring.score_states(boltz_env, msd_poses.df, "_msd_seq", s7_state_defs, "s7_binder_fastas")
        msd_df = msd_poses.df.merge(s7_scored, on="poses_description", how="left")
        funnel.log("s7_msd_four_state", len(msd_df), f"{len(s7_state_defs)}-state MSD scoring")

        _hm = msd_df["s7a_msd_boltz_holo_iptm"]
        _am = msd_df["s7b_msd_boltz_apo_iptm"]
        # switch_score is a display column only; ranking uses switch_harmonic, as
        # in the DynamicMPNN ranking, for a like-for-like comparison.
        msd_df["switch_score"] = _hm + _am
        msd_df["switch_harmonic"] = np.where((_hm + _am) > 0, 2 * _hm * _am / (_hm + _am), 0.0)
        msd_df["state_balance"] = (_hm - _am).abs()
        msd_rank_col = "af2_switch_plddt" if "af2_switch_plddt" in msd_df.columns else "switch_harmonic"
        msd_df = msd_df.sort_values(msd_rank_col, ascending=False)

        msd_out_path = os.path.join(OUTPUTS, "msd_final_all_ranked.csv")
        msd_df.to_csv(msd_out_path, index=False)

        print(f"\nProteinMPNN-MSD scored designs: {len(msd_df)}")
        print(
            msd_df[["poses_description", "s7a_msd_boltz_holo_iptm", "s7b_msd_boltz_apo_iptm",
                    "state_balance", "switch_harmonic", "switch_score"]]
            .head(10)
            .to_string()
        )
        print(f"\nSaved to {msd_out_path}")
        print("\nCompare against the DynamicMPNN results in final_all_ranked.csv")
        dynamic_gate_csv = os.path.join(OUTPUTS, "s5_5_af2_gate_all.csv")
        msd_gate_csv = os.path.join(OUTPUTS, "s7_5_msd_af2_gate_all.csv")
        if os.path.isfile(dynamic_gate_csv) and os.path.isfile(msd_gate_csv):
            method_summary, paired_methods = method_comparison.write_backbone_comparison(
                dynamic_gate_csv, msd_gate_csv, OUTPUTS
            )
            print("\nBackbone-clustered method comparison:")
            print(method_summary.to_string(index=False))
            funnel.log("method_comparison", len(paired_methods),
                       "paired backbones with equal initial sequence budgets")

        boltz_interface_status = boltz_interface_evaluation.write_boltz_interface_evaluation(OUTPUTS)
        if boltz_interface_status.get("available"):
            boltz_validated = bool(boltz_interface_status.get("validated_against_null", False))
            boltz_pairs = int(boltz_interface_status.get("n_paired_backbones", 0))
            print("\nFinal Boltz interface metric:")
            print(f"  validated against paired null: {boltz_validated}")
            print(f"  paired backbones: {boltz_pairs}")
            print("  interpretation: predictor-confidence diagnostic")
            funnel.log("boltz_interface_evaluation", boltz_pairs,
                       f"null_validated={boltz_validated}")
    elif score_msd:
        print("\nevaluation.score_msd is true but no MSD sequences are available "
              "(proteinmpnn_msd.script unconfigured, or stage 5b did not run); "
              "MSD scoring is skipped.")


def run_evaluation_stage(ctx):
    """Gate-level hit files, consensus attribution, then figures and summary.

    Every part is non-fatal: evaluation runs at the end of a multi-hour pipeline,
    so a reporting failure must not discard the results of the run.
    """
    OUTPUTS, funnel = ctx.OUTPUTS, ctx.funnel

    # Figures and summary are generated automatically; no separate
    # `python protein_only_evaluation.py <run-dir>` invocation is required.
    print("\n[Evaluation] figures and summary")

    # Gate-level hit files, per sequence-design arm. final_{strict,relaxed}.csv
    # hold only the post-forwarding top-K, so every other passing sequence would
    # otherwise be scored, tiered and then discarded unwritten.
    try:
        import tier_hits
        hits_table = tier_hits.write_tier_hits(OUTPUTS)
        if not hits_table.empty:
            print("\nGate-level AF2 hits by arm (independent of forwarding budget):")
            _hc = [c for c in ("label", "n_scored", "n_strict", "n_relaxed",
                               "n_backbones_relaxed", "n_relaxed_null",
                               "null_gate_supported") if c in hits_table.columns]
            print(hits_table[_hc].to_string(index=False))
            for _, _r in hits_table.iterrows():
                funnel.log(f"hits_{_r['arm']}", int(_r.get("n_relaxed", 0) or 0),
                           f"AF2 relaxed-tier hits written to hits_{_r['arm']}_relaxed.csv "
                           f"(run-level null stop-go: {_r['null_gate_supported']})")
    except Exception as e:
        print(f"  tier_hits skipped: {type(e).__name__}: {e}")

    # Attribution of the consensus outcome. The af2_rmsd_boltz_vs_af2_* columns
    # alone do not identify which predictor diverged; this adds both comparisons
    # against the design backbone.
    try:
        import consensus_diagnostics
        _cd = consensus_diagnostics.diagnose(OUTPUTS)
        if not _cd.empty:
            _cs = consensus_diagnostics.summarise(_cd)
            _cs.to_csv(os.path.join(OUTPUTS, "consensus_diagnostics_summary.csv"),
                       index=False)
            print("\nConsensus attribution (CA RMSD against the design backbone):")
            print(_cs.round(2).to_string(index=False))
    except Exception as e:
        print(f"  consensus_diagnostics skipped: {type(e).__name__}: {e}")

    # Non-fatal, as elsewhere in this stage: it runs before results_report
    # below, so an exception here, from a missing artefact in an alternative
    # front end or a plotting-backend failure, would otherwise also discard the
    # curated results/ folder.
    try:
        from protein_only_evaluation import run_protein_only_evaluation
        run_protein_only_evaluation(OUTPUTS)
    except Exception as e:
        print(f"  protein_only_evaluation skipped: {type(e).__name__}: {e}")

    # Curated, ordered results/ folder, with empty directories pruned.
    try:
        import results_report
        results_report.write_results(OUTPUTS)
    except Exception as e:
        print(f"  results_report skipped: {type(e).__name__}: {e}")

    print("\nPipeline complete.")
