"""Keyed pairing and pre-sequence geometry gates for two protein states."""
from __future__ import annotations

import json
import os

import pandas as pd

from switch_geometry import aligned_binder_rmsd, paired_state_geometry, sha256_file


def _matching_state2_rows(source_id: str, state2_df: pd.DataFrame, description_col: str):
    prefix = f"{source_id}_binder_pcna"
    descriptions = state2_df[description_col].astype(str)
    return state2_df[descriptions.str.startswith(prefix)]


def pair_state_outputs(
    state1_df: pd.DataFrame,
    state2_df: pd.DataFrame,
    state1_id_col: str = "s1_rfd3_holo_description",
    state2_description_col: str = "s2_rfd3_apo_description",
    state2_location_col: str = "s2_rfd3_apo_location",
    expected_variants: int = 1,
) -> pd.DataFrame:
    """Expand state 1 to an explicitly keyed set of state-2 candidates.

    Pairing is always performed by the state-1 source identifier, never row
    order. Every candidate receives a stable pair id before geometry or
    designability filtering, so one-to-many sampling remains auditable.
    """
    if expected_variants < 1:
        raise ValueError("expected_variants must be at least 1")
    if state1_id_col not in state1_df:
        raise KeyError(f"Missing state-1 lineage column {state1_id_col}")
    if state1_df[state1_id_col].astype(str).duplicated().any():
        raise RuntimeError("State-1 lineage identifiers must be unique before state-2 expansion")
    for column in (state2_description_col, state2_location_col):
        if column not in state2_df:
            raise KeyError(f"Missing state-2 output column {column}")
    if state2_df[state2_description_col].astype(str).duplicated().any():
        raise RuntimeError("State-2 output descriptions are not unique")
    if state2_df[state2_location_col].astype(str).duplicated().any():
        raise RuntimeError("State-2 output paths are not unique")

    rows = []
    errors = []
    for source_id in state1_df[state1_id_col].astype(str):
        matches = _matching_state2_rows(source_id, state2_df, state2_description_col)
        matches = matches.sort_values(state2_description_col).reset_index(drop=True)
        if len(matches) != expected_variants:
            errors.append(
                f"{source_id}: expected {expected_variants} state-2 output(s), found {len(matches)}"
            )
            continue
        for variant_index, match in matches.iterrows():
            rows.append({
                state1_id_col: source_id,
                "state2_variant_index": int(variant_index),
                "state_pair_id": f"{source_id}__s2v{variant_index:03d}",
                "state2_pdb": match[state2_location_col],
                "state2_description": match[state2_description_col],
            })
    if errors:
        raise RuntimeError("State lineage pairing failed:\n  - " + "\n  - ".join(errors[:20]))
    paired = state1_df.merge(
        pd.DataFrame(rows), on=state1_id_col, how="left", validate="one_to_many"
    )
    if paired["state2_pdb"].isna().any() or paired["state_pair_id"].duplicated().any():
        raise RuntimeError("State lineage expansion produced missing or duplicate pair identifiers")
    return paired


def select_best_state2_candidate(
    candidates: pd.DataFrame,
    state1_id_col: str = "s1_rfd3_holo_description",
    score_col: str = "state2_designability_score",
    rmsd_col: str = "binder_ca_rmsd",
    target_rmsd: float = 3.0,
) -> pd.DataFrame:
    """Select one independently designable state-2 candidate per state 1.

    AF2 designability is the primary key. RMSD is only a deterministic
    tie-break toward a predeclared, non-extreme conformational change; geometry
    eligibility must be applied before this function.
    """
    required = (state1_id_col, "state_pair_id", score_col, rmsd_col)
    missing = [column for column in required if column not in candidates]
    if missing:
        raise KeyError(f"Missing state-2 ranking columns: {missing}")
    ranked = candidates.copy()
    ranked["_state2_rmsd_target_delta"] = (
        pd.to_numeric(ranked[rmsd_col], errors="coerce") - float(target_rmsd)
    ).abs()
    ranked = ranked.sort_values(
        [state1_id_col, score_col, "_state2_rmsd_target_delta", "state_pair_id"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    ranked["state2_candidate_rank"] = ranked.groupby(state1_id_col).cumcount() + 1
    ranked["state2_selected"] = ranked["state2_candidate_rank"].eq(1)
    return ranked


def validate_staging_manifest(manifest_path: str, expected_source_ids) -> list[dict]:
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"Missing state lineage manifest: {manifest_path}")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    observed = [str(row["source_id"]) for row in manifest]
    expected = [str(value) for value in expected_source_ids]
    if len(observed) != len(set(observed)):
        raise RuntimeError("State lineage manifest contains duplicate source ids")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise RuntimeError(f"State lineage manifest mismatch; missing={missing[:10]} extra={extra[:10]}")
    integrity_fields = (
        ("source_structure", "source_sha256"),
        ("staged_binder", "staged_binder_sha256"),
        ("staged_complex", "staged_complex_sha256"),
        ("spec_json", "spec_sha256"),
    )
    integrity_errors = []
    for row in manifest:
        source_id = str(row.get("source_id", "<unknown>"))
        for path_field, digest_field in integrity_fields:
            path = row.get(path_field)
            expected_digest = row.get(digest_field)
            if not path or not expected_digest:
                integrity_errors.append(
                    f"{source_id}: missing {path_field}/{digest_field} integrity metadata"
                )
                continue
            if not os.path.isfile(path):
                integrity_errors.append(f"{source_id}: missing staged file {path}")
                continue
            observed_digest = sha256_file(path)
            if observed_digest != expected_digest:
                integrity_errors.append(
                    f"{source_id}: SHA-256 mismatch for {path_field} ({path})"
                )
    if integrity_errors:
        raise RuntimeError(
            "State lineage manifest integrity validation failed:\n  - "
            + "\n  - ".join(integrity_errors[:20])
        )
    return manifest


def add_geometry_metrics(
    paired_df: pd.DataFrame,
    state1_location_col: str,
    state1_binder_chain: str,
    state2_binder_chain: str,
    interface_cutoff: float = 5.0,
    clash_cutoff: float = 2.5,
) -> pd.DataFrame:
    records = []
    for _, row in paired_df.iterrows():
        metrics = paired_state_geometry(
            row[state1_location_col],
            row["state2_pdb"],
            state1_binder_chain,
            state2_binder_chain,
            interface_cutoff=interface_cutoff,
            clash_cutoff=clash_cutoff,
        )
        records.append(metrics)
    return pd.concat([paired_df.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def validate_generation_sanity(
    paired_df: pd.DataFrame,
    state1_id_col: str = "s1_rfd3_holo_description",
    state2_binder_chain: str = "A",
    minimum_state_change: float = 0.25,
    duplicate_tolerance: float = 1e-3,
) -> None:
    """Reject fixed-coordinate or duplicated state-2 generations.

    Smoke thresholds may be relaxed for integration coverage, but they must
    never permit a completely fixed binder or a fake multi-candidate search.
    `binder_ca_rmsd` is the already target-aligned state-1/state-2 change.
    Candidate-to-candidate RMSD is computed after binder alignment, so exact
    coordinate copies are detected independently of rigid placement.
    """
    required = {state1_id_col, "state2_pdb", "binder_ca_rmsd"}
    missing = required - set(paired_df.columns)
    if missing:
        raise KeyError(f"Missing generation-sanity columns: {sorted(missing)}")

    for source_id, group in paired_df.groupby(state1_id_col, sort=False):
        rmsd = pd.to_numeric(group["binder_ca_rmsd"], errors="coerce")
        if not rmsd.notna().any() or float(rmsd.max()) < float(minimum_state_change):
            observed = float(rmsd.max()) if rmsd.notna().any() else float("nan")
            raise RuntimeError(
                f"State-2 generation did not move the binder for {source_id}: "
                f"maximum state-1/state-2 C-alpha RMSD is {observed:.4f} A, "
                f"below the hard sanity floor {minimum_state_change:.4f} A. "
                "Check RFD3 select_fixed_atoms/partial_t."
            )

        paths = group["state2_pdb"].astype(str).tolist()
        if len(paths) < 2:
            continue
        for left_index, left_path in enumerate(paths):
            for right_path in paths[left_index + 1:]:
                difference = aligned_binder_rmsd(
                    left_path, right_path, state2_binder_chain, state2_binder_chain
                )
                if difference <= duplicate_tolerance:
                    raise RuntimeError(
                        f"State-2 candidate search for {source_id} produced "
                        "coordinate-duplicate binder backbones "
                        f"({os.path.basename(left_path)} vs "
                        f"{os.path.basename(right_path)}: C-alpha RMSD "
                        f"{difference:.6f} A <= {duplicate_tolerance} A)."
                    )


def geometry_pass_mask(df: pd.DataFrame, config: dict) -> pd.Series:
    minimum_rmsd = float(config.get("min_binder_ca_rmsd", 1.5))
    maximum_rmsd = float(config.get("max_binder_ca_rmsd", 8.0))
    minimum_jaccard = float(config.get("min_interface_jaccard", 0.25))
    minimum_reuse = float(config.get("min_interface_reuse_fraction", 0.60))
    minimum_clashes = int(config.get("min_target_target_clash_pairs", 5))
    minimum_contacts = int(config.get("min_interface_residues_per_state", 5))
    return (
        df["binder_ca_rmsd"].between(minimum_rmsd, maximum_rmsd, inclusive="both")
        & (df["interface_jaccard"] >= minimum_jaccard)
        & (df["interface_reuse_fraction"] >= minimum_reuse)
        & (df["target_target_clash_pairs"] >= minimum_clashes)
        & (df["state1_interface_n"] >= minimum_contacts)
        & (df["state2_interface_n"] >= minimum_contacts)
    ).fillna(False)
