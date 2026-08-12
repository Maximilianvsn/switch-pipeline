"""Config loading, validation, CLI parsing, and target-PDB construction.

`validate_config` checks a run's settings before any GPU time is spent, and is
kept as a single unit rather than distributed across the stages it guards.

`make_holo_pdb` belongs here rather than in `structure_io` because it performs
config-driven target assembly (hotspots, contigs, chain naming) rather than
generic structure I/O.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import yaml
from glob import glob
from pathlib import Path


def _parse_hotspot_str(hotspots: str) -> list[tuple[str, int]]:
    """Parse "A56,A58,A113,A115" -> [("A", 56), ("A", 58), ...]."""
    out = []
    for tok in hotspots.split(","):
        tok = tok.strip()
        if not tok:
            continue
        chain, resnum = tok[0], int(tok[1:])
        out.append((chain, resnum))
    return out


def _contig_target_start(contig: str) -> int | None:
    """Parse the target's starting residue number from an RFD3 contig
    string, e.g. "A19-115,/0,80" -> 19. Boltz numbers its output chains
    sequentially from 1 to match the raw sequence it was given (verified
    empirically against real output CIFs), so
    boltz_resnum = original_resnum - start + 1 remaps a hotspot residue
    number (in the original target PDB's numbering) into Boltz's own
    output numbering — needed to locate hotspot residues in a predicted
    structure rather than the input PDB.
    """
    m = re.match(r"^[A-Za-z](\d+)-\d+", contig.split(",")[0].strip())
    return int(m.group(1)) if m else None


def resolve_path(path: str, ws: str) -> str:
    """Resolve a path relative to the workspace root, or leave absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(ws, path)


def load_config(config_path: str, ws: str) -> dict:
    """Load YAML config and resolve relative paths against the workspace."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a YAML mapping")

    for target in ("holo_target", "apo_target"):
        section = cfg.get(target)
        if isinstance(section, dict) and section.get("pdb"):
            section["pdb"] = resolve_path(section["pdb"], ws)


    if cfg.get("dynamicmpnn", {}).get("checkpoint"):
        cfg["dynamicmpnn"]["checkpoint"] = resolve_path(
            cfg["dynamicmpnn"]["checkpoint"], ws
        )

    for _dt in cfg.get("specificity", {}).get("decoy_targets", []) or []:
        if _dt.get("pdb"):
            _dt["pdb"] = resolve_path(_dt["pdb"], ws)

    # Strip ALL whitespace from target sequences. YAML folded scalars (">-")
    # join wrapped lines with a space, so a sequence written across several
    # indented lines silently carries embedded spaces (e.g. PD-L1 arrived as a
    # 98-char sequence with a space mid-chain instead of the true 97 residues).
    # Boltz tolerated it but was folding/searching a corrupted target; it also
    # breaks precomputed-MSA reuse (the MSA must match the exact sequence).
    for tgt in ("holo_target", "apo_target"):
        if cfg.get(tgt, {}).get("sequence"):
            cfg[tgt]["sequence"] = "".join(str(cfg[tgt]["sequence"]).split())

    return cfg


def validate_config(cfg: dict) -> None:
    """Fail fast on config/PDB inconsistencies that would otherwise surface as
    confusing errors deep in a multi-hour run — or, worse, silently produce
    wrong designs (e.g. a sequence/contig length mismatch that misaligns every
    hotspot remap). Hard errors raise ValueError; recoverable issues print a
    CONFIG WARNING. Call once, right after load_config().
    """
    print("\n[Config check]")
    errors, warns = [], []

    for key in ("holo_target", "apo_target"):
        if not isinstance(cfg.get(key), dict):
            errors.append(f"missing or invalid required mapping '{key}'")
    if errors:
        raise ValueError("Config invalid:\n  - " + "\n  - ".join(errors))

    holo, apo = cfg["holo_target"], cfg["apo_target"]

    def _pdb_residues(pdb_path: str) -> set:
        """(chain, resnum) present in a PDB — the numbering RFD3 hotspots use."""
        out = set()
        try:
            with open(pdb_path) as f:
                for line in f:
                    if line.startswith("ATOM"):
                        try:
                            out.add((line[21], int(line[22:26])))
                        except ValueError:
                            pass
        except Exception:
            pass
        return out

    # 1. required input files exist (paths already resolved by load_config)
    for label, path in [("holo_target.pdb", holo.get("pdb")), ("apo_target.pdb", apo.get("pdb"))]:
        if not path or not os.path.isfile(path):
            errors.append(f"{label} not found: {path}")

    # 2. holo contig target span must match the provided sequence length —
    #    Boltz numbers the target chain from 1, and hotspots are remapped by
    #    (resnum - contig_start + 1); a length mismatch silently misaligns them.
    m = re.match(r"^([A-Za-z])(\d+)-(\d+)", str(holo.get("contig", "")).split(",")[0].strip())
    seq = re.sub(r"\s+", "", holo.get("sequence", "") or "")
    if m:
        hchain, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        span = hi - lo + 1
        if seq and len(seq) != span:
            errors.append(f"holo sequence length ({len(seq)}) != holo contig target span "
                          f"{hchain}{lo}-{hi} ({span} res) — hotspot remap / interface metrics would misalign")
    else:
        warns.append(f"could not parse a target segment (e.g. 'A19-115') from holo contig "
                     f"'{holo.get('contig','')}'")

    # 3. hotspots must be residues that actually EXIST in the target PDB (this
    #    is the numbering RFD3's select_hotspots uses) — catches typos / wrong
    #    numbering that would otherwise place hotspots on nothing.
    for tgt_label, tgt in [("holo", holo), ("apo", apo)]:
        pdb = tgt.get("pdb")
        if not pdb or not os.path.isfile(pdb):
            continue
        present = _pdb_residues(pdb)
        if not present:
            continue
        for hc, hr in _parse_hotspot_str(tgt.get("hotspots", "")):
            if (hc, hr) not in present:
                warns.append(f"{tgt_label} hotspot {hc}{hr} not found in {os.path.basename(pdb)} "
                             f"— check the chain/residue numbering")

    # 4. other required fields
    if not holo.get("hotspots"):
        warns.append("holo_target.hotspots is empty — RFD3 has no interface to aim at")
    if not apo.get("hotspots"):
        warns.append("apo_target.hotspots is empty — the apo interface is unconstrained")
    if not holo.get("binder_chain"):
        warns.append("holo_target.binder_chain not set (pipeline defaults to 'B')")

    # 5. optional tool checkpoints
    dmp = cfg.get("dynamicmpnn", {}).get("checkpoint")
    if dmp and not os.path.isfile(dmp):
        warns.append(f"dynamicmpnn.checkpoint not found: {dmp}")

    # 6. Decoy targets (specificity control)
    if cfg.get("specificity", {}).get("enabled"):
        decoys = cfg.get("specificity", {}).get("decoy_targets", []) or []
        if not decoys:
            warns.append("specificity.enabled is true but decoy_targets is empty — Step 8 will skip")
        for _dt in decoys:
            if not _dt.get("name"):
                errors.append("a specificity.decoy_targets entry is missing 'name'")
            if not _dt.get("sequence"):
                errors.append(f"decoy target '{_dt.get('name', '?')}' is missing 'sequence'")
            if _dt.get("pdb") and not os.path.isfile(_dt["pdb"]):
                warns.append(f"decoy target '{_dt.get('name', '?')}' pdb not found: {_dt.get('pdb')} "
                             f"(not fatal — only 'sequence' is used for scoring)")

    evaluation_cfg = cfg.get("evaluation", {})
    sampling_cfg = cfg.get("sampling", {})
    af2_cfg = cfg.get("af2", {})
    if not af2_cfg.get("enabled", False):
        errors.append(
            "ranking requires af2.enabled=true — the AF2 initial-guess gate is the "
            "only predictor shown to separate real designs from scrambled nulls, "
            "and the final tiers are derived from it"
        )
    s2_desig_cfg = af2_cfg.get("state2_designability", {})
    for block_name, block in (
        ("sampling", sampling_cfg),
        ("smoke", cfg.get("smoke", {})),
        ("geometry_calibration", cfg.get("geometry_calibration", {})),
    ):
        try:
            apo_batch = int(block.get("apo_batch", 1))
        except (TypeError, ValueError):
            errors.append(f"{block_name}.apo_batch must be an integer")
            continue
        if apo_batch < 1:
            errors.append(f"{block_name}.apo_batch must be at least 1")
        if apo_batch > 1 and not (
            af2_cfg.get("enabled", False) and s2_desig_cfg.get("enabled", False)
        ):
            errors.append(
                f"{block_name}.apo_batch={apo_batch} requires "
                "af2.state2_designability.enabled=true"
            )

    calibration_values = cfg.get("geometry_calibration", {}).get("partial_t_values", [])
    if calibration_values:
        try:
            calibration_values = [float(value) for value in calibration_values]
            if len(calibration_values) < 2 or len(calibration_values) != len(set(calibration_values)):
                errors.append("geometry_calibration.partial_t_values must contain at least two unique values")
            if any(value <= 0.0 or value > 15.0 for value in calibration_values):
                errors.append("geometry_calibration.partial_t_values must be in (0, 15] Angstroms")
        except (TypeError, ValueError):
            errors.append("geometry_calibration.partial_t_values must be numeric")

    for label, design_cfg in (
        ("af2.designability", af2_cfg.get("designability", {})),
        ("af2.state2_designability", s2_desig_cfg),
    ):
        if not design_cfg.get("enabled", False):
            continue
        try:
            n_seqs = int(design_cfg.get("n_seqs", 1))
            top_k = int(design_cfg.get("pre_af2_top_k", n_seqs))
        except (TypeError, ValueError):
            errors.append(f"{label}.n_seqs and pre_af2_top_k must be integers")
            continue
        if n_seqs < 1 or top_k < 1 or top_k > n_seqs:
            errors.append(
                f"{label} requires 1 <= pre_af2_top_k <= n_seqs; "
                f"got top_k={top_k}, n_seqs={n_seqs}"
            )

    if evaluation_cfg.get("enforce_equal_method_budget", False) and evaluation_cfg.get("score_msd", False):
        dmpnn_n = int(sampling_cfg.get("dmpnn_nseq", 0))
        msd_n = int(sampling_cfg.get("mpnn_msd_nseq", 0))
        if dmpnn_n != msd_n:
            errors.append(f"unfair method comparison: dmpnn_nseq={dmpnn_n} != mpnn_msd_nseq={msd_n}")
        if int(sampling_cfg.get("adaptive_target", 0)) > 0:
            errors.append("unfair method comparison: adaptive DynamicMPNN top-up must be disabled")

    partial_t = apo.get("partial_t", 2.0)
    try:
        partial_t = float(partial_t)
        if not (0.0 < partial_t <= 15.0):
            errors.append("apo_target.partial_t must be in (0, 15] Angstroms")
    except (TypeError, ValueError):
        errors.append("apo_target.partial_t must be numeric Angstroms")

    for w in warns:
        print(f"  CONFIG WARNING: {w}")
    if errors:
        raise ValueError("Config invalid — fix these before running:\n  - " + "\n  - ".join(errors))
    print(f"  Config check passed ({len(warns)} warning(s)).")


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-state de novo protein binder design pipeline"
    )
    p.add_argument(
        "--config", required=True,
        help="Path to YAML target config (e.g. configs/pdl1_pcna_protein_only.yaml)",
    )
    p.add_argument(
        "--run-name", default=None,
        help="Unique name for this run. Outputs go to outputs/<run-name>/. "
             "Re-using a name resumes from existing checkpoints.",
    )
    p.add_argument("--smoke", action="store_true", help="Smoke test with minimal numbers")
    p.add_argument(
        "--geometry-calibration", action="store_true",
        help="Run Steps 1-2 for a matched partial_t sweep, write comparison artifacts, and exit",
    )
    p.add_argument(
        "--partial-t-values", nargs="+", type=float, default=None,
        help="Coordinate-noise values in Angstroms for --geometry-calibration",
    )
    return p.parse_args()


# Pipeline

