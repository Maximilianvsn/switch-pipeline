"""Unit tests for the AF2 runner, gating and evaluation helpers."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import af2_runner
import paired_nulls
import partial_t_calibration
import state_pairing
from switch_geometry import aligned_binder_rmsd, paired_state_geometry, sha256_file


def _write_pdb(path, chains):
    serial, lines = 1, []
    for chain_id, coords in chains:
        for residue_number, (x, y, z) in enumerate(coords, start=1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id}{residue_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
            )
            serial += 1
        lines.append("TER\n")
    path.write_text("".join(lines) + "END\n")


class CorePipelineTests(unittest.TestCase):
    def test_partial_t_recommendation_uses_yield_then_target_rmsd(self):
        summary = pd.DataFrame({
            "partial_t_angstrom": [2.0, 5.0, 10.0, 15.0],
            "n_geometry_pass": [4, 8, 8, 2],
            "n_backbones_with_geometry_pass": [2, 5, 5, 1],
            "n_state1_backbones": [6, 6, 6, 6],
            "n_state2_pairs": [12, 12, 12, 12],
            "geometry_pass_rate": [4 / 12, 8 / 12, 8 / 12, 2 / 12],
            "geometry_pass_rate_ci_low": [0.14, 0.39, 0.40, 0.05],
            "geometry_pass_rate_ci_high": [0.61, 0.86, 0.86, 0.45],
            "backbone_success_rate": [2 / 6, 5 / 6, 5 / 6, 1 / 6],
            "backbone_success_rate_ci_low": [0.10, 0.44, 0.45, 0.03],
            "backbone_success_rate_ci_high": [0.70, 0.97, 0.97, 0.56],
            "median_rmsd_passing_angstrom": [1.2, 2.8, 5.5, 7.0],
        })
        result = partial_t_calibration.choose_partial_t(summary, target_rmsd=3.0)
        self.assertEqual(result["recommended_partial_t_angstrom"], 5.0)

    def test_keyed_pairing_ignores_row_order(self):
        state1 = pd.DataFrame({
            "s1_rfd3_holo_description": ["bb_1", "bb_2"],
            "s1_rfd3_holo_location": ["one", "two"],
        })
        state2 = pd.DataFrame({
            "s2_rfd3_apo_description": ["bb_2_binder_pcna_0", "bb_1_binder_pcna_0"],
            "s2_rfd3_apo_location": ["two_state2", "one_state2"],
        })
        paired = state_pairing.pair_state_outputs(state1, state2)
        self.assertEqual(paired["state2_pdb"].tolist(), ["one_state2", "two_state2"])

    def test_one_to_many_state2_lineage_is_explicit_and_stable(self):
        state1 = pd.DataFrame({
            "s1_rfd3_holo_description": ["bb"],
            "s1_rfd3_holo_location": ["one"],
        })
        state2 = pd.DataFrame({
            "s2_rfd3_apo_description": [
                "bb_binder_pcna_0_model_1",
                "bb_binder_pcna_0_model_0",
            ],
            "s2_rfd3_apo_location": ["variant_1", "variant_0"],
        })
        paired = state_pairing.pair_state_outputs(
            state1, state2, expected_variants=2
        )
        self.assertEqual(
            paired["state_pair_id"].tolist(), ["bb__s2v000", "bb__s2v001"]
        )
        self.assertEqual(paired["state2_pdb"].tolist(), ["variant_0", "variant_1"])

    def test_state2_ranking_prioritizes_designability_then_rmsd(self):
        candidates = pd.DataFrame({
            "s1_rfd3_holo_description": ["a", "a", "b", "b"],
            "state_pair_id": ["a0", "a1", "b0", "b1"],
            "state2_designability_score": [0.80, 0.90, 0.85, 0.85],
            "binder_ca_rmsd": [3.0, 7.0, 1.5, 2.8],
        })
        ranked = state_pairing.select_best_state2_candidate(
            candidates, target_rmsd=3.0
        )
        selected = set(ranked.loc[ranked["state2_selected"], "state_pair_id"])
        self.assertEqual(selected, {"a1", "b1"})

    def test_null_is_backbone_balanced(self):
        frame = pd.DataFrame({
            "poses_description": ["a1", "a2", "b1", "b2"],
            "backbone": ["a", "a", "b", "b"],
            "sequence": ["ACDEFG", "ACDFEG", "HIKLMN", "HILKMN"],
        })
        null = paired_nulls.balanced_scrambles(frame, "sequence", "backbone", n=10)
        self.assertEqual(len(null), len(frame))
        self.assertEqual(null.groupby("_null_backbone").size().to_dict(), {"a": 2, "b": 2})
        self.assertEqual(set(null["_real_design_id"]), set(frame["poses_description"]))
        multi_chain = frame.iloc[[0]].assign(sequence="TARGET:BINDER")
        with self.assertRaisesRegex(ValueError, "binder-only"):
            paired_nulls.balanced_scrambles(multi_chain, "sequence", "backbone")

    def test_af2_nested_job_name_includes_run_and_stage(self):
        name = af2_runner._slurm_job_name(
            "/work/outputs/smoke_proteins_fullpath_20260717_v1/s5_5_af2_gate/af2"
        )
        self.assertEqual(
            name, "smoke_proteins_fullpath_20260717_v1__s5_5_af2_gate__af2"
        )

    def test_af2_request_selects_sequence_by_explicit_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            backbone = Path(directory) / "two_chain.pdb"
            _write_pdb(backbone, [
                ("A", [(i * 3, 0, 0) for i in range(4)]),
                ("B", [(i * 3, 4, 0) for i in range(5)]),
            ])
            frame = pd.DataFrame({
                "design_id": ["d1"], "backbone": [str(backbone)],
                "sequence": ["AAAA:BBBBB"],
            })
            state2 = af2_runner.build_state_requests(
                frame, "design_id", "backbone", "B", "A", "sequence",
                str(Path(directory) / "state2"), "s2__",
            )
            state1 = af2_runner.build_state_requests(
                frame, "design_id", "backbone", "A", "B", "sequence",
                str(Path(directory) / "state1"), "s1__",
            )
            self.assertEqual(state2.loc[0, "seq"], "AAAA")
            self.assertEqual(state1.loc[0, "seq"], "BBBBB")

    def test_staging_manifest_rehashes_every_recorded_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("source", "binder", "complex", "spec"):
                path = root / f"{name}.dat"
                path.write_text(name)
                paths[name] = path
            manifest = [{
                "source_id": "bb",
                "source_structure": str(paths["source"]),
                "source_sha256": sha256_file(paths["source"]),
                "staged_binder": str(paths["binder"]),
                "staged_binder_sha256": sha256_file(paths["binder"]),
                "staged_complex": str(paths["complex"]),
                "staged_complex_sha256": sha256_file(paths["complex"]),
                "spec_json": str(paths["spec"]),
                "spec_sha256": sha256_file(paths["spec"]),
            }]
            manifest_path = root / "lineage_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(
                state_pairing.validate_staging_manifest(str(manifest_path), ["bb"]),
                manifest,
            )
            paths["complex"].write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                state_pairing.validate_staging_manifest(str(manifest_path), ["bb"])

    def test_generation_sanity_rejects_fixed_or_duplicate_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            state2 = Path(directory) / "state2.pdb"
            _write_pdb(state2, [
                ("A", [(0, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0)]),
                ("B", [(0, 4, 0), (3, 4, 0), (6, 4, 0), (9, 4, 0)]),
            ])
            fixed = pd.DataFrame({
                "s1_rfd3_holo_description": ["bb", "bb"],
                "state2_pdb": [str(state2), str(state2)],
                "binder_ca_rmsd": [0.04, 0.05],
            })
            with self.assertRaisesRegex(RuntimeError, "did not move"):
                state_pairing.validate_generation_sanity(fixed)
            duplicates = fixed.assign(binder_ca_rmsd=[2.0, 2.0])
            with self.assertRaisesRegex(RuntimeError, "coordinate-duplicate"):
                state_pairing.validate_generation_sanity(duplicates)

    def test_generation_sanity_is_per_backbone_and_detects_partial_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pdb"
            second = Path(directory) / "second.pdb"
            _write_pdb(first, [
                ("A", [(0, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0)]),
                ("B", [(0, 4, 0), (3, 4, 0), (6, 4, 0), (9, 4, 0)]),
            ])
            _write_pdb(second, [
                ("A", [(0, 0, 0), (3, 0, 1), (6, 0, 0), (9, 0, 0)]),
                ("B", [(0, 4, 0), (3, 4, 1), (6, 4, 0), (9, 4, 0)]),
            ])
            mixed = pd.DataFrame({
                "s1_rfd3_holo_description": ["fixed", "fixed", "moving", "moving"],
                "state2_pdb": [str(first), str(second), str(first), str(second)],
                "binder_ca_rmsd": [0.04, 0.05, 2.0, 2.1],
            })
            with self.assertRaisesRegex(RuntimeError, "for fixed"):
                state_pairing.validate_generation_sanity(mixed)

            partial_duplicates = pd.DataFrame({
                "s1_rfd3_holo_description": ["bb", "bb", "bb"],
                "state2_pdb": [str(first), str(first), str(second)],
                "binder_ca_rmsd": [2.0, 2.1, 2.2],
            })
            with self.assertRaisesRegex(RuntimeError, "coordinate-duplicate"):
                state_pairing.validate_generation_sanity(partial_duplicates)

    def test_geometry_handles_different_binder_chain_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.pdb", Path(directory) / "b.pdb"
            _write_pdb(first, [
                ("A", [(0, 3, 0), (3, 3, 0), (6, 3, 0), (9, 3, 0)]),
                ("B", [(0, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0)]),
            ])
            _write_pdb(second, [
                ("A", [(10, 5, 0), (13, 5, 1.5), (16, 5, 0), (19, 5, 0)]),
                ("B", [(10, 8, 0), (13, 8, 1.5), (16, 8, 0), (19, 8, 0)]),
            ])
            rmsd = aligned_binder_rmsd(str(first), str(second), "B", "A")
            self.assertGreater(rmsd, 0.4)
            metrics = paired_state_geometry(str(first), str(second), "B", "A", interface_cutoff=4.0)
            self.assertEqual(metrics["interface_jaccard"], 1.0)
            self.assertGreater(metrics["target_target_clash_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
