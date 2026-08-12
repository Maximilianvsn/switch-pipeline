"""Unit tests for lineage-keyed state pairing and backbone-balanced nulls."""
import unittest

import pandas as pd

import paired_nulls
import state_pairing


def test_state_outputs_are_joined_by_lineage_key_not_row_order():
    state1 = pd.DataFrame({
        "s1_rfd3_holo_description": ["bb_1", "bb_2"],
        "s1_rfd3_holo_location": ["one.pdb", "two.pdb"],
    })
    state2 = pd.DataFrame({
        "s2_rfd3_apo_description": [
            "bb_2_binder_pcna_0_model_0", "bb_1_binder_pcna_0_model_0"
        ],
        "s2_rfd3_apo_location": ["two_state2.pdb", "one_state2.pdb"],
    })
    paired = state_pairing.pair_state_outputs(state1, state2)
    assert paired["state2_pdb"].tolist() == ["one_state2.pdb", "two_state2.pdb"]


def test_duplicate_state_output_is_a_hard_lineage_error():
    state1 = pd.DataFrame({"s1_rfd3_holo_description": ["bb"]})
    state2 = pd.DataFrame({
        "s2_rfd3_apo_description": ["bb_binder_pcna_0", "bb_binder_pcna_1"],
        "s2_rfd3_apo_location": ["a.pdb", "b.pdb"],
    })
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "expected 1"):
        state_pairing.pair_state_outputs(state1, state2)


def test_scramble_null_matches_real_search_multiplicity_per_backbone():
    frame = pd.DataFrame({
        "poses_description": ["a1", "a2", "b1", "b2"],
        "backbone": ["a", "a", "b", "b"],
        "sequence": ["ACDEFG", "ACDFEG", "HIKLMN", "HILKMN"],
    })
    null = paired_nulls.balanced_scrambles(frame, "sequence", "backbone", n=10)
    assert len(null) == len(frame)
    assert null.groupby("_null_backbone").size().to_dict() == {"a": 2, "b": 2}
    assert set(null["_real_design_id"]) == set(frame["poses_description"])
    assert all(sorted(a) == sorted(b) for a, b in zip(null["_scr_seq"], null["sequence"]))
    assert all(a != b for a, b in zip(null["_scr_seq"], null["sequence"]))


def test_stop_go_requires_every_metric_to_discriminate():
    table = pd.DataFrame({"auc": [0.9, 0.8, 0.75, 0.69], "n_pairs": [20] * 4, "paired_win_rate": [0.8] * 4, "paired_win_rate_ci_low": [0.6] * 4})
    assert not paired_nulls.passes_stop_go(table, min_auc=0.70, min_pairs=20)
    table.loc[3, "auc"] = 0.70
    assert paired_nulls.passes_stop_go(table, min_auc=0.70, min_pairs=20)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_state_outputs_are_joined_by_lineage_key_not_row_order,
            test_duplicate_state_output_is_a_hard_lineage_error,
            test_scramble_null_matches_real_search_multiplicity_per_backbone,
            test_stop_go_requires_every_metric_to_discriminate,
        )
    )
