"""Unit tests for the evaluation and reporting stage."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import protein_only_evaluation


class ProteinOnlyEvaluationTests(unittest.TestCase):
    def _write_provenance(self, directory: Path, mode: str = "smoke"):
        (directory / "run_provenance.json").write_text(json.dumps({
            "mode": mode,
            "config": {
                "evaluation": {"min_null_auc": 0.70, "min_null_pairs": 20},
                "smoke": {"min_null_auc": 0.0, "min_null_pairs": 1},
            },
        }))

    def test_smoke_overrides_do_not_lower_thesis_evidence_requirements(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            self._write_provenance(directory, mode="smoke")
            settings = protein_only_evaluation._run_settings(str(directory))
            self.assertEqual(settings["mode"], "smoke")
            self.assertEqual(settings["min_null_auc"], 0.70)
            self.assertEqual(settings["min_null_pairs"], 20)

    def test_partial_metric_table_cannot_pass_stop_go(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            plots = directory / "plots"
            plots.mkdir()
            self._write_provenance(directory, mode="production")
            pd.DataFrame([{
                "metric": "af2_holo_plddt", "auc": 1.0,
                "paired_win_rate": 1.0, "paired_win_rate_ci_low": 0.9,
                "paired_win_rate_ci_high": 1.0, "n_pairs": 20,
            }]).to_csv(directory / "af2_null_separation.csv", index=False)

            report = protein_only_evaluation._null_report(str(directory), str(plots))
            self.assertFalse(report["all_metrics_pass"])
            self.assertEqual(len(report["missing_metrics"]), 3)

    def test_exact_complete_metric_table_can_pass_stop_go(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            plots = directory / "plots"
            plots.mkdir()
            self._write_provenance(directory, mode="production")
            pd.DataFrame([{
                "metric": metric, "auc": 1.0,
                "paired_win_rate": 1.0, "paired_win_rate_ci_low": 0.9,
                "paired_win_rate_ci_high": 1.0, "n_pairs": 20,
            } for metric in sorted(protein_only_evaluation.EXPECTED_AF2_NULL_METRICS)]).to_csv(
                directory / "af2_null_separation.csv", index=False
            )

            report = protein_only_evaluation._null_report(str(directory), str(plots))
            self.assertTrue(report["all_metrics_pass"])
            self.assertEqual(report["missing_metrics"], [])
            self.assertEqual(report["unexpected_metrics"], [])


if __name__ == "__main__":
    unittest.main()
