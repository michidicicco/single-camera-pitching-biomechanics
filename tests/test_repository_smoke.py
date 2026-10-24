"""Nonparticipant smoke tests for thesis-facing public computations.

Run with: python -m unittest discover -s tests -v
These tests do not claim to reproduce restricted participant-level results.
"""
import importlib
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
utils = importlib.import_module("analysis_utils")
profiles = importlib.import_module("mechanical_profiles")
adjusted = importlib.import_module("pitcher_adjusted_analysis")
validator = importlib.import_module("validate_events")
merger = importlib.import_module("merge_analysis_outputs")


class RepositorySmokeTests(unittest.TestCase):
    def test_32_original_features_and_cluster_recovery(self):
        self.assertEqual(len(utils.PRIMARY_MODEL_FEATURES), 32)
        rng = np.random.default_rng(14)
        X = np.vstack([
            rng.normal(loc=-4, scale=0.2, size=(15, 3)),
            rng.normal(loc=4, scale=0.2, size=(15, 3)),
        ])
        labels, _, _, _ = utils.kmeans_numpy(X, k=2, seed=13, n_init=10)
        self.assertEqual(len(set(labels)), 2)
        self.assertGreater(utils.silhouette_score_numpy(X, labels), 0.8)

    def test_archived_and_clean_higher_labels_agree(self):
        inputs = pd.DataFrame({
            "pitcher_id": ["P1"] * 4 + ["P2"] * 4,
            "relative_mechanical_profile": [
                "lower", "moderate", "higher", "higher",
                "lower", "moderate", "higher", "higher",
            ],
        })
        clean = inputs["relative_mechanical_profile"].astype("string").str.strip().str.lower()
        archived = clean.replace({"higher": "higher_watch_relative_mechanical_profile"})
        # Exercise equivalence required by the published adjusted-analysis mapping.
        for labels in (clean, archived):
            indicator = labels.isin(
                {"higher", "higher_watch_relative_mechanical_profile"}
            ).astype(float)
            self.assertEqual(indicator.sum(), 4)
            z = adjusted.within_pitcher_standardize(indicator, inputs["pitcher_id"])
            self.assertTrue(np.isfinite(z).all())

    def test_profile_qc_independence(self):
        base = pd.DataFrame({
            "pitcher_id": ["P1", "P1"],
            "pitch_id": [1, 2],
            "relative_mechanical_band_pooled": ["higher", "lower"],
            "relative_mechanical_band_within_pitcher": ["higher", "lower"],
            "mechanical_flag_count": [1, 1],
            "measurement_qc_status": ["pass", "watch"],
        })
        a = profiles.classify_dataframe(base, 2.0, 4.0)
        swapped = base.copy()
        swapped["measurement_qc_status"] = ["watch", "pass"]
        b = profiles.classify_dataframe(swapped, 2.0, 4.0)
        self.assertEqual(a["relative_mechanical_profile"].tolist(), b["relative_mechanical_profile"].tolist())
        self.assertNotEqual(a["measurement_review_priority"].tolist(), b["measurement_review_priority"].tolist())

    def test_strict_merge_uses_pitch_ids(self):
        profile = pd.DataFrame({
            "pitcher_id": ["A", "B"],
            "pitch_id": [1, 1],
            "relative_mechanical_profile": ["higher", "lower"],
        })
        clusters = pd.DataFrame({
            "pitcher_id": ["B", "A"],
            "pitch_id": [1, 1],
            "cluster": [2, 0],
        })
        merged, method = merger.strict_merge(profile, clusters, "cluster")
        self.assertEqual(method, "unique_pitch_id")
        self.assertEqual(merged["kmeans_cluster"].tolist(), [0, 2])

    def test_final_holdout_manifest_consistency(self):
        from run_pipeline import parameter_record, VALIDATED_DEFAULTS
        from argparse import Namespace
        config = json.loads((ROOT / "config/pipeline_parameters.json").read_text())
        inputs = Namespace(reuse_pose=False, **VALIDATED_DEFAULTS)
        manifest = parameter_record(inputs)
        self.assertEqual(manifest["holdout_validation_reference"]["ffc_mae_frames"], 0.60)
        for key, value in config["event_validation_reference"].items():
            if key != "provenance":
                self.assertEqual(manifest["holdout_validation_reference"][key], value)


if __name__ == "__main__":
    unittest.main()
