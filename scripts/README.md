# Analysis Scripts

The public codebase uses stable descriptive filenames rather than development-version suffixes.

| Script | Role |
|---|---|
| `extract_pose.py` | Frame-level MediaPipe pose extraction |
| `extract_features.py` | Pitch-level event localization and biomechanical feature extraction |
| `quality_control.py` | Measurement quality control |
| `relative_mechanics.py` | Pooled and within-pitcher relative mechanical characterization |
| `prepare_analysis_data.py` | Deterministic model matrix construction, PCA, and primary K-means analysis |
| `cluster_analysis.py` | Initialization, participant-omission, balanced-resampling, and feature-domain robustness analyses |
| `mechanical_profiles.py` | Dataset-relative mechanical profile construction |
| `merge_analysis_outputs.py` | Strict unique-key merge of independent analytical pathways |
| `manual_event_annotator.py` | Blinded manual event annotation utility |
| `manual_event_annotator_holdout.py` | Independent holdout event-annotation utility |
| `validate_events.py` | FFC and release timing agreement analysis |
| `characterize_clusters.py` | Aggregate biomechanical cluster characterization |
| `feature_holdout_analysis.py` | Relative-mechanics feature-holdout clustering |
| `pitcher_adjusted_analysis.py` | Pitcher-adjusted cluster relevance analysis |
| `within_pitcher_clustering.py` | Within-pitcher-normalized clustering |
| `null_bootstrap_analysis.py` | Null-model and pitcher-stratified bootstrap analysis |
| `analysis_utils.py` | Shared utilities for secondary analyses |
| `run_pipeline.py` | Raw-video orchestration through relative mechanical characterization |

## Study analysis order

```text
run_pipeline.py
    -> prepare_analysis_data.py
    -> cluster_analysis.py
    -> mechanical_profiles.py
    -> merge_analysis_outputs.py
    -> secondary analyses as required
```

The relative mechanical and clustering pathways are generated independently and integrated only after each path is complete.

See [`../docs/reproducibility_validation.md`](../docs/reproducibility_validation.md) for the regression comparison against the archived study outputs.
