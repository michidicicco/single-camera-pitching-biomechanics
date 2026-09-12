# Public Script Migration

The publication-facing repository uses neutral, stable filenames. Development suffixes and provisional labels should not appear in the public codebase.

| Role | Public filename |
|---|---|
| Frame-level pose extraction | `extract_pose.py` |
| Pitch-level feature extraction | `extract_features.py` |
| Measurement QC | `quality_control.py` |
| Relative mechanical calculations | `relative_mechanics.py` |
| Primary model-matrix construction and PCA/K-means | `prepare_analysis_data.py` |
| Cluster robustness analyses | `cluster_analysis.py` |
| Relative mechanical profile generation | `mechanical_profiles.py` |
| Post hoc cluster/profile integration | `merge_analysis_outputs.py` |
| Blinded event annotation | `manual_event_annotator.py` |
| Independent holdout annotation | `manual_event_annotator_holdout.py` |
| Event validation | `validate_events.py` |
| Cluster biomechanical summaries | `characterize_clusters.py` |
| Screening-feature holdout analysis | `feature_holdout_analysis.py` |
| Pitcher-adjusted analysis | `pitcher_adjusted_analysis.py` |
| Within-pitcher normalized clustering | `within_pitcher_clustering.py` |
| Null-model and bootstrap analysis | `null_bootstrap_analysis.py` |
| Raw-video orchestration | `run_pipeline.py` |

Do not populate this folder from older similarly named development scripts. The canonical study scripts must be copied from the exact analysis set that generated the publication results, renamed, and regression-tested before release.
