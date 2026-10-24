# Aggregate Study Results

This folder contains publication-safe aggregate outputs from the associated 156-pitch study analysis.

Participant-level cluster assignments, pitch-level feature tables, raw video, manual event annotations, and identifiable filenames are intentionally not distributed here.

Values describe the study cohort and are not population norms or clinical thresholds.

## Thesis-aligned interpretation

- `event_validation_summary.csv` reports the **final 25-pitch holdout** against manual video annotations (FFC MAE 0.60 frames / 17.3 ms; ball-release MAE 0.44 frames / 14.7 ms; FFC-to-release duration MAE 0.96 frames / 29.3 ms). Holdout pitches were distinct from development annotations, but came from pitchers represented in development. An earlier intermediate event-validation summary had different errors and must not be cited as the final holdout.
- `study_summary.json` distinguishes the **50-initialization candidate-selection** silhouette (`0.2607199563`) from the **100-initialization frozen primary refit** (`0.2540620404`). The legacy `primary_silhouette` field is retained as the candidate-selection value for backward compatibility.
- `cluster_summary.csv` is a post hoc comparison of descriptive relative mechanical profiles; the relative scoring and clustering pathways share five biomechanical inputs, so agreement is not independent clinical validation.
- The public release does **not** contain participant-level raw data or every supplementary analysis from the thesis; the study-specific reported figures are not all independently reproducible from the public aggregate results alone.
