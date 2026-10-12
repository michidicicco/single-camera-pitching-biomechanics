# Hierarchical clustering comparison (secondary analysis)

This script compares the study's archived custom NumPy K-means with Ward-linkage agglomerative hierarchical clustering. It is an exploratory **cross-algorithm sensitivity analysis**, not a replacement for the validated pooled K-means result.

## Required local inputs

Run the existing `scripts/prepare_analysis_data.py` first. Keep the resulting files local:

- `outputs/analysis/model_ready.csv` (includes pitcher IDs and retained biomechanics)
- `outputs/analysis/retained_features.csv` (the exact names of the cleaned model features)

The feature list is mandatory and prevents accidental introduction of identifiers, QC metadata, or relative mechanical profile labels into model inputs.

## Run (from repository root)

```bash
conda activate pitching-biomechanics
python scripts/compare_clustering_methods.py --input outputs/analysis/model_ready.csv --features-file outputs/analysis/retained_features.csv --output-dir outputs/hierarchical_comparison --min-k 2 --max-k 6 --balanced-resamples 100
```

Windows PowerShell can use the same command on a single line.

For the verified 156-pitch local analysis, use:

```powershell
python scripts/compare_clustering_methods.py --input "outputs/analysis_cleaned_v2/pitch_features_model_ready_cleaned.csv" --features-file "outputs/analysis_cleaned_v2/retained_features.csv" --frozen-assignments "outputs/analysis_cleaned_v2/cleaned_cluster_assignments.csv" --output-dir "outputs/hierarchical_comparison"
```

This additionally computes the 100-initialization, 500-iteration, k=5 K-means refit (seed 13) and checks its partitions against the original frozen assignments using unique_pitch_id. The 50-initialization model-selection and resampling values are still reported separately. Do not conflate these two K-means fits.

## Outputs (aggregate-only)

- `model_selection.csv`: per-algorithm/per-k silhouette, Davies-Bouldin, Calinski-Harabasz
- `cross_algorithm_agreement.csv`: same-k adjusted Rand index comparing K-means with Ward
- `robustness_summary.csv`: leave-one-pitcher-out and pitcher-balanced ARI summaries for each algorithm and k
- `silhouette_comparison.png`: per-k comparison figure
- `summary.json`: number of observations/features, best k, 50-initialization model-selection ARI, frozen-refit silhouette, frozen-versus-Ward ARIs at k=5 and k=6, and (if supplied) ID-matched frozen-assignment verification

No pitch-level labels, videos, or pitcher identifiers are written to these files. Do not upload your private `model_ready.csv` or other participant data to GitHub.

## Interpretation

Cross-algorithm ARI is agreement, not biological validity. Per-method silhouette, DBI (lower better), and CHI (higher better) provide complementary separation criteria. Ward is deterministic at fixed data/settings, so repeated random-seed testing does not measure its stability. Participant omission and balanced sampling are more appropriate perturbations.

The frozen-reference verification requires an exact unique_pitch_id match and will abort on missing or duplicate IDs. The current script refits scaling within each resampled dataset and compares re-fitted labels against the original full-data labels on shared observations. This is a reproducibility/sensitivity comparison, not external generalization to unseen pitchers. The archived primary K-means method and results remain unchanged.
