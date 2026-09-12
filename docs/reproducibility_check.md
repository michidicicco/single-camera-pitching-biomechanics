# Reproducibility Check

The publication-facing scripts were derived from the canonical study scripts and renamed to remove development-version suffixes and provisional terminology without intentionally changing the study calculations. The public primary-analysis code uses the prespecified 36-feature candidate set and reproduces the retained 32-feature model matrix.

Regression checks were performed against the archived 156-pitch study outputs. Exact numerical reproduction should use the documented study software environment because numerical-library versions can affect optimization details even when the analysis logic is unchanged.

## Regression checks

- Measurement-QC status, review count, exclusion count, and model eligibility matched for all 156 pitches.
- Pooled and within-pitcher relative mechanical scores matched to floating-point precision, with maximum absolute differences below `7e-16`.
- Relative mechanical bands reproduced the same lower/moderate/higher assignments after neutral label mapping.
- The retained primary model matrix contained the same 32 biomechanical features in the same order.
- Primary K-means assignments matched for all 156 pitches.
- Primary PCA explained-variance values matched to floating-point precision.
- Primary K-means selected `k = 5` with silhouette `0.2607199563`.
- Cluster robustness reproduced silhouette `0.2540620404`, mean initialization ARI `0.8887180235`, mean leave-one-pitcher-out ARI `0.9560888578`, and mean pitcher-balanced resampling ARI `0.8397765681`.
- Relative mechanical profiles reproduced `48` lower, `58` moderate, and `50` higher observations after neutral label mapping.
- Relative-mechanics feature-holdout clustering reproduced `k = 2`, silhouette `0.2719438487`, and ARI `0.2179695310` versus the primary solution.
- Pitcher-adjusted effect sizes and within-pitcher permutation p-values matched the archived study outputs.
- Within-pitcher-normalized clustering reproduced `k = 2`, silhouette `0.1394119045`, and ARI `0.0397220969` versus the primary pooled solution.
- Within-pitcher normalization with relative-mechanics feature holdout reproduced `k = 2`, silhouette `0.1334758380`, and ARI `0.0489281033`.
- The public null-model implementation was compared directly with the archived implementation under matched test settings; the first 20 permutation results in the audit matched exactly. The archived 2,000-permutation/2,000-bootstrap study summary is provided in `results/null_bootstrap_summary.json`.

## Naming changes

Public-facing names use neutral terminology such as `relative_mechanical_score`, `relative_mechanical_band`, and `relative_mechanical_profile`. Historical development labels are not part of the public interface.

## Data scope

Regression testing used the archived participant-level study outputs locally. Those participant-level files are not included in the public repository.
