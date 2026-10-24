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

## Final-thesis result clarification

The final 25-pitch holdout reported FFC MAE 0.60 frames (17.3 ms), release MAE 0.44 frames (14.7 ms), and FFC-to-release duration MAE 0.96 frames (29.3 ms). `results/event_validation_summary.csv` refers to this **final holdout**, not an intermediate development-stage annotation summary.

The K-means candidate selection and frozen refit should not be conflated: silhouette 0.2607199563 came from the 50-initialization candidate-selection fit; the 100-initialization frozen final fit yielded 0.2540620404. Null/bootstrap tests used the candidate-selection configuration at fixed k = 5.

The public `pitcher_adjusted_analysis.py` accepts the public normalized `higher` profile label. Historical local research tables used a longer higher-profile label, and those historical strings are not the clean repository's default interface. Conversion to the public label convention is necessary before using historical inputs with that public script; the published aggregate effect sizes do not imply the historical strings can be passed through unchanged.

The archived Zenodo release is versioned separately; edits to the living GitHub repository do not retroactively alter archived Zenodo bytes.
