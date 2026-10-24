# Final Thesis Alignment Audit — 24 October 2026

This audit documents the correspondence between the final M.S. thesis and the living public repository. The publication-facing filenames and core study algorithms were retained.

## Confirmed against study materials supplied for the thesis audit

- Study scope: 8 pitchers, 156 pitches, 122 QC-pass, 34 QC-watch, none excluded.
- Model matrix: 32 retained biomechanical features from 36 candidates.
- Descriptive PCA: PC1 29.7267%; first five components 72.0379%.
- K-means model selection: k=5, 50-initialization candidate silhouette 0.2607199563.
- Frozen five-cluster refit: 100-initialization silhouette 0.2540620404; sizes 21, 50, 32, 15, 38.
- Within-pitcher normalization: k=2, silhouette 0.1394119, ARI 0.0397221 versus the pooled partition.
- Five-feature holdout: k=2, silhouette 0.2719438, ARI 0.2179695.
- Relative mechanical profile counts: lower 48, moderate 58, higher 50; profile/cluster count table matched.
- Pitcher-adjusted exploratory tests: six effect sizes and associated within-pitcher permutation p-values matched archived aggregate values. Five of six met the reported *unadjusted* criterion.
- Final V7 holdout: 25 pitches, FFC 0.60 frames/17.33 ms, release 0.44 frames/14.67 ms, duration 0.96 frames MAE; FFC within one frame 100%, release 84%. Earlier exported automated-feature tables produced **different** event estimates (FFC 0.80 frames, release 0.72 frames); they must not be substituted for the V7 holdout output.
- Preprocessing condition summaries: 50/100/200 ms retained 32/31/32 features, preferred k=6/6/5; representation agreement with the 200-ms reference included ARIs 0.796427 and 0.858606 and five-PC Procrustes disparities 0.193585 and 0.146057.

## Automated public-code checks

GitHub Actions runs Python compilation, JSON parsing and five synthetic/nonparticipant computational smoke tests. The expanded workflow succeeded on commit `ff1811447bb711bc3a6d6d9ab3aefdfc41e1e833`. These checks are regression/smoke tests; they do **not** replace reproduction of the restricted raw-data pipeline.

## Limits / not independently re-executed in this final repository audit

- Entire raw-video-to-feature pipeline on the restricted study videos, including the exact provenance of the V7 per-pitch event predictions.
- All preprocessing experiments re-extracted from pose trajectories (reported aggregate comparisons were checked).
- Archived candidate-versus-Ward and frozen-versus-Ward comparisons at every k, including supplementary stability tables.
- Detailed k=5-to-k=6 cluster-split counts and assignment transitions.
- Full 2,000-iteration null/bootstrap study from raw restricted inputs using the published code in this audit (archived summary and previous matched small-setting checks are documented elsewhere).
- Exact binary equality of the living GitHub repository with the historical Zenodo v1.0.0 archive.

## Scientific interpretation and privacy

PCA is descriptive; primary clustering uses standardized retained original features, not PCA scores. Clusters and relative mechanical profiles are cohort-specific exploratory descriptors and do not measure UCL strain, elbow torque, injury probability or clinical risk. Scoring and clustering are separate computational pathways but share five underlying biomechanical variables, so their post hoc agreement is not independent validation. Participant-level data and identifiable videos remain private.

## Release status

The only published GitHub release observed during this audit was `v1.0.0`, dated 12 September 2026. GitHub changes since that tag do not modify the original version or the Zenodo archive. **This document does not claim that a new GitHub or Zenodo version has been released.** Any future release should identify its new tag/version and archive DOI distinctly.
