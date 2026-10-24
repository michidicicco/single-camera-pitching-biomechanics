# v1.0.1 — Final Thesis Alignment and Reproducibility Update

This maintenance release aligns the public research software and aggregate documentation with the finalized master's thesis analyses. **No existing script filenames were renamed, and no primary scientific algorithm or frozen event-detection parameter was intentionally changed.**

## Updates

- Reconciled final V7 25-pitch event-validation summaries: front-foot contact MAE **0.60 frames (17.3 ms)**, release MAE **0.44 frames (14.7 ms)**, and FFC-to-release duration MAE **0.96 frames (29.3 ms)**. The earlier exported feature tables corresponded to different event predictions and are not the final V7 holdout.
- Distinguished the 50-initialization K-means candidate-selection silhouette (**0.2607199563**) from the 100-initialization frozen refit silhouette (**0.2540620404**).
- Clarified study sample size (8 pitchers, 156 pitches), QC disposition (122 pass, 34 watch, 0 excluded), and 32 retained biomechanical features.
- Documented relative mechanical profile counts (48 lower, 58 moderate, 50 higher), participant-dependent clustering, and the difference between descriptive concordance and independent clinical validation.
- Added compatibility for the archived `higher_watch_relative_mechanical_profile` label alongside the public `higher` label in pitcher-adjusted analysis.
- Added nonparticipant synthetic regression smoke tests and a GitHub Actions test job.
- Added [thesis-alignment audit](final_thesis_alignment_audit.md) with the scope and limits of verified results.

## Verification

The GitHub Actions workflow for commit `cf698d594c27a810cf50a5f0e20c6bcf50f391ea` passed. Several final thesis numerical results were reconciled against author-supplied analytical output files. Full raw-video reruns, detailed Ward transitions, and a complete independent reproduction of all restricted-data analyses remain outside the scope of the public automated test suite.

## Scope and privacy

This is a research-oriented, exploratory 2D pitching-kinematics pipeline. It is **not** a clinically validated estimator of elbow joint loading, UCL strain, or injury probability. No identifiable participant videos or participant-level study datasets are included.

## Release provenance

- Prior GitHub release: `v1.0.0` (September 12, 2026)
- Proposed new GitHub tag: `v1.0.1`
- Target commit: `cf698d594c27a810cf50a5f0e20c6bcf50f391ea`
- Zenodo: publish a **new version-specific archive** when the GitHub release is published; do not claim that the existing archived v1.0.0 has changed. Update citation metadata and DOI links after Zenodo assigns the new version DOI.

This document contains proposed release notes. **Its presence does not mean that a GitHub release or Zenodo version has been published.**
