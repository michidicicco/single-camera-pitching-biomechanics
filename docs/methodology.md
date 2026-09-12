# Methodology Overview

The framework converts a side-view pitching clip into pitch-level biomechanical measurements and multivariable movement-pattern summaries.

1. **Pose estimation:** frame-level anatomical landmarks are extracted using MediaPipe Pose Landmarker.
2. **Trajectory preprocessing:** short internal gaps may be interpolated up to the validated maximum duration and trajectories are smoothed using time-based Savitzky-Golay filtering.
3. **Event localization:** FFC and release are estimated automatically and evaluated against an independent manual-reference holdout set.
4. **Measurement QC:** frame-rate validity, pose detection, landmark visibility, event ordering, event confidence, missing-data gaps, and FFC-to-release timing are assessed separately from mechanical characterization.
5. **Feature extraction:** pitch-level variables summarize elbow motion, wrist speed, trunk behavior, projected hip-shoulder separation, stride, knee mechanics, and event timing.
6. **Relative mechanical characterization:** pooled and within-pitcher reference structures are generated from selected standardized kinematic variables.
7. **PCA:** PCA describes covariance structure and supports visualization.
8. **K-means clustering:** clustering is performed on the standardized retained original biomechanical features, not on PCA coordinates.
9. **Robustness analyses:** initialization stability, participant omission, participant-balanced resampling, feature-domain analyses, within-pitcher normalization, feature holdout, and null/bootstrap analyses characterize the structure and its limitations.

Cluster labels summarize regions of similarity in a continuous feature space. They are not diagnostic classes, injury categories, or universal pitching types.
