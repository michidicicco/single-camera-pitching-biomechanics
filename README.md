# Single-Camera Pitching Biomechanics

Open-source research software for single-camera 2D baseball pitching biomechanics.

This repository supports a field-deployable workflow for frame-level pose estimation, automated front-foot contact (FFC) and ball-release localization, trajectory preprocessing, measurement quality control, pitch-level kinematic feature extraction, pooled and within-pitcher relative mechanical characterization, PCA, K-means clustering, and robustness analyses.

The project was developed as part of a biomedical engineering graduate research study at the University of North Dakota.

## Scope

This software is intended for research-oriented biomechanical characterization. It does **not** directly measure or estimate UCL strain, elbow valgus torque, tissue stress, injury probability, clinical diagnosis, or whether a pitch is "safe" or "unsafe." Relative mechanical scores and profiles are dataset-relative descriptors, not clinical thresholds.

## Study Snapshot

The associated study dataset contained 8 pitchers and 156 pitch observations. Measurement QC classified 122 observations as pass and 34 for review; no observations were excluded from the primary analysis. The primary multivariable matrix retained 32 biomechanical features.

K-means clustering was performed on the 32 standardized retained biomechanical features, not on PCA scores. Candidate values of `k = 2` through `k = 6` were evaluated. The selected solution used `k = 5` with silhouette `0.2607199563`.

These values describe the associated study and should not be interpreted as universal population parameters.

## Processing Pipeline

```text
Side-view video
      |
      v
Pose estimation
      |
      v
Trajectory preprocessing
      |
      v
FFC + release localization
      |
      v
Measurement quality control
      |
      v
Kinematic feature extraction
      |
      +------------------------------+
      |                              |
      v                              v
Relative mechanical            PCA + K-means
characterization               clustering
      |                              |
      v                              v
Mechanical profiles          Robustness analyses
      |                              |
      +--------------+---------------+
                     |
                     v
          Post hoc integration
```

The relative mechanical and clustering pathways are generated independently and are merged only after both analyses are complete.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── requirements.txt
├── environment.yml
├── config/
├── docs/
├── scripts/
├── examples/
├── results/
└── figures/
```

## Analysis Code

The publication-facing code uses stable descriptive filenames rather than development-version suffixes. Core scripts include:

```text
extract_pose.py
extract_features.py
quality_control.py
relative_mechanics.py
prepare_analysis_data.py
cluster_analysis.py
mechanical_profiles.py
merge_analysis_outputs.py
characterize_clusters.py
feature_holdout_analysis.py
pitcher_adjusted_analysis.py
within_pitcher_clustering.py
null_bootstrap_analysis.py
validate_events.py
run_pipeline.py
```

See [`scripts/README.md`](scripts/README.md) for the complete script map and analysis order.

## Reproducibility Validation

The publication-facing scripts were regression-tested against the archived 156-pitch study outputs. The retained feature set, primary cluster assignments, PCA variance, relative mechanical scores, profile counts, robustness metrics, feature-holdout results, pitcher-adjusted results, and within-pitcher analyses were reproduced. The null/bootstrap implementation was also compared directly against the archived implementation using matched test settings.

See [`docs/reproducibility_validation.md`](docs/reproducibility_validation.md) for details.

## Aggregate Results

Publication-safe aggregate study outputs are provided in [`results/`](results/). Participant-level cluster assignments and pitch-level feature tables are not publicly distributed.

## Video Acquisition

Recommended acquisition conditions include side-view orientation, full-body visibility, landscape video, a stable camera approximately perpendicular to the throwing direction, approximately 10–15 ft camera distance when feasible, camera height around waist-to-chest level, target frame rate of at least 30 fps, adequate lighting, and minimal motion blur or occlusion.

See [`docs/acquisition_protocol.md`](docs/acquisition_protocol.md).

## Metadata

The basic metadata schema is:

```text
video_file
pitcher_id
pitch_id
throwing_side
camera_view
height_m
```

The canonical pitch identifier is:

```text
unique_pitch_id = pitcher_id + "_" + pitch_id
```

All joins between analytical outputs use the verified unique pitch identifier. Row-order merging is not permitted.

## Reproducibility

The validated parameter set and software environment are documented in [`config/`](config/) and [`docs/reproducibility.md`](docs/reproducibility.md).

## Data Availability and Privacy

Raw participant videos are **not** distributed in this repository because video of human participants may be identifiable. This repository distributes source code, processing parameters, software-environment information, data schemas, synthetic/example inputs, publication-safe aggregate results, and reproducibility documentation.

Participant-level research data should only be released when permitted by the study consent, IRB/ethics requirements, institutional policy, and applicable privacy constraints.

See [`docs/data_availability.md`](docs/data_availability.md).

## Pose Model

The MediaPipe Pose Landmarker model file is not bundled here. Users should obtain the appropriate model through the official MediaPipe distribution and place it locally according to the pipeline instructions.

## Citation

A machine-readable citation file is provided in [`CITATION.cff`](CITATION.cff). After the first archived software release receives a DOI, the DOI and associated thesis/journal publication should be added as the preferred citation.

## License

Source code and original repository materials are released under the MIT License unless a file or third-party dependency states otherwise. Third-party software, models, and datasets remain subject to their own licenses and terms.
