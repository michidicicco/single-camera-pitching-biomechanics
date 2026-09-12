# Data Dictionary

## Required Metadata Fields

| Field | Description |
|---|---|
| `video_file` | Source video filename |
| `pitcher_id` | Participant identifier used within the study dataset |
| `pitch_id` | Pitch identifier within participant |
| `throwing_side` | Throwing-side landmark convention used by the analysis |
| `camera_view` | Recording view |
| `height_m` | Participant height in meters when available |

The canonical key is `unique_pitch_id = pitcher_id + "_" + pitch_id`. `pitch_id` alone is not assumed to be globally unique.

## Primary Retained Biomechanical Features

The associated study retained 32 biomechanical features for the primary PCA/K-means analysis:

- `ffc_to_release_ms`
- `stride_length_norm_at_release`
- `stride_length_norm_max_ffc_to_release`
- `elbow_angle_deg_at_ffc`
- `elbow_angle_deg_at_release`
- `elbow_angle_deg_mean_ffc_to_release`
- `elbow_angle_deg_range_ffc_to_release`
- `elbow_ang_vel_max_ffc_to_release`
- `elbow_ang_vel_mean_ffc_to_release`
- `wrist_speed_max_ffc_to_release`
- `wrist_speed_mean_ffc_to_release`
- `wrist_speed_at_release`
- `upper_arm_angle_deg_at_release`
- `forearm_angle_deg_at_release`
- `trunk_tilt_deg_at_ffc`
- `trunk_tilt_deg_at_release`
- `trunk_tilt_deg_mean_ffc_to_release`
- `trunk_tilt_deg_range_ffc_to_release`
- `trunk_tilt_vel_max_ffc_to_release`
- `hip_shoulder_sep_deg_at_ffc`
- `hip_shoulder_sep_deg_at_release`
- `hip_shoulder_sep_deg_max_ffc_to_release`
- `hip_shoulder_sep_deg_mean_ffc_to_release`
- `hip_shoulder_sep_vel_max_ffc_to_release`
- `lead_knee_angle_deg_at_ffc`
- `lead_knee_angle_deg_at_release`
- `lead_knee_extension_deg_ffc_to_release`
- `trail_knee_angle_deg_at_ffc`
- `trail_knee_angle_deg_at_release`
- `max_sep_to_release_ms`
- `max_elbow_flex_to_release_ms`
- `max_trunk_tilt_vel_to_release_ms`

These variables are projected video-derived kinematic measures and should not be interpreted as equivalent to laboratory 3D joint kinematics or kinetics.

## Public Naming

Public-facing outputs should use neutral terminology such as `relative_mechanical_score`, `relative_mechanical_band`, `relative_mechanical_profile`, `measurement_qc_status`, `cluster`, and `unique_pitch_id`.
