# Relative Mechanical Characterization

The relative mechanical pathway summarizes selected video-derived kinematic characteristics using pooled and within-pitcher reference frameworks.

| Feature | Weight |
|---|---:|
| Maximum throwing-elbow angular velocity | 0.30 |
| Maximum normalized wrist speed | 0.20 |
| Maximum projected hip-shoulder separation velocity | 0.20 |
| Maximum trunk-tilt angular velocity | 0.15 |
| Trunk tilt at release | 0.15 |

The selected variables are standardized before weighting.

**Pooled reference:** compares a pitch with the complete eligible study sample.

**Within-pitcher reference:** compares a pitch with that pitcher's own recorded distribution. A within-pitcher reference requires at least five pitches.

Relative mechanical bands use the 33rd and 67th percentiles of the applicable reference distribution. The public interface uses `lower`, `moderate`, and `higher`. These labels indicate relative position within the selected reference distribution and do not indicate injury probability, tissue damage, clinical abnormality, or safety.

Measurement-QC status is kept separate from the mechanical profile so lower data quality does not automatically make a pitch mechanically more elevated.
