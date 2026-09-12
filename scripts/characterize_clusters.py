#!/usr/bin/env python
"""
Describe the biomechanical characteristics of each VALIDATED primary K-means cluster.

This is a secondary interpretive analysis. It does not redefine the primary
clusters and does not label clusters as clinical classes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from analysis_utils import ensure_unique_pitch_id, resolve_relative_mechanics_features


CURATED_BIOMECH_FEATURES = [
    "ffc_to_release_ms",
    "stride_length_norm_at_ffc",
    "stride_length_norm_at_release",
    "stride_length_norm_max_ffc_to_release",
    "elbow_angle_deg_at_ffc",
    "elbow_angle_deg_at_release",
    "elbow_angle_deg_mean_ffc_to_release",
    "elbow_angle_deg_range_ffc_to_release",
    "elbow_ang_vel_max_ffc_to_release",
    "elbow_ang_vel_mean_ffc_to_release",
    "wrist_speed_norm_bh_per_s_max_ffc_to_release",
    "wrist_speed_norm_bh_per_s_mean_ffc_to_release",
    "wrist_speed_norm_bh_per_s_at_release",
    "wrist_speed_max_ffc_to_release",
    "wrist_speed_mean_ffc_to_release",
    "wrist_speed_at_release",
    "upper_arm_angle_deg_at_release",
    "forearm_angle_deg_at_release",
    "trunk_tilt_deg_at_ffc",
    "trunk_tilt_deg_at_release",
    "trunk_tilt_deg_mean_ffc_to_release",
    "trunk_tilt_deg_range_ffc_to_release",
    "trunk_tilt_vel_max_ffc_to_release",
    "hip_shoulder_sep_deg_at_ffc",
    "hip_shoulder_sep_deg_at_release",
    "hip_shoulder_sep_deg_max_ffc_to_release",
    "hip_shoulder_sep_deg_mean_ffc_to_release",
    "hip_shoulder_sep_vel_max_ffc_to_release",
    "lead_knee_angle_deg_at_ffc",
    "lead_knee_angle_deg_at_release",
    "lead_knee_extension_deg_ffc_to_release",
    "trail_knee_angle_deg_at_ffc",
    "trail_knee_angle_deg_at_release",
    "max_sep_to_release_ms",
    "max_elbow_flex_to_release_ms",
    "max_trunk_tilt_vel_to_release_ms",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Merged profiles-and-clusters CSV.")
    p.add_argument("--cluster-col", default="kmeans_cluster")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-n", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_unique_pitch_id(pd.read_csv(args.input))
    if args.cluster_col not in df.columns:
        raise ValueError(f"Missing cluster column: {args.cluster_col}")

    features = [c for c in CURATED_BIOMECH_FEATURES if c in df.columns]
    if not features:
        raise ValueError("No curated biomechanical features were found.")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    means = X.mean()
    sd = X.std(ddof=0).replace(0, np.nan)
    Z = (X - means) / sd

    raw_rows = []
    z_rows = []
    for cluster, idx in df.groupby(args.cluster_col).groups.items():
        for feature in features:
            vals = X.loc[idx, feature].dropna()
            zvals = Z.loc[idx, feature].dropna()
            raw_rows.append({
                "cluster": cluster, "feature": feature, "n": len(vals),
                "mean": vals.mean(), "median": vals.median(),
                "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            })
            z_rows.append({
                "cluster": cluster, "feature": feature, "n": len(zvals),
                "mean_z": zvals.mean(),
            })

    raw = pd.DataFrame(raw_rows)
    ztab = pd.DataFrame(z_rows)
    raw.to_csv(out_dir / "cluster_biomechanical_summary_original_units.csv", index=False)
    ztab.to_csv(out_dir / "cluster_biomechanical_summary_zscore.csv", index=False)

    top = (
        ztab.assign(abs_mean_z=lambda d: d["mean_z"].abs())
        .sort_values(["cluster", "abs_mean_z"], ascending=[True, False])
        .groupby("cluster", as_index=False)
        .head(args.top_n)
    )
    top.to_csv(out_dir / "cluster_top_distinguishing_biomechanical_features.csv", index=False)

    relative_mechanics_features = resolve_relative_mechanics_features(df.columns)
    filtering_rows = []
    for logical, col in relative_mechanics_features.items():
        for cluster, idx in df.groupby(args.cluster_col).groups.items():
            vals = pd.to_numeric(df.loc[idx, col], errors="coerce").dropna()
            filtering_rows.append({
                "relative_mechanics_feature": logical,
                "source_column": col,
                "cluster": cluster,
                "n": len(vals),
                "mean": vals.mean(),
                "median": vals.median(),
                "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            })
    pd.DataFrame(filtering_rows).to_csv(
        out_dir / "cluster_filtering_relevant_feature_summary.csv", index=False
    )

    if "relative_mechanical_profile" in df.columns:
        ct = pd.crosstab(df[args.cluster_col], df["relative_mechanical_profile"])
        ct.to_csv(out_dir / "cluster_by_relative_mechanical_profile_counts.csv")
        prop = pd.crosstab(
            df[args.cluster_col], df["relative_mechanical_profile"], normalize="index"
        )
        prop.to_csv(out_dir / "cluster_by_relative_mechanical_profile_proportions.csv")

    if "pitcher_id" in df.columns:
        pd.crosstab(df["pitcher_id"], df[args.cluster_col]).to_csv(
            out_dir / "pitcher_by_cluster_counts.csv"
        )

    lines = [
        "CLUSTER BIOMECHANICAL SUMMARY",
        "=" * 72,
        "",
        "Interpretation note: these are descriptive mechanical",
        "patterns, not validated clinical classes.",
        "",
    ]
    for cluster in sorted(df[args.cluster_col].dropna().unique()):
        lines.append(f"Cluster {cluster} (n={(df[args.cluster_col] == cluster).sum()}):")
        sub = top[top["cluster"] == cluster]
        for _, row in sub.iterrows():
            direction = "higher" if row["mean_z"] > 0 else "lower"
            lines.append(
                f"  - {row['feature']}: {direction} than pooled mean "
                f"(cluster mean z={row['mean_z']:.3f})"
            )
        lines.append("")
    (out_dir / "cluster_biomechanical_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("Cluster biomechanical characterization complete.")
    print(f"Output directory: {out_dir}")
    print(f"Clusters: {sorted(df[args.cluster_col].dropna().unique().tolist())}")
    print(f"Curated biomechanical features summarized: {len(features)}")
    print(f"Relative-mechanics features resolved: {relative_mechanics_features}")


if __name__ == "__main__":
    main()
