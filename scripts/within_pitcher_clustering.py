#!/usr/bin/env python
"""
Secondary within-pitcher-normalized clustering.

Each retained primary model feature is z-scored within pitcher before K-means.
This asks what multivariable pitch-to-pitch structure remains after removing
each pitcher's average mechanical signature.

Optional --exclude-relative-mechanics-features additionally removes the five
variables used in the relative mechanical score before clustering.

This analysis does not replace the primary pooled clustering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from analysis_utils import (
    adjusted_rand_index,
    ensure_unique_pitch_id,
    kmeans_numpy,
    model_feature_columns,
    resolve_relative_mechanics_features,
    select_k,
    silhouette_score_numpy,
    standardize_matrix,
    within_pitcher_zscore,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ready", required=True)
    p.add_argument("--full-data", required=False,
                   help="Optional validated merged profiles-and-clusters CSV for comparisons.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--exclude-relative-mechanics-features", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ensure_unique_pitch_id(pd.read_csv(args.model_ready))
    if "pitcher_id" not in model.columns:
        raise ValueError("pitcher_id is required in model-ready input.")

    feature_cols = model_feature_columns(model)
    removed_relative_mechanics = {}
    if args.exclude_relative_mechanics_features:
        removed_relative_mechanics = resolve_relative_mechanics_features(feature_cols)
        remove_cols = set(removed_relative_mechanics.values())
        feature_cols = [c for c in feature_cols if c not in remove_cols]

    if len(feature_cols) < 2:
        raise ValueError("Fewer than two features remain.")

    Xwp = within_pitcher_zscore(model, feature_cols)
    if Xwp.isna().any().any():
        raise ValueError("Unexpected missing values after within-pitcher normalization.")

    Z, _, _ = standardize_matrix(Xwp)

    selection, best_k = select_k(Z, max_k=args.max_k, seed=args.seed, n_init=50)
    labels, centers, inertia, n_iter = kmeans_numpy(
        Z, k=best_k, seed=args.seed, n_init=100, max_iter=500
    )
    sil = silhouette_score_numpy(Z, labels)

    selection.to_csv(out_dir / "within_pitcher_kmeans_model_selection.csv", index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(
        out_dir / "within_pitcher_cluster_feature_list.csv", index=False
    )

    ids = [c for c in ["unique_pitch_id", "video_file", "pitcher_id", "pitch_id"] if c in model.columns]
    assignments = model[ids].copy()
    assignments["within_pitcher_cluster"] = labels
    assignments.to_csv(out_dir / "within_pitcher_cluster_assignments.csv", index=False)

    pd.crosstab(assignments["pitcher_id"], assignments["within_pitcher_cluster"]).to_csv(
        out_dir / "within_pitcher_pitcher_by_cluster_counts.csv"
    )

    ari_vs_primary = None
    if args.full_data:
        full = ensure_unique_pitch_id(pd.read_csv(args.full_data))
        merged = assignments.merge(full, on="unique_pitch_id", how="left", validate="one_to_one",
                                   suffixes=("", "_full"))

        primary = None
        for c in ["kmeans_cluster", "cluster"]:
            if c in merged.columns:
                primary = c
                break
        if primary:
            valid = merged[primary].notna()
            if valid.sum() >= 3:
                ari_vs_primary = adjusted_rand_index(
                    merged.loc[valid, "within_pitcher_cluster"],
                    merged.loc[valid, primary],
                )

        if "relative_mechanical_profile" in merged.columns:
            pd.crosstab(
                merged["within_pitcher_cluster"], merged["relative_mechanical_profile"]
            ).to_csv(out_dir / "within_pitcher_cluster_by_profile_counts.csv")
            pd.crosstab(
                merged["within_pitcher_cluster"], merged["relative_mechanical_profile"],
                normalize="index"
            ).to_csv(out_dir / "within_pitcher_cluster_by_profile_proportions.csv")

    summary = {
        "analysis": "secondary within-pitcher-normalized clustering",
        "n_pitches": int(len(model)),
        "n_pitchers": int(model["pitcher_id"].nunique()),
        "features_used": int(len(feature_cols)),
        "relative_mechanics_features_excluded": removed_relative_mechanics,
        "best_k": int(best_k),
        "silhouette": float(sil),
        "ari_vs_primary_pooled_clusters": ari_vs_primary,
        "interpretation": (
            "This sensitivity analysis removes each pitcher's feature-specific mean "
            "and scale before clustering, emphasizing within-pitcher pitch-to-pitch "
            "variation rather than pooled between-pitcher mechanical signatures."
        ),
    }
    (out_dir / "within_pitcher_clustering_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "within_pitcher_clustering_summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8"
    )

    print("Within-pitcher-normalized clustering complete.")
    print(f"Output directory: {out_dir}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Best k: {best_k}")
    print(f"Silhouette: {sil:.4f}")
    if ari_vs_primary is not None:
        print(f"ARI vs primary pooled clusters: {ari_vs_primary:.4f}")


if __name__ == "__main__":
    main()
