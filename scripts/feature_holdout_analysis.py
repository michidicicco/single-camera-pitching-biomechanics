#!/usr/bin/env python
"""
Secondary relative-mechanics feature-holdout clustering analysis.

The five variables used in the study-defined relative mechanical score are
withheld from cluster construction when present in the primary model-ready
matrix. Clusters are then compared post hoc with those held-out relative-mechanics
variables and the independently generated relative mechanical profile.

This is a sensitivity analysis; it does not replace the primary clustering.
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
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ready", required=True)
    p.add_argument("--full-data", required=True,
                   help="Merged profiles-and-clusters CSV used for post hoc profile comparison.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ensure_unique_pitch_id(pd.read_csv(args.model_ready))
    full = ensure_unique_pitch_id(pd.read_csv(args.full_data))

    feature_cols = model_feature_columns(model)
    relative_mechanics_in_model = resolve_relative_mechanics_features(feature_cols)
    heldout_model_cols = sorted(set(relative_mechanics_in_model.values()))
    remaining = [c for c in feature_cols if c not in heldout_model_cols]

    if len(remaining) < 2:
        raise ValueError("Fewer than two model features remain after relative-mechanics-feature holdout.")

    X = model[remaining].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Model-ready input contains missing values in retained secondary features.")

    Z, _, _ = standardize_matrix(X)
    selection, best_k = select_k(Z, max_k=args.max_k, seed=args.seed, n_init=50)
    labels, centers, inertia, n_iter = kmeans_numpy(
        Z, k=best_k, seed=args.seed, n_init=100, max_iter=500
    )
    sil = silhouette_score_numpy(Z, labels)

    selection.to_csv(out_dir / "feature_holdout_kmeans_model_selection.csv", index=False)
    pd.DataFrame({"feature": remaining}).to_csv(
        out_dir / "feature_holdout_model_feature_list.csv", index=False
    )
    pd.DataFrame([
        {"logical_relative_mechanics_feature": logical, "model_column_withheld": col}
        for logical, col in relative_mechanics_in_model.items()
    ]).to_csv(out_dir / "feature_holdout_features_removed_from_clustering.csv", index=False)

    assignment_cols = [c for c in ["unique_pitch_id", "video_file", "pitcher_id", "pitch_id"] if c in model.columns]
    assignments = model[assignment_cols].copy()
    assignments["feature_holdout_cluster"] = labels
    assignments.to_csv(out_dir / "feature_holdout_cluster_assignments.csv", index=False)

    merged = assignments.merge(
        full, on="unique_pitch_id", how="left", validate="one_to_one",
        suffixes=("", "_full")
    )

    relative_mechanics_full = resolve_relative_mechanics_features(full.columns)
    summary_rows = []
    for logical, col in relative_mechanics_full.items():
        actual = col if col in merged.columns else f"{col}_full"
        if actual not in merged.columns:
            continue
        for cluster, idx in merged.groupby("feature_holdout_cluster").groups.items():
            vals = pd.to_numeric(merged.loc[idx, actual], errors="coerce").dropna()
            summary_rows.append({
                "relative_mechanics_feature": logical,
                "source_column": col,
                "feature_holdout_cluster": cluster,
                "n": len(vals),
                "mean": vals.mean(),
                "median": vals.median(),
                "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            })
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "feature_holdout_cluster_relative_mechanics_feature_summary.csv", index=False
    )

    profile_col = "relative_mechanical_profile"
    if profile_col in merged.columns:
        pd.crosstab(
            merged["feature_holdout_cluster"], merged[profile_col]
        ).to_csv(out_dir / "feature_holdout_cluster_by_profile_counts.csv")
        pd.crosstab(
            merged["feature_holdout_cluster"], merged[profile_col], normalize="index"
        ).to_csv(out_dir / "feature_holdout_cluster_by_profile_proportions.csv")

    primary_cluster_col = None
    for candidate in ["kmeans_cluster", "cluster"]:
        if candidate in merged.columns:
            primary_cluster_col = candidate
            break

    ari_vs_primary = None
    if primary_cluster_col is not None:
        valid = merged[primary_cluster_col].notna()
        if valid.sum() >= 3:
            ari_vs_primary = adjusted_rand_index(
                merged.loc[valid, "feature_holdout_cluster"],
                merged.loc[valid, primary_cluster_col],
            )

    summary = {
        "analysis": "secondary relative-mechanics feature-holdout clustering",
        "n_pitches": int(len(model)),
        "primary_model_features_available": int(len(feature_cols)),
        "relative_mechanics_features_removed_from_cluster_input": relative_mechanics_in_model,
        "remaining_cluster_features": int(len(remaining)),
        "best_k": int(best_k),
        "silhouette": float(sil),
        "ari_vs_primary_cluster_solution": ari_vs_primary,
        "interpretation": (
            "Clusters were formed without the study-defined relative-mechanics features "
            "that were present in the primary model matrix. Post hoc differences in "
            "held-out relative-mechanics variables provide sensitivity evidence of biomechanical "
            "relevance, not external validation of clinical outcomes."
        ),
    }
    (out_dir / "feature_holdout_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "feature_holdout_analysis_summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8"
    )

    print("Relative-mechanics feature-holdout clustering complete.")
    print(f"Output directory: {out_dir}")
    print(f"Primary model features: {len(feature_cols)}")
    print(f"Relative-mechanics model features withheld: {relative_mechanics_in_model}")
    print(f"Remaining cluster features: {len(remaining)}")
    print(f"Best k: {best_k}")
    print(f"Silhouette: {sil:.4f}")
    if ari_vs_primary is not None:
        print(f"ARI vs primary cluster solution: {ari_vs_primary:.4f}")


if __name__ == "__main__":
    main()
