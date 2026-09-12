#!/usr/bin/env python
"""
Pitcher-adjusted biomechanical relevance of the primary clusters.

For each relative-mechanics variable, values are standardized within pitcher.
A stratified permutation test then shuffles cluster labels WITHIN pitcher and
tests the observed between-cluster eta-squared. This asks whether cluster-related
mechanical differences remain after removing each pitcher's baseline level.

This is a secondary analysis and does not establish clinical outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from analysis_utils import (
    ensure_unique_pitch_id,
    eta_squared,
    resolve_relative_mechanics_features,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Merged profiles-and-clusters CSV.")
    p.add_argument("--cluster-col", default="kmeans_cluster")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--permutations", type=int, default=5000)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def within_pitcher_standardize(series: pd.Series, pitcher: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    temp = pd.DataFrame({"value": pd.to_numeric(series, errors="coerce"), "pitcher": pitcher})
    for _, idx in temp.groupby("pitcher").groups.items():
        v = temp.loc[idx, "value"]
        mean = v.mean()
        sd = v.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (v - mean) / sd
    return out


def stratified_permutation_p(values, labels, pitcher, observed, n_perm, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels).copy()
    pitcher = np.asarray(pitcher)
    stats = np.empty(n_perm, dtype=float)
    unique_pitchers = pd.unique(pitcher)

    for b in range(n_perm):
        perm = labels.copy()
        for p in unique_pitchers:
            idx = np.flatnonzero(pitcher == p)
            perm[idx] = rng.permutation(perm[idx])
        stats[b] = eta_squared(values, perm)

    pval = (1 + np.sum(stats >= observed - 1e-15)) / (n_perm + 1)
    return float(pval), stats


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_unique_pitch_id(pd.read_csv(args.input))
    if "pitcher_id" not in df.columns:
        raise ValueError("pitcher_id is required.")
    if args.cluster_col not in df.columns:
        raise ValueError(f"Missing cluster column: {args.cluster_col}")

    resolved = resolve_relative_mechanics_features(df.columns)
    if not resolved:
        raise ValueError("No relative-mechanics features were found.")

    test_rows = []
    cluster_rows = []

    analysis_vars = [(logical, col) for logical, col in resolved.items()]

    if "relative_mechanical_profile" in df.columns:
        df["_higher_profile_indicator"] = (
            df["relative_mechanical_profile"].astype(str)
            == "higher"
        ).astype(float)
        analysis_vars.append(("higher_profile_indicator", "_higher_profile_indicator"))

    for i, (logical, col) in enumerate(analysis_vars):
        y = within_pitcher_standardize(df[col], df["pitcher_id"])
        valid = y.notna() & df[args.cluster_col].notna() & df["pitcher_id"].notna()

        vals = y.loc[valid].to_numpy(dtype=float)
        labels = df.loc[valid, args.cluster_col].to_numpy()
        pitcher = df.loc[valid, "pitcher_id"].to_numpy()

        obs = eta_squared(vals, labels)
        pval, perm_stats = stratified_permutation_p(
            vals, labels, pitcher, obs, args.permutations, args.seed + i
        )

        test_rows.append({
            "outcome": logical,
            "source_column": col,
            "n": int(valid.sum()),
            "pitcher_adjusted_eta_squared": obs,
            "within_pitcher_stratified_permutation_p": pval,
            "permutations": args.permutations,
            "null_eta_squared_mean": float(np.nanmean(perm_stats)),
            "null_eta_squared_95th_percentile": float(np.nanpercentile(perm_stats, 95)),
        })

        temp = pd.DataFrame({
            "cluster": labels,
            "within_pitcher_z": vals,
        })
        for cluster, block in temp.groupby("cluster"):
            cluster_rows.append({
                "outcome": logical,
                "source_column": col,
                "cluster": cluster,
                "n": len(block),
                "mean_within_pitcher_z": block["within_pitcher_z"].mean(),
                "median_within_pitcher_z": block["within_pitcher_z"].median(),
                "sd_within_pitcher_z": block["within_pitcher_z"].std(ddof=1)
                    if len(block) > 1 else np.nan,
            })

    tests = pd.DataFrame(test_rows)
    clusters = pd.DataFrame(cluster_rows)
    tests.to_csv(out_dir / "pitcher_adjusted_cluster_relevance_tests.csv", index=False)
    clusters.to_csv(out_dir / "pitcher_adjusted_cluster_means.csv", index=False)

    pd.crosstab(df["pitcher_id"], df[args.cluster_col]).to_csv(
        out_dir / "pitcher_by_primary_cluster_counts.csv"
    )

    if "relative_mechanical_profile" in df.columns:
        pd.crosstab(
            [df["pitcher_id"], df[args.cluster_col]], df["relative_mechanical_profile"]
        ).to_csv(out_dir / "pitcher_cluster_by_profile_counts.csv")

    summary = {
        "analysis": "pitcher-adjusted cluster biomechanical relevance",
        "n_pitches": int(len(df)),
        "n_pitchers": int(df["pitcher_id"].nunique()),
        "cluster_column": args.cluster_col,
        "permutations_per_outcome": int(args.permutations),
        "relative_mechanics_features_evaluated": resolved,
        "interpretation": (
            "Values were standardized within pitcher and cluster labels were permuted "
            "within pitcher. Small permutation p-values therefore indicate cluster-related "
            "differences beyond each pitcher's baseline distribution; they do not validate "
            "clinical outcomes."
        ),
    }
    (out_dir / "pitcher_adjusted_cluster_relevance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Pitcher-adjusted cluster relevance analysis complete.")
    print(f"Output directory: {out_dir}")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
