#!/usr/bin/env python
"""
Secondary null-model and bootstrap analysis for the primary clustering.

Null test:
  Independently shuffles each model feature WITHIN pitcher, preserving each
  pitcher's univariate feature distributions while disrupting multivariable
  coordination. The observed fixed-k silhouette is compared with the null.

Bootstrap:
  Resamples pitches with replacement WITHIN pitcher, preserving the participant
  composition, and estimates a percentile interval for the fixed-k silhouette.

By default k=5 is evaluated to match the primary study solution.
Use --reselect-k only as an additional sensitivity analysis; it is slower.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from analysis_utils import (
    ensure_unique_pitch_id,
    kmeans_numpy,
    model_feature_columns,
    select_k,
    silhouette_score_numpy,
    standardize_matrix,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ready", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--n-init", type=int, default=50,
                   help="Number of K-means initializations per resample.")
    p.add_argument("--reselect-k", action="store_true",
                   help="Re-select k=2..max-k inside every resample; substantially slower.")
    return p.parse_args()


def fit_stat(X, k, seed, n_init, reselect_k, max_k):
    Z, _, _ = standardize_matrix(X)
    chosen_k = k
    if reselect_k:
        _, chosen_k = select_k(Z, max_k=max_k, seed=seed, n_init=n_init)
    labels, centers, inertia, n_iter = kmeans_numpy(
        Z, k=chosen_k, seed=seed, n_init=n_init, max_iter=300
    )
    return chosen_k, silhouette_score_numpy(Z, labels)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_unique_pitch_id(pd.read_csv(args.model_ready))
    if "pitcher_id" not in df.columns:
        raise ValueError("pitcher_id is required.")

    features = model_feature_columns(df)
    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Model-ready input contains missing model-feature values.")

    observed_k, observed_sil = fit_stat(
        X, args.k, args.seed, max(args.n_init, 50), args.reselect_k, args.max_k
    )

    rng = np.random.default_rng(args.seed)
    groups = {p: np.asarray(idx) for p, idx in df.groupby("pitcher_id").groups.items()}

    perm_rows = []
    for b in range(args.permutations):
        Xp = X.copy()
        for col in features:
            for p, idx in groups.items():
                vals = Xp.loc[idx, col].to_numpy(copy=True)
                Xp.loc[idx, col] = rng.permutation(vals)
        k_b, sil_b = fit_stat(
            Xp, args.k, args.seed + 1000 + b, args.n_init,
            args.reselect_k, args.max_k
        )
        perm_rows.append({"iteration": b + 1, "k": k_b, "silhouette": sil_b})

    perm = pd.DataFrame(perm_rows)
    perm.to_csv(out_dir / "cluster_null_permutation_results.csv", index=False)
    null_p = (1 + (perm["silhouette"] >= observed_sil - 1e-15).sum()) / (len(perm) + 1)

    boot_rows = []
    for b in range(args.bootstraps):
        sampled_idx = []
        for p, idx in groups.items():
            sampled_idx.extend(rng.choice(idx, size=len(idx), replace=True).tolist())
        Xb = X.loc[sampled_idx].reset_index(drop=True)
        k_b, sil_b = fit_stat(
            Xb, args.k, args.seed + 100000 + b, args.n_init,
            args.reselect_k, args.max_k
        )
        boot_rows.append({"iteration": b + 1, "k": k_b, "silhouette": sil_b})

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out_dir / "cluster_pitcher_stratified_bootstrap_results.csv", index=False)

    ci_low, ci_high = np.nanpercentile(boot["silhouette"], [2.5, 97.5])

    summary = {
        "analysis": "secondary cluster null-model and pitcher-stratified bootstrap",
        "n_pitches": int(len(df)),
        "n_pitchers": int(df["pitcher_id"].nunique()),
        "model_features": int(len(features)),
        "observed_k": int(observed_k),
        "observed_silhouette_recomputed": float(observed_sil),
        "null_permutations": int(args.permutations),
        "null_mean_silhouette": float(perm["silhouette"].mean()),
        "null_95th_percentile_silhouette": float(np.nanpercentile(perm["silhouette"], 95)),
        "null_empirical_p": float(null_p),
        "bootstrap_resamples": int(args.bootstraps),
        "bootstrap_mean_silhouette": float(boot["silhouette"].mean()),
        "bootstrap_95pct_percentile_interval": [float(ci_low), float(ci_high)],
        "reselect_k_in_each_resample": bool(args.reselect_k),
        "n_init_per_resample": int(args.n_init),
        "interpretation": (
            "The null preserves within-pitcher univariate feature distributions while "
            "disrupting multivariable coordination. This tests whether the observed "
            "geometric structure exceeds that expected after destroying feature "
            "relationships; it does not establish clinical outcomes."
        ),
    }
    (out_dir / "cluster_null_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "cluster_null_bootstrap_summary.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8"
    )

    print("Cluster null/bootstrap analysis complete.")
    print(f"Output directory: {out_dir}")
    print(f"Observed k: {observed_k}")
    print(f"Observed silhouette (recomputed): {observed_sil:.4f}")
    print(f"Null empirical p: {null_p:.6f}")
    print(f"Bootstrap 95% silhouette interval: [{ci_low:.4f}, {ci_high:.4f}]")


if __name__ == "__main__":
    main()
