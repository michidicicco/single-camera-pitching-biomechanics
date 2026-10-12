#!/usr/bin/env python
"""Cross-algorithm clustering comparison (secondary thesis sensitivity analysis).

Runs the archived repository's NumPy K-means implementation and Ward hierarchical
clustering on the SAME standardized, explicitly selected biomechanical features.
Writes AGGREGATE summaries only (no participant or pitch identifiers).
Never overwrites or modifies the original primary analysis outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from prepare_analysis_data import kmeans


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="outputs/analysis/model_ready.csv")
    p.add_argument("--features-file", default="outputs/analysis/retained_features.csv",
                   help="CSV with one 'feature' column. Prevents metadata leakage.")
    p.add_argument("--output-dir", default="outputs/hierarchical_comparison")
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--balanced-resamples", type=int, default=100)
    p.add_argument("--pitches-per-pitcher", type=int, default=0,
                   help="Default: minimum pitch count across pitchers.")
    return p.parse_args()


def metrics(z, labels):
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(z):
        raise ValueError("Invalid cluster count for separation metrics")
    return dict(
        silhouette=float(silhouette_score(z, labels)),
        davies_bouldin=float(davies_bouldin_score(z, labels)),
        calinski_harabasz=float(calinski_harabasz_score(z, labels)),
    )


def fit_both(z, k, seed, n_init=50, max_iter=300):
    km_labels, _, _, _ = kmeans(z, k, seed, n_init=n_init, max_iter=max_iter)
    ward_labels = fcluster(linkage(z, method="ward", metric="euclidean"), t=k, criterion="maxclust")
    return km_labels, ward_labels


def score_models(z, k, seed):
    km, ward = fit_both(z, k, seed)
    return km, ward, metrics(z, km), metrics(z, ward)


def check_and_load(input_path, features_path):
    df = pd.read_csv(input_path)
    fdf = pd.read_csv(features_path)
    if "feature" not in fdf.columns:
        raise ValueError("features-file must have a 'feature' column")
    features = fdf["feature"].dropna().astype(str).tolist()
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature list is empty or contains duplicate names")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError("Missing retained features: " + ", ".join(missing))
    if "pitcher_id" not in df.columns:
        raise ValueError("pitcher_id required for participant-aware robustness")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature matrix must be complete and finite")
    if X.shape[0] <= 7 or X.shape[1] < 2:
        raise ValueError("Need >7 pitches and >=2 features")
    pitchers = df["pitcher_id"].astype(str).to_numpy()
    if pd.isna(df["pitcher_id"]).any() or len(np.unique(pitchers)) < 2:
        raise ValueError("Need valid IDs for at least two pitchers")
    return X.to_numpy(float), pitchers, features


def main():
    a = cli()
    if not (2 <= a.min_k <= a.max_k):
        raise ValueError("Require 2 <= min-k <= max-k")
    if a.balanced_resamples < 0:
        raise ValueError("balanced-resamples must be nonnegative")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, pitchers, features = check_and_load(a.input, a.features_file)
    z = StandardScaler().fit_transform(X)
    if a.max_k >= len(z):
        raise ValueError("max-k must be less than the number of pitches")

    selection_rows = []
    baseline = {}
    for k in range(a.min_k, a.max_k + 1):
        km, ward, mk, mw = score_models(z, k, a.seed)
        baseline[k] = (km, ward)
        selection_rows.extend([
            {"method": "kmeans", "k": k, **mk},
            {"method": "ward", "k": k, **mw},
        ])
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(out / "model_selection.csv", index=False)

    # Model-selection silhouettes are compared under identical features and scaling.
    best = {
        method: int(g.loc[g.silhouette.idxmax(), "k"])
        for method, g in selection.groupby("method")
    }
    agreement = []
    for k in range(a.min_k, a.max_k + 1):
        km, ward = baseline[k]
        agreement.append({"k": k, "ARI_kmeans_vs_ward": float(adjusted_rand_score(km, ward))})
    pd.DataFrame(agreement).to_csv(out / "cross_algorithm_agreement.csv", index=False)

    # Use matching fixed-k assignments for fair cross-method comparison.
    # Primary k=5 stays a separate reported analysis, not overwritten here.
    stability = []
    unique_pitchers = sorted(set(pitchers))
    for pid in unique_pitchers:
        keep = pitchers != pid
        zs = StandardScaler().fit_transform(X[keep])
        for k in range(a.min_k, a.max_k + 1):
            if len(zs) <= k:
                continue
            sub_km, sub_ward = fit_both(zs, k, a.seed)
            original_km, original_ward = baseline[k]
            stability.extend([
                {"analysis": "leave_one_pitcher_out", "method": "kmeans", "k": k,
                 "ARI": float(adjusted_rand_score(original_km[keep], sub_km)),
                 "silhouette": float(silhouette_score(zs, sub_km))},
                {"analysis": "leave_one_pitcher_out", "method": "ward", "k": k,
                 "ARI": float(adjusted_rand_score(original_ward[keep], sub_ward)),
                 "silhouette": float(silhouette_score(zs, sub_ward))},
            ])

    groups = {pid: np.flatnonzero(pitchers == pid) for pid in unique_pitchers}
    n_per = a.pitches_per_pitcher or min(len(v) for v in groups.values())
    if n_per < 1 or any(len(v) < n_per for v in groups.values()):
        raise ValueError("pitches-per-pitcher exceeds an available pitcher count")
    rng = np.random.default_rng(a.seed)
    for rep in range(a.balanced_resamples):
        ix = np.sort(np.concatenate([
            rng.choice(groups[pid], size=n_per, replace=False) for pid in unique_pitchers
        ]))
        zs = StandardScaler().fit_transform(X[ix])
        for k in range(a.min_k, min(a.max_k, len(ix)-1) + 1):
            sub_km, sub_ward = fit_both(zs, k, a.seed + rep)
            original_km, original_ward = baseline[k]
            stability.extend([
                {"analysis": "pitcher_balanced", "method": "kmeans", "k": k,
                 "ARI": float(adjusted_rand_score(original_km[ix], sub_km)),
                 "silhouette": float(silhouette_score(zs, sub_km))},
                {"analysis": "pitcher_balanced", "method": "ward", "k": k,
                 "ARI": float(adjusted_rand_score(original_ward[ix], sub_ward)),
                 "silhouette": float(silhouette_score(zs, sub_ward))},
            ])
    stable = pd.DataFrame(stability)
    if len(stable):
        stable.groupby(["analysis", "method", "k"], as_index=False).agg(
            n_replicates=("ARI", "size"),
            mean_ARI=("ARI", "mean"),
            sd_ARI=("ARI", "std"),
            mean_silhouette=("silhouette", "mean"),
        ).to_csv(out / "robustness_summary.csv", index=False)

    summary = {
        "data_input": str(a.input), "feature_list": str(a.features_file),
        "n_pitches": int(len(X)), "n_pitchers": len(unique_pitchers),
        "n_features": len(features),
        "methods": ["archived_custom_kmeans", "ward_euclidean"],
        "preferred_k_by_silhouette": best,
        "reference_k5_ARI_kmeans_vs_ward": (
            float(adjusted_rand_score(*baseline[5])) if 5 in baseline else None
        ),
        "selection_kmeans_n_init": 50,
        "interpretation": "Exploratory algorithm comparison; not validation of universal pitching types.",
        "privacy": "Aggregate outputs only; no pitch-level or participant-level assignments exported.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Compact comparison plot; no participant identifiers.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for method, sub in selection.groupby("method"):
        plt.plot(sub.k, sub.silhouette, marker="o", label=method)
    plt.xticks(range(a.min_k, a.max_k+1))
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette coefficient")
    plt.title("K-means vs Ward: model selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "silhouette_comparison.png", dpi=200)
    plt.close()
    print(json.dumps(summary, indent=2))
    print("Aggregate results written to:", out)


if __name__ == "__main__":
    main()
