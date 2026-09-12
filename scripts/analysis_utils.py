#!/usr/bin/env python
"""
Shared utilities for study secondary/sensitivity analyses.

These analyses are intentionally separate from the primary study pipeline.
They do not retune the validated event detector or replace the prespecified primary
PCA/K-means analysis.
"""
from __future__ import annotations

from collections import OrderedDict
import numpy as np
import pandas as pd


MODEL_METADATA_COLUMNS = {
    "video_file", "pitcher_id", "pitch_id", "unique_pitch_id",
    "measurement_qc_status", "model_eligible",
    "relative_mechanical_band_pooled", "relative_mechanical_band_within_pitcher",
    "relative_mechanical_score_pooled", "relative_mechanical_score_within_pitcher",
    "mechanical_flag_count", "mechanical_flag_profile",
}

# The study-defined relative mechanical score uses five biomechanical quantities. Alias lists allow the scripts to work whether
# the normalized wrist-speed field is retained under its explicit or legacy alias.
RELATIVE_MECHANICS_FEATURE_ALIASES = OrderedDict([
    ("max_throwing_elbow_angular_velocity", [
        "elbow_ang_vel_max_ffc_to_release",
    ]),
    ("max_normalized_relative_wrist_speed", [
        "wrist_speed_norm_bh_per_s_max_ffc_to_release",
        "wrist_speed_max_ffc_to_release",
    ]),
    ("max_projected_hip_shoulder_separation_velocity", [
        "hip_shoulder_sep_vel_max_ffc_to_release",
    ]),
    ("max_trunk_tilt_velocity", [
        "trunk_tilt_vel_max_ffc_to_release",
    ]),
    ("trunk_tilt_at_release", [
        "trunk_tilt_deg_at_release",
    ]),
])


def build_unique_pitch_id(df: pd.DataFrame) -> pd.Series:
    if "unique_pitch_id" in df.columns:
        return df["unique_pitch_id"].astype(str)
    if {"pitcher_id", "pitch_id"}.issubset(df.columns):
        return df["pitcher_id"].astype(str) + "_" + df["pitch_id"].astype(str)
    if "video_file" in df.columns:
        return df["video_file"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    raise ValueError("Cannot construct unique_pitch_id from supplied data.")


def ensure_unique_pitch_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["unique_pitch_id"] = build_unique_pitch_id(out)
    if out["unique_pitch_id"].duplicated().any():
        dup = out.loc[out["unique_pitch_id"].duplicated(keep=False), "unique_pitch_id"].tolist()
        raise ValueError(f"unique_pitch_id is not unique. Examples: {dup[:10]}")
    return out


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric model features from the canonical model-ready CSV."""
    cols = []
    for col in df.columns:
        if col in MODEL_METADATA_COLUMNS:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            cols.append(col)
    return cols


def resolve_relative_mechanics_features(columns) -> dict[str, str]:
    """
    Resolve one actual data column for each feature used in the relative mechanical score.
    Returns logical_name -> actual_column for features found.
    """
    cols = set(columns)
    resolved = {}
    for logical, aliases in RELATIVE_MECHANICS_FEATURE_ALIASES.items():
        for alias in aliases:
            if alias in cols:
                resolved[logical] = alias
                break
    return resolved


def standardize_matrix(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = X.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=0)
    sd = np.where((~np.isfinite(sd)) | (sd == 0), 1.0, sd)
    Z = (arr - mean) / sd
    return Z, mean, sd


def pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def kmeans_numpy(
    X: np.ndarray,
    k: int,
    seed: int,
    n_init: int = 50,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Deterministic NumPy K-means matching the study cleaning script logic."""
    rng_master = np.random.default_rng(seed)
    n = X.shape[0]
    if k < 2 or k > n:
        raise ValueError(f"k must be between 2 and n. Got k={k}, n={n}.")

    best_labels = best_centers = None
    best_inertia = np.inf
    best_iter = 0

    for _ in range(n_init):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        labels = np.full(n, -1, dtype=int)

        for iteration in range(1, max_iter + 1):
            d = pairwise_sq_dists(X, centers)
            new_labels = d.argmin(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                members = X[new_labels == j]
                if len(members) == 0:
                    new_centers[j] = X[rng.integers(0, n)]
                else:
                    new_centers[j] = members.mean(axis=0)
            if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                labels = new_labels
                centers = new_centers
                break
            labels, centers = new_labels, new_centers

        inertia = float(np.min(pairwise_sq_dists(X, centers), axis=1).sum())
        if inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
            best_iter = iteration

    if best_labels is None:
        raise RuntimeError("K-means failed.")
    return best_labels, best_centers, best_inertia, best_iter


def silhouette_score_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return np.nan

    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    vals = []
    for i in range(len(X)):
        own = labels[i]
        same = labels == own
        same[i] = False
        a = float(D[i, same].mean()) if same.sum() else 0.0
        other_means = [
            float(D[i, labels == lab].mean())
            for lab in unique if lab != own and (labels == lab).sum()
        ]
        b = min(other_means) if other_means else 0.0
        denom = max(a, b)
        vals.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(vals))


def select_k(
    Z: np.ndarray,
    max_k: int = 6,
    seed: int = 13,
    n_init: int = 50,
) -> tuple[pd.DataFrame, int]:
    rows = []
    upper = min(max_k, len(Z) - 1)
    for k in range(2, upper + 1):
        labels, centers, inertia, n_iter = kmeans_numpy(
            Z, k=k, seed=seed, n_init=n_init, max_iter=300
        )
        rows.append({
            "k": k,
            "silhouette_score": silhouette_score_numpy(Z, labels),
            "inertia": inertia,
            "n_iter": n_iter,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Insufficient rows for K-means selection.")
    best = table.loc[table["silhouette_score"].astype(float).idxmax()]
    return table, int(best["k"])


def within_pitcher_zscore(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Z-score each feature within pitcher. If a feature has zero/undefined SD for one
    pitcher, that pitcher's values for that feature are set to 0 (no within-pitcher
    deviation on that dimension).
    """
    if "pitcher_id" not in df.columns:
        raise ValueError("pitcher_id is required for within-pitcher normalization.")
    out = pd.DataFrame(index=df.index, columns=feature_cols, dtype=float)
    for pitcher, idx in df.groupby("pitcher_id").groups.items():
        block = df.loc[idx, feature_cols].apply(pd.to_numeric, errors="coerce")
        mean = block.mean(axis=0)
        sd = block.std(axis=0, ddof=0).replace(0, np.nan)
        z = (block - mean) / sd
        out.loc[idx, feature_cols] = z.fillna(0.0)
    return out.astype(float)


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """One-way between-cluster eta-squared; used as an effect-size statistic."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    valid = np.isfinite(values)
    values = values[valid]
    labels = labels[valid]
    if len(values) < 3:
        return np.nan
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for lab in np.unique(labels):
        v = values[labels == lab]
        if len(v):
            ss_between += len(v) * (v.mean() - grand) ** 2
    return float(ss_between / ss_total)


def adjusted_rand_index(labels_a, labels_b) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for ARI calculation.") from exc
    return float(adjusted_rand_score(labels_a, labels_b))
