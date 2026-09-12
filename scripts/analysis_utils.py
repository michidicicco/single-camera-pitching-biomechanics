#!/usr/bin/env python
"""Shared utilities for study secondary and sensitivity analyses."""
from __future__ import annotations

from collections import OrderedDict
import numpy as np
import pandas as pd

PRIMARY_MODEL_FEATURES = [
    "ffc_to_release_ms",
    "stride_length_norm_at_release",
    "stride_length_norm_max_ffc_to_release",
    "elbow_angle_deg_at_ffc",
    "elbow_angle_deg_at_release",
    "elbow_angle_deg_mean_ffc_to_release",
    "elbow_angle_deg_range_ffc_to_release",
    "elbow_ang_vel_max_ffc_to_release",
    "elbow_ang_vel_mean_ffc_to_release",
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

RELATIVE_MECHANICS_FEATURE_ALIASES = OrderedDict([
    ("max_throwing_elbow_angular_velocity", ["elbow_ang_vel_max_ffc_to_release"]),
    ("max_normalized_relative_wrist_speed", [
        "wrist_speed_norm_bh_per_s_max_ffc_to_release",
        "wrist_speed_max_ffc_to_release",
    ]),
    ("max_projected_hip_shoulder_separation_velocity", [
        "hip_shoulder_sep_vel_max_ffc_to_release",
    ]),
    ("max_trunk_tilt_velocity", ["trunk_tilt_vel_max_ffc_to_release"]),
    ("trunk_tilt_at_release", ["trunk_tilt_deg_at_release"]),
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
    """Return the fixed 32-feature primary study matrix from a model-ready table."""
    missing = [c for c in PRIMARY_MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Model-ready input is missing primary biomechanical features: "
            + ", ".join(missing)
        )
    return list(PRIMARY_MODEL_FEATURES)


def resolve_relative_mechanics_features(columns) -> dict[str, str]:
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
    return (arr - mean) / sd, mean, sd


def pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def kmeans_numpy(
    X: np.ndarray,
    k: int,
    seed: int,
    n_init: int = 50,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, float, int]:
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
            new_labels = pairwise_sq_dists(X, centers).argmin(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                members = X[new_labels == j]
                new_centers[j] = members.mean(axis=0) if len(members) else X[rng.integers(0, n)]
            if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                labels, centers = new_labels, new_centers
                break
            labels, centers = new_labels, new_centers
        inertia = float(np.min(pairwise_sq_dists(X, centers), axis=1).sum())
        if inertia < best_inertia:
            best_labels, best_centers = labels.copy(), centers.copy()
            best_inertia, best_iter = inertia, iteration
    if best_labels is None:
        raise RuntimeError("K-means failed.")
    return best_labels, best_centers, best_inertia, best_iter


def silhouette_score_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return np.nan
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    values = []
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        a = float(D[i, same].mean()) if same.sum() else 0.0
        b = min(float(D[i, labels == lab].mean()) for lab in unique if lab != labels[i])
        denom = max(a, b)
        values.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(values))


def select_k(
    Z: np.ndarray,
    max_k: int = 6,
    seed: int = 13,
    n_init: int = 50,
) -> tuple[pd.DataFrame, int]:
    rows = []
    for k in range(2, min(max_k, len(Z) - 1) + 1):
        labels, _, inertia, n_iter = kmeans_numpy(Z, k, seed, n_init=n_init, max_iter=300)
        rows.append({
            "k": k,
            "silhouette_score": silhouette_score_numpy(Z, labels),
            "inertia": inertia,
            "n_iter": n_iter,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Insufficient rows for K-means selection.")
    best_k = int(table.loc[table["silhouette_score"].astype(float).idxmax(), "k"])
    return table, best_k


def within_pitcher_zscore(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if "pitcher_id" not in df.columns:
        raise ValueError("pitcher_id is required for within-pitcher normalization.")
    out = pd.DataFrame(index=df.index, columns=feature_cols, dtype=float)
    for _, idx in df.groupby("pitcher_id").groups.items():
        block = df.loc[idx, feature_cols].apply(pd.to_numeric, errors="coerce")
        mean = block.mean(axis=0)
        sd = block.std(axis=0, ddof=0).replace(0, np.nan)
        out.loc[idx, feature_cols] = ((block - mean) / sd).fillna(0.0)
    return out.astype(float)


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    valid = np.isfinite(values)
    values, labels = values[valid], labels[valid]
    if len(values) < 3:
        return np.nan
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    if ss_total <= 0:
        return 0.0
    ss_between = sum(
        len(v) * (v.mean() - grand) ** 2
        for lab in np.unique(labels)
        if len(v := values[labels == lab])
    )
    return float(ss_between / ss_total)


def adjusted_rand_index(labels_a, labels_b) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_a, labels_b))
