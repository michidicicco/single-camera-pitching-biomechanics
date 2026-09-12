#!/usr/bin/env python
"""
merge_analysis_outputs.py

Strictly merges independently generated K-means cluster assignments with the
relative mechanical profile table.

Row-order merging is not permitted. Equal row counts do not prove that rows
represent the same pitches. Merge keys must be explicit and unique.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd


CLUSTER_CANDIDATES = ["cluster", "kmeans_cluster", "cluster_label", "kmeans_label"]


def make_unique_pitch_id(df: pd.DataFrame) -> pd.Series:
    if "unique_pitch_id" in df.columns:
        return df["unique_pitch_id"].astype(str)
    if {"pitcher_id", "pitch_id"}.issubset(df.columns):
        return df["pitcher_id"].astype(str) + "_" + df["pitch_id"].astype(str)
    if "video_file" in df.columns:
        return df["video_file"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    raise ValueError("No explicit pitch-level merge key is available.")


def detect_cluster_column(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for candidate in CLUSTER_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    for c in df.columns:
        if "cluster" in c.lower() or "kmeans" in c.lower():
            if pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) > 1:
                return c
    raise ValueError("No usable cluster-label column found.")


def validate_unique_key(df: pd.DataFrame, key: str, label: str) -> None:
    if key not in df.columns:
        raise ValueError(f"{label} lacks key: {key}")
    if df[key].isna().any():
        raise ValueError(f"{label} contains missing {key}.")
    dups = df[key].astype(str).duplicated(keep=False)
    if dups.any():
        vals = df.loc[dups, key].astype(str).head(10).tolist()
        raise ValueError(f"{label} contains duplicate {key} values; examples: {vals}")


def strict_merge(profile_df: pd.DataFrame, cluster_df: pd.DataFrame, cluster_col: str) -> Tuple[pd.DataFrame, str]:
    left = profile_df.copy()
    right = cluster_df.copy()
    left["unique_pitch_id"] = make_unique_pitch_id(left)
    try:
        right["unique_pitch_id"] = make_unique_pitch_id(right)
        validate_unique_key(left, "unique_pitch_id", "profile file")
        validate_unique_key(right, "unique_pitch_id", "cluster file")
        r = right[["unique_pitch_id", cluster_col]].rename(columns={cluster_col: "kmeans_cluster"})
        return left.merge(r, on="unique_pitch_id", how="left", validate="one_to_one"), "unique_pitch_id"
    except ValueError:
        pass

    if "video_file" in left.columns and "video_file" in right.columns:
        validate_unique_key(left, "video_file", "profile file")
        validate_unique_key(right, "video_file", "cluster file")
        r = right[["video_file", cluster_col]].rename(columns={cluster_col: "kmeans_cluster"})
        return left.merge(r, on="video_file", how="left", validate="one_to_one"), "video_file"

    raise ValueError("No safe unique merge key is shared by both files. Row-order merge is intentionally disabled.")


def find_cluster_file(cluster_dir: Path, profile_df: pd.DataFrame) -> Tuple[Path, pd.DataFrame, str]:
    candidates = []
    for path in sorted(cluster_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(path)
            col = detect_cluster_column(df)
        except Exception:
            continue
        if pd.to_numeric(df[col], errors="coerce").nunique(dropna=True) <= 1:
            continue
        score = 0
        try:
            uid_left = set(make_unique_pitch_id(profile_df))
            uid_right = set(make_unique_pitch_id(df))
            score = len(uid_left & uid_right) + 1000
        except Exception:
            if "video_file" in profile_df.columns and "video_file" in df.columns:
                score = len(set(profile_df["video_file"].astype(str)) & set(df["video_file"].astype(str))) + 100
        candidates.append((score, path, df, col))
    if not candidates:
        raise FileNotFoundError(f"No usable cluster CSV found under {cluster_dir}")
    candidates.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    score, path, df, col = candidates[0]
    if score <= 0:
        raise ValueError("Cluster CSVs were found but no explicit pitch-key overlap could be established.")
    return path, df, col


def build_summary(merged: pd.DataFrame, source: Path, method: str) -> str:
    lines: List[str] = [
        "K-MEANS CLUSTER / RELATIVE MECHANICAL PROFILE SUMMARY",
        "=" * 72,
        "",
        "The two analytical paths were generated independently and merged only by pitch identifier.",
        "Agreement is internal exploratory consistency, not external validation.",
        "",
        f"Cluster source: {source}",
        f"Merge key: {method}",
        f"Rows: {len(merged)}",
        f"Rows with cluster: {int(merged['kmeans_cluster'].notna().sum())}",
        f"Rows without cluster: {int(merged['kmeans_cluster'].isna().sum())}",
        "",
        "Cluster counts:",
        merged["kmeans_cluster"].value_counts(dropna=False).sort_index().to_string(),
        "",
    ]
    if "relative_mechanical_profile" in merged.columns:
        lines += ["Relative mechanical profile by cluster:", pd.crosstab(merged["kmeans_cluster"], merged["relative_mechanical_profile"]).to_string(), ""]
    if "pitcher_id" in merged.columns:
        lines += ["Pitcher distribution by cluster:", pd.crosstab(merged["kmeans_cluster"], merged["pitcher_id"]).to_string(), ""]
    if "measurement_qc_status" in merged.columns:
        lines += ["Measurement QC status by cluster:", pd.crosstab(merged["kmeans_cluster"], merged["measurement_qc_status"]).to_string(), ""]

    agg = {"n_pitches": ("unique_pitch_id", "count")}
    for col, name in [
        ("relative_mechanical_profile_score", "mean_relative_mechanical_profile_score"),
        ("relative_mechanical_score_pooled", "mean_pooled_relative_mechanical_score"),
        ("relative_mechanical_score_within_pitcher", "mean_within_pitcher_relative_mechanical_score"),
        ("mechanical_flag_count", "mean_mechanical_flag_count"),
    ]:
        if col in merged.columns:
            agg[name] = (col, "mean")
    summary = merged.groupby("kmeans_cluster", dropna=False).agg(**agg).reset_index()
    lines += ["Cluster-level numeric summary:", summary.to_string(index=False)]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strictly merge K-means cluster assignments with relative mechanical profiles.")
    p.add_argument("--profile-file", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--cluster-file")
    group.add_argument("--cluster-dir")
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile_file)
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    profile_df = pd.read_csv(profile_path)
    if args.cluster_file:
        source = Path(args.cluster_file)
        cluster_df = pd.read_csv(source)
        cluster_col = detect_cluster_column(cluster_df)
    else:
        source, cluster_df, cluster_col = find_cluster_file(Path(args.cluster_dir), profile_df)

    merged, method = strict_merge(profile_df, cluster_df, cluster_col)
    if merged["kmeans_cluster"].isna().all():
        raise RuntimeError("Merge completed but no cluster assignments matched.")

    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    summary_path.write_text(build_summary(merged, source, method), encoding="utf-8")

    print("Cluster/profile merge complete.")
    print(f"Source: {source}")
    print(f"Merge key: {method}")
    print(f"Matched: {int(merged['kmeans_cluster'].notna().sum())}/{len(merged)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
