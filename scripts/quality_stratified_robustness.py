#!/usr/bin/env python
"""Quality-stratified robustness analysis for monocular pitching representations.

Starting from a QC-annotated pitch-level feature table, this script evaluates
whether progressively stricter pose/measurement-quality subsets materially
change the downstream PCA/K-means representation.

Default strata:
1. all_analyzable: all non-excluded observations,
2. qc_pass: observations classified as measurement-QC pass,
3. high_confidence: a configurable stricter subset requiring stronger pose,
   landmark-visibility, event-confidence, and missing-gap criteria.

For every stratum the standard deterministic model-matrix construction,
descriptive PCA, and K-means model selection are rerun. The script reports
sample size, retained feature count, selected k, silhouette, PCA variance,
Procrustes similarity of PCA score spaces, and adjusted Rand index (ARI)
relative to the all-analyzable reference on shared pitches.

The high-confidence thresholds are analysis choices for robustness assessment;
they are not clinical thresholds and do not redefine the validated study QC.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn.metrics import adjusted_rand_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PREP_SCRIPT = SCRIPT_DIR / "prepare_analysis_data.py"

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "features" / "pitch_features_qc_annotated.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "quality_stratified_robustness"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate downstream representation stability across measurement-quality strata."
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--random-state", type=int, default=13)

    p.add_argument("--high-min-pose-rate", type=float, default=0.95)
    p.add_argument("--high-min-critical-visibility", type=float, default=0.75)
    p.add_argument("--high-min-event-confidence", type=float, default=0.70)
    p.add_argument("--high-max-critical-gap-ms", type=float, default=50.0)
    p.add_argument("--high-min-event-window-ms", type=float, default=50.0)
    p.add_argument("--high-max-event-window-ms", type=float, default=500.0)
    p.add_argument(
        "--allow-ffc-fallback-high-confidence",
        action="store_true",
        help="Allow FFC fallback detections in the high-confidence subset.",
    )
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, shell=False)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def numeric(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        return num.fillna(0).astype(float) > 0
    txt = series.astype(str).str.strip().str.lower()
    return txt.isin(["1", "true", "yes", "y", "ok", "pass"])


def analyzable_mask(df: pd.DataFrame) -> pd.Series:
    if "measurement_qc_status" in df.columns:
        return ~df["measurement_qc_status"].astype(str).str.lower().eq("exclude")
    if "model_eligible" in df.columns:
        return boolish(df["model_eligible"])
    return pd.Series(True, index=df.index)


def qc_pass_mask(df: pd.DataFrame) -> pd.Series:
    if "measurement_qc_status" not in df.columns:
        raise ValueError("measurement_qc_status is required for the qc_pass stratum.")
    return df["measurement_qc_status"].astype(str).str.lower().eq("pass")


def high_confidence_mask(df: pd.DataFrame, a: argparse.Namespace) -> pd.Series:
    base = analyzable_mask(df)

    required = [
        "pose_detected_rate",
        "critical_visibility_min",
        "ffc_confidence",
        "release_confidence",
        "critical_landmark_longest_raw_gap_ms",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "High-confidence stratification requires QC metrics missing from the input: "
            + ", ".join(missing)
        )

    event_ms = numeric(df, "event_window_ms")
    if event_ms.isna().all():
        event_ms = numeric(df, "ffc_to_release_ms")

    mask = (
        base
        & (numeric(df, "pose_detected_rate") >= a.high_min_pose_rate)
        & (numeric(df, "critical_visibility_min") >= a.high_min_critical_visibility)
        & (numeric(df, "ffc_confidence") >= a.high_min_event_confidence)
        & (numeric(df, "release_confidence") >= a.high_min_event_confidence)
        & (numeric(df, "critical_landmark_longest_raw_gap_ms") <= a.high_max_critical_gap_ms)
        & (event_ms >= a.high_min_event_window_ms)
        & (event_ms <= a.high_max_event_window_ms)
    )

    if not a.allow_ffc_fallback_high_confidence:
        if "ffc_detection_method" in df.columns:
            mask &= ~df["ffc_detection_method"].astype(str).str.contains(
                "fallback", case=False, na=False
            )
        elif "qc_watch_ffc_fallback" in df.columns:
            mask &= ~boolish(df["qc_watch_ffc_fallback"])

    return mask.fillna(False)


def prepare_stratum(
    name: str,
    df: pd.DataFrame,
    mask: pd.Series,
    out_root: Path,
    max_k: int,
    random_state: int,
) -> dict[str, Path]:
    stratum_dir = out_root / name
    analysis_dir = stratum_dir / "analysis"
    stratum_dir.mkdir(parents=True, exist_ok=True)

    subset = df.loc[mask].copy()
    if len(subset) < 4:
        raise ValueError(f"Stratum {name!r} has too few rows for clustering: n={len(subset)}")

    # The subset itself defines eligibility for this sensitivity run.
    subset["model_eligible"] = 1
    subset_csv = stratum_dir / "input_subset.csv"
    subset.to_csv(subset_csv, index=False)

    run([
        sys.executable,
        str(PREP_SCRIPT),
        "--input",
        str(subset_csv),
        "--output-dir",
        str(analysis_dir),
        "--max-k",
        str(max_k),
        "--random-state",
        str(random_state),
    ])

    return {"subset": subset_csv, "analysis": analysis_dir}


def indexed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "unique_pitch_id" not in df.columns:
        raise ValueError(f"{path} is missing unique_pitch_id.")
    if df["unique_pitch_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate unique_pitch_id values.")
    return df.set_index("unique_pitch_id", drop=False)


def pca_similarity(reference_analysis: Path, test_analysis: Path, n_components: int = 5) -> dict:
    ref = indexed(reference_analysis / "pca_scores.csv")
    tst = indexed(test_analysis / "pca_scores.csv")
    common = ref.index.intersection(tst.index)
    pcs = [f"PC{i}" for i in range(1, n_components + 1)]
    pcs = [c for c in pcs if c in ref.columns and c in tst.columns]
    if len(common) < 3 or len(pcs) < 2:
        return {
            "pca_n_common": int(len(common)),
            "pca_n_components": len(pcs),
            "pca_procrustes_disparity": np.nan,
        }
    a = ref.loc[common, pcs].to_numpy(float)
    b = tst.loc[common, pcs].to_numpy(float)
    _, _, disparity = procrustes(a, b)
    return {
        "pca_n_common": int(len(common)),
        "pca_n_components": len(pcs),
        "pca_procrustes_disparity": float(disparity),
    }


def cluster_similarity(reference_analysis: Path, test_analysis: Path) -> dict:
    ref = indexed(reference_analysis / "cluster_assignments.csv")
    tst = indexed(test_analysis / "cluster_assignments.csv")
    common = ref.index.intersection(tst.index)
    ari = (
        float(adjusted_rand_score(ref.loc[common, "cluster"], tst.loc[common, "cluster"]))
        if len(common) >= 2
        else np.nan
    )
    return {"cluster_n_common": int(len(common)), "cluster_ARI_vs_reference": ari}


def stratum_summary(name: str, analysis_dir: Path) -> dict:
    summary = json.loads((analysis_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    pca = pd.read_csv(analysis_dir / "pca_explained_variance.csv")
    out = {
        "stratum": name,
        "n_model_rows": int(summary["complete_case_model_rows"]),
        "n_retained_features": int(summary["retained_model_features"]),
        "best_k": int(summary["best_k"]),
        "best_silhouette": float(summary["best_silhouette"]),
    }
    for _, row in pca.head(5).iterrows():
        out[f"{row['PC']}_explained_variance_ratio"] = float(row["explained_variance_ratio"])
    return out


def retained_feature_overlap(reference_analysis: Path, test_analysis: Path) -> dict:
    ref = set(pd.read_csv(reference_analysis / "retained_features.csv")["feature"].astype(str))
    tst = set(pd.read_csv(test_analysis / "retained_features.csv")["feature"].astype(str))
    inter = ref & tst
    union = ref | tst
    return {
        "retained_features_reference": len(ref),
        "retained_features_test": len(tst),
        "retained_features_shared": len(inter),
        "retained_feature_jaccard": len(inter) / len(union) if union else np.nan,
    }


def main() -> None:
    a = parse_args()
    if not a.input.exists():
        raise FileNotFoundError(f"QC-annotated input not found: {a.input}")

    a.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.input)

    masks = {
        "all_analyzable": analyzable_mask(df),
        "qc_pass": qc_pass_mask(df),
        "high_confidence": high_confidence_mask(df, a),
    }

    paths = {}
    summaries = []
    counts = []
    for name, mask in masks.items():
        n = int(mask.sum())
        counts.append({"stratum": name, "n_input_rows": n})
        paths[name] = prepare_stratum(
            name, df, mask, a.output_dir, a.max_k, a.random_state
        )
        summaries.append(stratum_summary(name, paths[name]["analysis"]))

    reference = paths["all_analyzable"]["analysis"]
    comparison_rows = []
    for name in masks:
        analysis = paths[name]["analysis"]
        comparison_rows.append({
            "reference_stratum": "all_analyzable",
            "test_stratum": name,
            **retained_feature_overlap(reference, analysis),
            **pca_similarity(reference, analysis),
            **cluster_similarity(reference, analysis),
        })

    pd.DataFrame(counts).to_csv(a.output_dir / "stratum_counts.csv", index=False)
    pd.DataFrame(summaries).to_csv(a.output_dir / "stratum_summary.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(
        a.output_dir / "representation_agreement_vs_all_analyzable.csv", index=False
    )

    manifest = {
        "analysis": "quality_stratified_representation_robustness",
        "reference_stratum": "all_analyzable",
        "strata": list(masks),
        "high_confidence_thresholds": {
            "minimum_pose_detection_rate": a.high_min_pose_rate,
            "minimum_critical_landmark_visibility": a.high_min_critical_visibility,
            "minimum_ffc_and_release_confidence": a.high_min_event_confidence,
            "maximum_critical_raw_gap_ms": a.high_max_critical_gap_ms,
            "event_window_ms": [
                a.high_min_event_window_ms,
                a.high_max_event_window_ms,
            ],
            "allow_ffc_fallback": bool(a.allow_ffc_fallback_high_confidence),
        },
        "interpretation": (
            "These strata are used to evaluate measurement-quality sensitivity. "
            "The high-confidence thresholds are investigator-defined robustness criteria, "
            "not clinical cutoffs and not replacements for the validated study QC."
        ),
    }
    (a.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nQuality-stratified robustness analysis complete.")
    print(f"Counts: {a.output_dir / 'stratum_counts.csv'}")
    print(f"Summary: {a.output_dir / 'stratum_summary.csv'}")
    print(
        "Agreement: "
        f"{a.output_dir / 'representation_agreement_vs_all_analyzable.csv'}"
    )


if __name__ == "__main__":
    main()
