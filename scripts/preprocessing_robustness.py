#!/usr/bin/env python
"""Sensitivity analysis for kinematic smoothing in the monocular-video pipeline.

This script reuses existing frame-level pose CSVs and reruns downstream feature
extraction under a set of kinematic Savitzky-Golay smoothing windows while
holding the validated event-localization settings fixed. It then reruns QC and
the primary PCA/K-means workflow for each condition and summarizes:

- feature-by-feature agreement with a reference smoothing condition,
- PCA explained-variance structure,
- Procrustes similarity of the leading PCA score space,
- selected cluster count and silhouette coefficient,
- adjusted Rand index (ARI) relative to the reference partition.

The intent is to quantify preprocessing sensitivity, not to retune the
validated study pipeline. The validated event-localization settings remain
fixed unless the called scripts themselves are intentionally modified.
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
FEATURE_SCRIPT = SCRIPT_DIR / "extract_features.py"
QC_SCRIPT = SCRIPT_DIR / "quality_control.py"
PREP_SCRIPT = SCRIPT_DIR / "prepare_analysis_data.py"

DEFAULT_POSE_DIR = PROJECT_ROOT / "outputs" / "pose"
DEFAULT_METADATA = PROJECT_ROOT / "metadata" / "pitch_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessing_robustness"

EVENT_DEFAULTS = {
    "max_interp_gap_ms": 100.0,
    "event_savgol_window_ms": 100.0,
    "ffc_event_savgol_window_ms": 50.0,
    "ffc_lookback_ms": 600.0,
    "ffc_stability_hold_ms": 50.0,
    "ffc_precontact_window_ms": 100.0,
    "ffc_ground_tolerance_foot_length": 0.15,
    "release_search_start_frac": 0.45,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate downstream representation robustness across kinematic smoothing windows."
    )
    p.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--windows-ms",
        type=float,
        nargs="+",
        default=[50.0, 100.0, 200.0],
        help="Kinematic Savitzky-Golay windows to evaluate.",
    )
    p.add_argument(
        "--reference-window-ms",
        type=float,
        default=200.0,
        help="Reference condition used for pairwise agreement summaries.",
    )
    p.add_argument("--max-k", type=int, default=6)
    p.add_argument("--random-state", type=int, default=13)
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


def label(window_ms: float) -> str:
    return f"{window_ms:g}ms".replace(".", "p")


def build_condition(
    window_ms: float,
    pose_dir: Path,
    metadata: Path,
    out_root: Path,
    max_k: int,
    random_state: int,
) -> dict[str, Path]:
    condition = out_root / label(window_ms)
    features_dir = condition / "features"
    analysis_dir = condition / "analysis"
    logs_dir = condition / "logs"
    for p in (features_dir, analysis_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)

    features_csv = features_dir / "pitch_features_master.csv"
    qc_csv = features_dir / "pitch_features_qc_annotated.csv"
    qc_report = logs_dir / "features_qc_report.txt"
    qc_flow = logs_dir / "features_qc_flow.csv"

    feature_cmd = [
        sys.executable,
        str(FEATURE_SCRIPT),
        "--pose-dir",
        str(pose_dir),
        "--metadata",
        str(metadata),
        "--out",
        str(features_csv),
        "--max-interp-gap-ms",
        str(EVENT_DEFAULTS["max_interp_gap_ms"]),
        "--savgol-window-ms",
        str(window_ms),
        "--event-savgol-window-ms",
        str(EVENT_DEFAULTS["event_savgol_window_ms"]),
        "--ffc-event-savgol-window-ms",
        str(EVENT_DEFAULTS["ffc_event_savgol_window_ms"]),
        "--ffc-lookback-ms",
        str(EVENT_DEFAULTS["ffc_lookback_ms"]),
        "--ffc-stability-hold-ms",
        str(EVENT_DEFAULTS["ffc_stability_hold_ms"]),
        "--ffc-precontact-window-ms",
        str(EVENT_DEFAULTS["ffc_precontact_window_ms"]),
        "--ffc-ground-tolerance-foot-length",
        str(EVENT_DEFAULTS["ffc_ground_tolerance_foot_length"]),
        "--release-search-start-frac",
        str(EVENT_DEFAULTS["release_search_start_frac"]),
    ]
    run(feature_cmd)

    run([
        sys.executable,
        str(QC_SCRIPT),
        "--input",
        str(features_csv),
        "--annotated-out",
        str(qc_csv),
        "--report-out",
        str(qc_report),
        "--flow-out",
        str(qc_flow),
    ])

    run([
        sys.executable,
        str(PREP_SCRIPT),
        "--input",
        str(qc_csv),
        "--output-dir",
        str(analysis_dir),
        "--max-k",
        str(max_k),
        "--random-state",
        str(random_state),
    ])

    return {
        "condition": condition,
        "features": features_csv,
        "qc": qc_csv,
        "analysis": analysis_dir,
    }


def read_indexed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "unique_pitch_id" not in df.columns:
        raise ValueError(f"{path} is missing unique_pitch_id.")
    if df["unique_pitch_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate unique_pitch_id values.")
    return df.set_index("unique_pitch_id", drop=False)


def shared_numeric_features(reference: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    meta = {
        "video_file", "pitcher_id", "pitch_id", "unique_pitch_id",
        "measurement_qc_status", "model_eligible",
        "relative_mechanical_band_pooled", "relative_mechanical_band_within_pitcher",
        "relative_mechanical_score_pooled", "relative_mechanical_score_within_pitcher",
        "mechanical_flag_count", "mechanical_flag_profile",
    }
    cols = []
    for c in reference.columns.intersection(test.columns):
        if c in meta:
            continue
        r = pd.to_numeric(reference[c], errors="coerce")
        t = pd.to_numeric(test[c], errors="coerce")
        if r.notna().any() and t.notna().any():
            cols.append(c)
    return sorted(cols)


def feature_agreement(reference: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    common_ids = reference.index.intersection(test.index)
    ref = reference.loc[common_ids]
    tst = test.loc[common_ids]
    rows = []
    for feature in shared_numeric_features(ref, tst):
        x = pd.to_numeric(ref[feature], errors="coerce")
        y = pd.to_numeric(tst[feature], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 3:
            continue
        xv = x[valid].to_numpy(float)
        yv = y[valid].to_numpy(float)
        sd = float(np.std(xv, ddof=0))
        corr = float(np.corrcoef(xv, yv)[0, 1]) if np.std(xv) > 0 and np.std(yv) > 0 else np.nan
        mae = float(np.mean(np.abs(yv - xv)))
        rows.append({
            "feature": feature,
            "n_common": int(valid.sum()),
            "pearson_r": corr,
            "mae": mae,
            "reference_sd": sd,
            "normalized_mae_by_reference_sd": mae / sd if sd > 0 else np.nan,
            "mean_difference_test_minus_reference": float(np.mean(yv - xv)),
        })
    return pd.DataFrame(rows)


def pca_procrustes(reference_analysis: Path, test_analysis: Path, n_components: int = 5) -> dict:
    ref = read_indexed_csv(reference_analysis / "pca_scores.csv")
    tst = read_indexed_csv(test_analysis / "pca_scores.csv")
    common = ref.index.intersection(tst.index)
    pcs = [f"PC{i}" for i in range(1, n_components + 1)]
    pcs = [c for c in pcs if c in ref.columns and c in tst.columns]
    if len(common) < 3 or len(pcs) < 2:
        return {"n_common": int(len(common)), "n_components": len(pcs), "procrustes_disparity": np.nan}
    a = ref.loc[common, pcs].to_numpy(float)
    b = tst.loc[common, pcs].to_numpy(float)
    _, _, disparity = procrustes(a, b)
    return {
        "n_common": int(len(common)),
        "n_components": len(pcs),
        "procrustes_disparity": float(disparity),
    }


def cluster_agreement(reference_analysis: Path, test_analysis: Path) -> dict:
    ref = read_indexed_csv(reference_analysis / "cluster_assignments.csv")
    tst = read_indexed_csv(test_analysis / "cluster_assignments.csv")
    common = ref.index.intersection(tst.index)
    if len(common) < 2:
        ari = np.nan
    else:
        ari = float(adjusted_rand_score(ref.loc[common, "cluster"], tst.loc[common, "cluster"]))
    return {"n_common": int(len(common)), "adjusted_rand_index": ari}


def condition_summary(window_ms: float, analysis_dir: Path) -> dict:
    summary = json.loads((analysis_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    pca = pd.read_csv(analysis_dir / "pca_explained_variance.csv")
    out = {
        "window_ms": float(window_ms),
        "n_model_rows": int(summary["complete_case_model_rows"]),
        "n_retained_features": int(summary["retained_model_features"]),
        "best_k": int(summary["best_k"]),
        "best_silhouette": float(summary["best_silhouette"]),
    }
    for _, row in pca.head(5).iterrows():
        out[f"{row['PC']}_explained_variance_ratio"] = float(row["explained_variance_ratio"])
    return out


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    windows = list(dict.fromkeys(float(x) for x in a.windows_ms))
    if a.reference_window_ms not in windows:
        windows.append(float(a.reference_window_ms))
    windows = sorted(windows)

    if not a.pose_dir.exists():
        raise FileNotFoundError(f"Pose directory not found: {a.pose_dir}")
    if not a.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {a.metadata}")

    condition_paths = {}
    summaries = []
    for window in windows:
        paths = build_condition(
            window, a.pose_dir, a.metadata, a.output_dir, a.max_k, a.random_state
        )
        condition_paths[window] = paths
        summaries.append(condition_summary(window, paths["analysis"]))

    summary_df = pd.DataFrame(summaries).sort_values("window_ms")
    summary_df.to_csv(a.output_dir / "condition_summary.csv", index=False)

    ref_paths = condition_paths[float(a.reference_window_ms)]
    ref_model = read_indexed_csv(ref_paths["analysis"] / "model_ready.csv")

    pairwise_rows = []
    all_feature_rows = []
    for window in windows:
        test_paths = condition_paths[window]
        test_model = read_indexed_csv(test_paths["analysis"] / "model_ready.csv")

        f = feature_agreement(ref_model, test_model)
        if len(f):
            f.insert(0, "test_window_ms", float(window))
            f.insert(0, "reference_window_ms", float(a.reference_window_ms))
            all_feature_rows.append(f)

        pca_stats = pca_procrustes(ref_paths["analysis"], test_paths["analysis"])
        cluster_stats = cluster_agreement(ref_paths["analysis"], test_paths["analysis"])
        pairwise_rows.append({
            "reference_window_ms": float(a.reference_window_ms),
            "test_window_ms": float(window),
            **pca_stats,
            **cluster_stats,
        })

    if all_feature_rows:
        pd.concat(all_feature_rows, ignore_index=True).to_csv(
            a.output_dir / "feature_agreement_vs_reference.csv", index=False
        )

    pd.DataFrame(pairwise_rows).to_csv(
        a.output_dir / "representation_agreement_vs_reference.csv", index=False
    )

    manifest = {
        "analysis": "kinematic_smoothing_sensitivity",
        "windows_ms": windows,
        "reference_window_ms": float(a.reference_window_ms),
        "event_localization_parameters": EVENT_DEFAULTS,
        "interpretation": (
            "Kinematic smoothing is varied while event-localization parameters remain fixed. "
            "This is a sensitivity analysis and does not redefine the validated study pipeline."
        ),
    }
    (a.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nPreprocessing robustness analysis complete.")
    print(f"Summary: {a.output_dir / 'condition_summary.csv'}")
    print(f"Feature agreement: {a.output_dir / 'feature_agreement_vs_reference.csv'}")
    print(f"Representation agreement: {a.output_dir / 'representation_agreement_vs_reference.csv'}")


if __name__ == "__main__":
    main()
