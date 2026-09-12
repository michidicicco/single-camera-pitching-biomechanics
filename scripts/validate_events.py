#!/usr/bin/env python
"""
validate_events.py

Compares algorithm-estimated front-foot contact (FFC) and ball-release frames
against a manually annotated subset.

Manual annotation CSV should contain a pitch key plus:
- manual_ffc_frame_idx
- manual_release_frame_idx

Supported keys, in priority order:
- unique_pitch_id
- pitcher_id + pitch_id
- video_file

Outputs:
- pitch-level event errors
- summary metrics for FFC, release, and FFC-to-release duration

This validates event timing agreement only. It does not validate kinetics,
tissue-level mechanical measures, or clinical outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def add_unique_pitch_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "unique_pitch_id" in out.columns:
        out["unique_pitch_id"] = out["unique_pitch_id"].astype(str)
        return out
    if {"pitcher_id", "pitch_id"}.issubset(out.columns):
        out["unique_pitch_id"] = out["pitcher_id"].astype(str) + "_" + out["pitch_id"].astype(str)
        return out
    if "video_file" in out.columns:
        out["unique_pitch_id"] = out["video_file"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
        return out
    raise ValueError("No pitch identifier is available.")


def metric_summary(errors: pd.Series, abs_errors: pd.Series, ms_abs: pd.Series, event: str) -> dict:
    valid = errors.notna() & abs_errors.notna()
    e = errors[valid].astype(float)
    ae = abs_errors[valid].astype(float)
    ms = ms_abs[valid].astype(float)
    if len(e) == 0:
        return {"event": event, "n": 0}
    return {
        "event": event,
        "n": int(len(e)),
        "mean_signed_frame_error": float(e.mean()),
        "mean_absolute_frame_error": float(ae.mean()),
        "median_absolute_frame_error": float(ae.median()),
        "rmse_frames": float(np.sqrt(np.mean(e**2))),
        "mean_absolute_error_ms": float(ms.mean()) if len(ms) else np.nan,
        "median_absolute_error_ms": float(ms.median()) if len(ms) else np.nan,
        "within_1_frame_fraction": float((ae <= 1).mean()),
        "within_2_frames_fraction": float((ae <= 2).mean()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate automated FFC/release event frames against manual annotations.")
    p.add_argument("--features", required=True, help="Feature CSV containing automated event frames and FPS.")
    p.add_argument("--manual", required=True, help="Manual annotation CSV.")
    p.add_argument("--out", required=True, help="Pitch-level event error CSV.")
    p.add_argument("--summary-out", required=True, help="Summary CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features = add_unique_pitch_id(pd.read_csv(args.features))
    manual = add_unique_pitch_id(pd.read_csv(args.manual))

    required_auto = ["unique_pitch_id", "ffc_frame_idx", "release_frame_idx", "fps"]
    required_manual = ["unique_pitch_id", "manual_ffc_frame_idx", "manual_release_frame_idx"]
    for label, df, req in [("features", features, required_auto), ("manual", manual, required_manual)]:
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"{label} missing required columns: {missing}")

    if features["unique_pitch_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate unique_pitch_id values.")
    if manual["unique_pitch_id"].duplicated().any():
        raise ValueError("Manual annotation table contains duplicate unique_pitch_id values.")

    auto_cols = [
        c for c in [
            "unique_pitch_id", "video_file", "pitcher_id", "pitch_id", "fps",
            "ffc_frame_idx", "release_frame_idx", "ffc_confidence", "release_confidence",
            "ffc_detection_method", "release_detection_method",
        ] if c in features.columns
    ]
    manual_cols = [
        c for c in [
            "unique_pitch_id", "manual_ffc_frame_idx", "manual_release_frame_idx",
            "annotator", "notes",
        ] if c in manual.columns
    ]

    merged = manual[manual_cols].merge(
        features[auto_cols], on="unique_pitch_id", how="left", validate="one_to_one"
    )

    fps = pd.to_numeric(merged["fps"], errors="coerce")
    for event in ["ffc", "release"]:
        auto = pd.to_numeric(merged[f"{event}_frame_idx"], errors="coerce")
        man = pd.to_numeric(merged[f"manual_{event}_frame_idx"], errors="coerce")
        err = auto - man
        merged[f"{event}_signed_frame_error"] = err
        merged[f"{event}_absolute_frame_error"] = err.abs()
        merged[f"{event}_signed_error_ms"] = err * 1000.0 / fps
        merged[f"{event}_absolute_error_ms"] = merged[f"{event}_signed_error_ms"].abs()

    auto_dur_frames = (
        pd.to_numeric(merged["release_frame_idx"], errors="coerce")
        - pd.to_numeric(merged["ffc_frame_idx"], errors="coerce")
    )
    manual_dur_frames = (
        pd.to_numeric(merged["manual_release_frame_idx"], errors="coerce")
        - pd.to_numeric(merged["manual_ffc_frame_idx"], errors="coerce")
    )
    dur_err = auto_dur_frames - manual_dur_frames
    merged["ffc_to_release_signed_frame_error"] = dur_err
    merged["ffc_to_release_absolute_frame_error"] = dur_err.abs()
    merged["ffc_to_release_signed_error_ms"] = dur_err * 1000.0 / fps
    merged["ffc_to_release_absolute_error_ms"] = merged["ffc_to_release_signed_error_ms"].abs()

    summary_rows = [
        metric_summary(
            merged["ffc_signed_frame_error"],
            merged["ffc_absolute_frame_error"],
            merged["ffc_absolute_error_ms"],
            "front_foot_contact",
        ),
        metric_summary(
            merged["release_signed_frame_error"],
            merged["release_absolute_frame_error"],
            merged["release_absolute_error_ms"],
            "ball_release",
        ),
        metric_summary(
            merged["ffc_to_release_signed_frame_error"],
            merged["ffc_to_release_absolute_frame_error"],
            merged["ffc_to_release_absolute_error_ms"],
            "ffc_to_release_duration",
        ),
    ]
    summary = pd.DataFrame(summary_rows)

    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Event validation complete.")
    print(f"Matched manual annotations: {int(merged['ffc_frame_idx'].notna().sum())}/{len(merged)}")
    print(summary.to_string(index=False))
    print(f"Pitch-level errors: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
