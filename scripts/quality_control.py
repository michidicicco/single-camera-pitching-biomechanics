#!/usr/bin/env python
"""
quality_control.py

Row-level quality-control annotation for the pitching feature dataset.

Key design principle:
Measurement quality is kept separate from mechanical characterization. A pitch can be
mechanically unusual without being poor-quality, and a poor-quality recording
must not be made to look mechanically "higher" because of QC flags.

Outputs:
- annotated CSV with QC flags/status and model_eligible
- text report
- compact QC flow CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "features" / "pitch_features_master.csv"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "logs" / "features_qc_report.txt"
DEFAULT_ANNOTATED = PROJECT_ROOT / "outputs" / "features" / "pitch_features_qc_annotated.csv"
DEFAULT_FLOW = PROJECT_ROOT / "outputs" / "logs" / "features_qc_flow.csv"

CORE_MECHANICAL_COLUMNS = [
    "elbow_ang_vel_max_ffc_to_release",
    "wrist_speed_max_ffc_to_release",
    "trunk_tilt_deg_at_release",
    "trunk_tilt_vel_max_ffc_to_release",
    "hip_shoulder_sep_vel_max_ffc_to_release",
    "stride_length_norm_at_ffc",
    "lead_knee_extension_deg_ffc_to_release",
    "ffc_to_release_ms",
]

VISIBILITY_COLUMNS = [
    "throw_shoulder_visibility_mean",
    "throw_elbow_visibility_mean",
    "throw_wrist_visibility_mean",
    "lead_heel_visibility_mean",
    "lead_ankle_visibility_mean",
    "trail_ankle_visibility_mean",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate pitch-level measurement QC separately from mechanical characterization.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--annotated-out", type=Path, default=DEFAULT_ANNOTATED)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--flow-out", type=Path, default=DEFAULT_FLOW)
    p.add_argument("--min-pose-rate", type=float, default=0.90)
    p.add_argument("--min-critical-visibility", type=float, default=0.50)
    p.add_argument("--min-event-confidence", type=float, default=0.55)
    p.add_argument("--max-critical-gap-ms", type=float, default=100.0)
    p.add_argument("--min-event-window-ms", type=float, default=50.0)
    p.add_argument("--max-event-window-ms", type=float, default=500.0)
    return p.parse_args()


def unique_pitch_id(df: pd.DataFrame) -> pd.Series:
    if "unique_pitch_id" in df.columns:
        return df["unique_pitch_id"].astype(str)
    if {"pitcher_id", "pitch_id"}.issubset(df.columns):
        return df["pitcher_id"].astype(str) + "_" + df["pitch_id"].astype(str)
    if "video_file" in df.columns:
        return df["video_file"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    return pd.Series([f"row_{i:04d}" for i in range(len(df))], index=df.index)


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


def add_qc_flags(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["unique_pitch_id"] = unique_pitch_id(out)

    fps = numeric(out, "fps")
    if "fps_valid_for_time_derivatives" in out.columns:
        fps_valid = boolish(out["fps_valid_for_time_derivatives"])
    else:
        fps_valid = fps.notna() & (fps > 0)
    out["qc_exclude_invalid_fps"] = (~fps_valid).astype(int)

    ffc = numeric(out, "ffc_frame_idx")
    release = numeric(out, "release_frame_idx")
    seq_from_indices = ffc.notna() & release.notna() & (release > ffc)
    if "sequence_ok" in out.columns:
        seq_ok = boolish(out["sequence_ok"]) & seq_from_indices
    else:
        seq_ok = seq_from_indices
    out["qc_exclude_invalid_event_sequence"] = (~seq_ok).astype(int)

    if "processing_status" in out.columns:
        proc_ok = out["processing_status"].astype(str).str.lower().eq("ok")
        out["qc_exclude_processing_failure"] = (~proc_ok).astype(int)
    else:
        out["qc_exclude_processing_failure"] = 0

    present_core = [c for c in CORE_MECHANICAL_COLUMNS if c in out.columns]
    if present_core:
        missing_core = out[present_core].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)
    else:
        missing_core = pd.Series(True, index=out.index)
    out["qc_exclude_missing_required_mechanics"] = missing_core.astype(int)

    out["qc_exclude_duplicate_pitch_record"] = out["unique_pitch_id"].duplicated(keep=False).astype(int)
    if "video_file" in out.columns:
        dup_video = out["video_file"].astype(str).duplicated(keep=False)
        out["qc_exclude_duplicate_video"] = dup_video.astype(int)
    else:
        out["qc_exclude_duplicate_video"] = 0

    pose = numeric(out, "pose_detected_rate")
    out["qc_watch_low_pose_rate"] = ((pose.notna()) & (pose < args.min_pose_rate)).astype(int)

    available_vis = [c for c in VISIBILITY_COLUMNS if c in out.columns]
    if available_vis:
        vis_df = out[available_vis].apply(pd.to_numeric, errors="coerce")
        out["critical_visibility_min"] = vis_df.min(axis=1, skipna=True)
        out["qc_watch_low_critical_visibility"] = (
            out["critical_visibility_min"].notna()
            & (out["critical_visibility_min"] < args.min_critical_visibility)
        ).astype(int)
    else:
        out["critical_visibility_min"] = np.nan
        out["qc_watch_low_critical_visibility"] = 0

    ffc_conf = numeric(out, "ffc_confidence")
    rel_conf = numeric(out, "release_confidence")
    out["qc_watch_low_ffc_confidence"] = (
        ffc_conf.notna() & (ffc_conf < args.min_event_confidence)
    ).astype(int)
    out["qc_watch_low_release_confidence"] = (
        rel_conf.notna() & (rel_conf < args.min_event_confidence)
    ).astype(int)

    gap_ms = numeric(out, "critical_landmark_longest_raw_gap_ms")
    out["qc_watch_long_critical_gap"] = (
        gap_ms.notna() & (gap_ms > args.max_critical_gap_ms)
    ).astype(int)

    event_ms = numeric(out, "event_window_ms")
    if event_ms.isna().all() and "ffc_to_release_ms" in out.columns:
        event_ms = numeric(out, "ffc_to_release_ms")
    out["qc_watch_implausibly_short_event_window"] = (
        event_ms.notna() & (event_ms < args.min_event_window_ms)
    ).astype(int)
    out["qc_watch_implausibly_long_event_window"] = (
        event_ms.notna() & (event_ms > args.max_event_window_ms)
    ).astype(int)

    if "ffc_detection_method" in out.columns:
        out["qc_watch_ffc_fallback"] = (
            out["ffc_detection_method"].astype(str).str.contains("fallback", case=False, na=False)
        ).astype(int)
    else:
        out["qc_watch_ffc_fallback"] = 0

    hard_cols = [c for c in out.columns if c.startswith("qc_exclude_")]
    watch_cols = [c for c in out.columns if c.startswith("qc_watch_")]
    out["measurement_qc_exclusion_count"] = out[hard_cols].sum(axis=1)
    out["measurement_qc_watch_count"] = out[watch_cols].sum(axis=1)

    out["measurement_qc_status"] = np.select(
        [
            out["measurement_qc_exclusion_count"] > 0,
            out["measurement_qc_watch_count"] > 0,
        ],
        ["exclude", "watch"],
        default="pass",
    )
    out["model_eligible"] = (out["measurement_qc_status"] != "exclude").astype(int)
    out["mechanical_analysis_eligible"] = out["model_eligible"]

    def reason_string(row: pd.Series) -> str:
        reasons = []
        for c in hard_cols + watch_cols:
            if int(row.get(c, 0)) == 1:
                reasons.append(c)
        return ";".join(reasons) if reasons else "none"

    out["measurement_qc_reasons"] = out.apply(reason_string, axis=1)
    return out


def build_flow(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"stage": "input_pitch_records", "n": int(len(df))},
        {"stage": "measurement_qc_pass", "n": int((df["measurement_qc_status"] == "pass").sum())},
        {"stage": "measurement_qc_watch", "n": int((df["measurement_qc_status"] == "watch").sum())},
        {"stage": "measurement_qc_exclude", "n": int((df["measurement_qc_status"] == "exclude").sum())},
        {"stage": "model_eligible", "n": int((df["model_eligible"] == 1).sum())},
    ]
    for c in [c for c in df.columns if c.startswith("qc_exclude_") or c.startswith("qc_watch_")]:
        rows.append({"stage": c, "n": int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())})
    return pd.DataFrame(rows)


def zero_variance_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) and s.nunique() <= 1:
            cols.append(c)
    return cols


def build_report(df: pd.DataFrame, flow: pd.DataFrame, args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines += ["FEATURES QC REPORT", "=" * 70, ""]
    lines.append(f"Input rows: {len(df)}")
    lines.append("")
    lines.append("Operational QC thresholds (investigator-defined; not clinical thresholds):")
    lines.append(f"- minimum pose-detection rate: {args.min_pose_rate}")
    lines.append(f"- minimum critical-landmark visibility: {args.min_critical_visibility}")
    lines.append(f"- minimum event algorithmic confidence: {args.min_event_confidence}")
    lines.append(f"- maximum critical raw missing-landmark gap: {args.max_critical_gap_ms} ms")
    lines.append(f"- event-window review range: {args.min_event_window_ms} to {args.max_event_window_ms} ms")
    lines.append("")
    lines.append("QC flow:")
    lines.append(flow.to_string(index=False))
    lines.append("")

    lines.append("QC status by pitcher:")
    if "pitcher_id" in df.columns:
        lines.append(pd.crosstab(df["pitcher_id"], df["measurement_qc_status"], dropna=False).to_string())
    else:
        lines.append("pitcher_id unavailable")
    lines.append("")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    lines.append("Columns with missing values:")
    lines.append(missing.to_string() if len(missing) else "None")
    lines.append("")

    lines.append("Zero-variance numeric columns:")
    zv = zero_variance_columns(df)
    lines.append("\n".join(f"- {x}" for x in zv) if zv else "None")
    lines.append("")

    summary_cols = [
        "fps", "pose_detected_rate", "critical_visibility_min",
        "ffc_confidence", "release_confidence",
        "critical_landmark_longest_raw_gap_ms", "ffc_to_release_ms",
        "elbow_ang_vel_max_ffc_to_release", "wrist_speed_max_ffc_to_release",
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    lines.append("Key numeric summary:")
    lines.append(df[summary_cols].describe().round(4).to_string() if summary_cols else "None")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    annotated = add_qc_flags(df, args)
    flow = build_flow(annotated)
    report = build_report(annotated, flow, args)

    for p in [args.annotated_out, args.report_out, args.flow_out]:
        p.parent.mkdir(parents=True, exist_ok=True)

    annotated.to_csv(args.annotated_out, index=False)
    flow.to_csv(args.flow_out, index=False)
    args.report_out.write_text(report, encoding="utf-8")

    print(report)
    print()
    print(f"Annotated QC CSV: {args.annotated_out}")
    print(f"QC flow CSV     : {args.flow_out}")
    print(f"QC report       : {args.report_out}")


if __name__ == "__main__":
    main()
