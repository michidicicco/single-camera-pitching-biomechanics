#!/usr/bin/env python
"""
mechanical_profiles.py

Creates exploratory relative mechanical profiles without mixing measurement QC
into the mechanical score.

Mechanical profile score components:
- pooled relative mechanical band: lower=0, moderate=1, higher=2
- within-pitcher relative mechanical band: lower=0, moderate=1, higher=2
- mechanical flag burden: 0.5 * min(mechanical_flag_count, 4)

Measurement QC is carried separately as measurement_review_priority and never
raises the mechanical profile.

These profiles are dataset-relative descriptive categories, not diagnoses,
direct tissue measurements, or clinical outcome categories.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


PROFILE_LOWER = "lower"
PROFILE_MODERATE = "moderate"
PROFILE_HIGHER = "higher"

BAND_COL_POOLED = "relative_mechanical_band_pooled"
BAND_COL_WITHIN = "relative_mechanical_band_within_pitcher"
FLAG_COUNT_CANDIDATES = ["mechanical_flag_count"]


def normalize_string(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def band_to_score(value) -> float:
    txt = normalize_string(value)
    if "higher" in txt or txt == "high":
        return 2.0
    if "moderate" in txt or "middle" in txt:
        return 1.0
    if "lower" in txt or txt == "low":
        return 0.0
    return np.nan


def make_unique_pitch_id(df: pd.DataFrame) -> pd.Series:
    if "unique_pitch_id" in df.columns:
        return df["unique_pitch_id"].astype(str)
    if {"pitcher_id", "pitch_id"}.issubset(df.columns):
        return df["pitcher_id"].astype(str) + "_" + df["pitch_id"].astype(str)
    if "video_file" in df.columns:
        return df["video_file"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    return pd.Series([f"row_{i:04d}" for i in range(len(df))], index=df.index)


def get_flag_count(df: pd.DataFrame) -> pd.Series:
    for col in FLAG_COUNT_CANDIDATES:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def classify_profile(score, pooled_score, within_score, flag_count, moderate_threshold, higher_threshold):
    if not np.isfinite(score):
        return "not_evaluable"
    flags = 0.0 if not np.isfinite(flag_count) else float(flag_count)
    higher_override = (
        (pooled_score >= 2 and within_score >= 2)
        or (flags >= 3 and (pooled_score >= 2 or within_score >= 2))
    )
    if higher_override or score >= higher_threshold:
        return PROFILE_HIGHER
    if score >= moderate_threshold:
        return PROFILE_MODERATE
    return PROFILE_LOWER


def mechanical_review_priority(profile: str) -> str:
    if profile == PROFILE_HIGHER:
        return "high_mechanical_review_priority"
    if profile == PROFILE_MODERATE:
        return "moderate_mechanical_review_priority"
    if profile == PROFILE_LOWER:
        return "routine_mechanical_review_priority"
    return "not_evaluable"


def measurement_review_priority(status: str) -> str:
    txt = normalize_string(status)
    if txt == "exclude":
        return "measurement_not_interpretable"
    if txt == "watch":
        return "measurement_review_recommended"
    if txt == "pass":
        return "routine_measurement_review"
    return "measurement_status_unknown"


def reason_flags(row: pd.Series) -> str:
    reasons: List[str] = []
    pooled = normalize_string(row.get(BAND_COL_POOLED, ""))
    within = normalize_string(row.get(BAND_COL_WITHIN, ""))
    if pooled and pooled != "unknown":
        reasons.append(f"pooled={pooled}")
    if within and within != "unknown":
        reasons.append(f"within_pitcher={within}")
    flags = row.get("mechanical_flag_count", np.nan)
    if pd.notna(flags):
        reasons.append(f"mechanical_flag_count={int(float(flags))}")
    return ";".join(reasons) if reasons else "no_evaluable_mechanical_components"


def classify_dataframe(df: pd.DataFrame, moderate_threshold: float, higher_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["unique_pitch_id"] = make_unique_pitch_id(out)
    if BAND_COL_POOLED not in out.columns or BAND_COL_WITHIN not in out.columns:
        raise ValueError("Input must contain pooled and within-pitcher relative mechanical bands.")

    pooled = out[BAND_COL_POOLED].apply(band_to_score)
    within = out[BAND_COL_WITHIN].apply(band_to_score)
    flags = get_flag_count(out)
    out["pooled_reference_component"] = pooled
    out["within_pitcher_reference_component"] = within
    out["mechanical_flag_score_component"] = flags.clip(lower=0, upper=4) * 0.5
    mechanical_score = pooled + within + out["mechanical_flag_score_component"]
    out["relative_mechanical_profile_score"] = mechanical_score

    out["relative_mechanical_profile"] = [
        classify_profile(
            score=float(mechanical_score.loc[i]) if pd.notna(mechanical_score.loc[i]) else np.nan,
            pooled_score=float(pooled.loc[i]) if pd.notna(pooled.loc[i]) else np.nan,
            within_score=float(within.loc[i]) if pd.notna(within.loc[i]) else np.nan,
            flag_count=float(flags.loc[i]) if pd.notna(flags.loc[i]) else np.nan,
            moderate_threshold=moderate_threshold,
            higher_threshold=higher_threshold,
        )
        for i in out.index
    ]

    out["mechanical_review_priority"] = out["relative_mechanical_profile"].apply(mechanical_review_priority)
    if "measurement_qc_status" in out.columns:
        out["measurement_review_priority"] = out["measurement_qc_status"].apply(measurement_review_priority)
    else:
        out["measurement_review_priority"] = "measurement_status_unknown"
    out["mechanical_profile_reason_flags"] = out.apply(reason_flags, axis=1)
    return out


def sensitivity_grid(df: pd.DataFrame, base_moderate: float, base_higher: float) -> pd.DataFrame:
    baseline = classify_dataframe(df, base_moderate, base_higher)["relative_mechanical_profile"]
    moderate_values = sorted(set([base_moderate - 0.5, base_moderate, base_moderate + 0.5]))
    higher_values = sorted(set([base_higher - 0.5, base_higher, base_higher + 0.5]))
    rows = []
    for m in moderate_values:
        for h in higher_values:
            if h <= m:
                continue
            test = classify_dataframe(df, m, h)["relative_mechanical_profile"]
            evaluable = baseline.ne("not_evaluable") & test.ne("not_evaluable")
            agreement = float((baseline[evaluable] == test[evaluable]).mean()) if evaluable.any() else np.nan
            rows.append({
                "moderate_threshold": m,
                "higher_threshold": h,
                "agreement_with_baseline": agreement,
                "n_compared": int(evaluable.sum()),
                "lower_n": int((test == PROFILE_LOWER).sum()),
                "moderate_n": int((test == PROFILE_MODERATE).sum()),
                "higher_n": int((test == PROFILE_HIGHER).sum()),
                "not_evaluable_n": int((test == "not_evaluable").sum()),
            })
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    lines = [
        "RELATIVE MECHANICAL PROFILE SUMMARY",
        "=" * 70,
        "",
        "Mechanical profile and measurement quality are separate outputs.",
        "Measurement QC does not contribute to the mechanical score.",
        "These categories are exploratory and dataset-relative; they are not clinical categories.",
        "",
        "Mechanical profile counts:",
        df["relative_mechanical_profile"].value_counts(dropna=False).to_string(),
        "",
        "Mechanical profile counts by category:",
        df["mechanical_review_priority"].value_counts(dropna=False).to_string(),
        "",
        "Measurement review priority counts:",
        df["measurement_review_priority"].value_counts(dropna=False).to_string(),
        "",
    ]
    if "pitcher_id" in df.columns:
        lines += [
            "Mechanical profile by pitcher:",
            pd.crosstab(df["pitcher_id"], df["relative_mechanical_profile"], dropna=False).to_string(),
            "",
        ]
    lines += ["Threshold sensitivity:", sensitivity.to_string(index=False)]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify exploratory relative mechanical profiles without mixing in measurement QC.")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out", required=True)
    p.add_argument("--sensitivity-out", default=None)
    p.add_argument("--moderate-threshold", type=float, default=2.0)
    p.add_argument("--higher-threshold", type=float, default=4.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    sensitivity_path = Path(args.sensitivity_out) if args.sensitivity_out else out_path.with_name(out_path.stem + "_threshold_sensitivity.csv")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.higher_threshold <= args.moderate_threshold:
        raise ValueError("Higher threshold must be greater than moderate threshold.")

    df = pd.read_csv(input_path)
    classified = classify_dataframe(df, args.moderate_threshold, args.higher_threshold)
    sensitivity = sensitivity_grid(df, args.moderate_threshold, args.higher_threshold)
    for p in [out_path, summary_path, sensitivity_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(out_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    summary_path.write_text(build_summary(classified, sensitivity), encoding="utf-8")

    print("Relative mechanical profile classification complete.")
    print(f"Output: {out_path}")
    print(f"Summary: {summary_path}")
    print(f"Sensitivity: {sensitivity_path}")
    print(classified["relative_mechanical_profile"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
