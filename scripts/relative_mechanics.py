#!/usr/bin/env python
"""
relative_mechanics.py

Creates dataset-relative mechanical descriptors after measurement QC.

Key properties:
- Relative mechanical scores are calculated only for QC-eligible pitches when a QC flag exists.
- Mechanical flags are separated from measurement-quality flags.
- QC does not increase or decrease the mechanical score.
- Baseline investigator-defined weights are accompanied by explicit sensitivity analysis.
- Equal-weight and one-feature weight perturbation scenarios are compared with the baseline using rank correlation and band agreement.
- Within-pitcher reference calculations require a minimum number of eligible pitches.

These are relative mechanical descriptors, not direct kinetic measurements,
tissue-level measures, diagnoses, or clinical predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "features" / "pitch_features_qc_annotated.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "features" / "pitch_features_with_relative_mechanics.csv"

MECHANICAL_WEIGHTS: Dict[str, float] = {
    "elbow_ang_vel_max_ffc_to_release": 0.30,
    "wrist_speed_max_ffc_to_release": 0.20,
    "hip_shoulder_sep_vel_max_ffc_to_release": 0.20,
    "trunk_tilt_vel_max_ffc_to_release": 0.15,
    "trunk_tilt_deg_at_release": 0.15,
}

MECHANICAL_COLUMNS = [
    "elbow_ang_vel_max_ffc_to_release",
    "trunk_tilt_deg_at_release",
    "trunk_tilt_vel_max_ffc_to_release",
    "hip_shoulder_sep_vel_max_ffc_to_release",
    "stride_length_norm_at_ffc",
    "lead_knee_extension_deg_ffc_to_release",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add dataset-relative mechanical descriptors.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sensitivity-out", type=Path, default=None)
    p.add_argument("--min-within-pitcher-n", type=int, default=5,
                   help="Minimum eligible pitches required for within-pitcher bands. Default: 5.")
    p.add_argument("--weight-perturbation", type=float, default=0.20,
                   help="One-feature relative weight perturbation for sensitivity analysis. Default: 0.20.")
    return p.parse_args()


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n  - " + "\n  - ".join(missing))


def eligible_mask(df: pd.DataFrame) -> pd.Series:
    if "mechanical_analysis_eligible" in df.columns:
        return pd.to_numeric(df["mechanical_analysis_eligible"], errors="coerce").fillna(0).astype(int) == 1
    if "model_eligible" in df.columns:
        return pd.to_numeric(df["model_eligible"], errors="coerce").fillna(0).astype(int) == 1
    if "measurement_qc_status" in df.columns:
        return ~df["measurement_qc_status"].astype(str).str.lower().eq("exclude")
    return pd.Series(True, index=df.index)


def zscore_on_mask(series: pd.Series, mask: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    ref = values[mask & values.notna()]
    if len(ref) < 2:
        return out
    std = ref.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        out.loc[mask & values.notna()] = 0.0
        return out
    out.loc[mask & values.notna()] = (values.loc[mask & values.notna()] - ref.mean()) / std
    return out


def groupwise_zscore_on_mask(series, groups, mask, min_group_n):
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    n_ref = pd.Series(0, index=values.index, dtype=int)
    for group, idx in groups.groupby(groups).groups.items():
        idx = pd.Index(idx)
        group_mask = mask.loc[idx]
        valid_idx = idx[group_mask.values & values.loc[idx].notna().values]
        n = len(valid_idx)
        n_ref.loc[idx] = n
        if n < min_group_n:
            continue
        ref = values.loc[valid_idx]
        std = ref.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            out.loc[valid_idx] = 0.0
        else:
            out.loc[valid_idx] = (ref - ref.mean()) / std
    return out, n_ref


def assign_bands(series: pd.Series, mask: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=series.index, dtype=object)
    valid = mask & pd.to_numeric(series, errors="coerce").notna()
    ref = pd.to_numeric(series[valid], errors="coerce")
    if len(ref) < 3:
        return out
    q33, q67 = ref.quantile([1 / 3, 2 / 3]).tolist()
    x = pd.to_numeric(series, errors="coerce")
    out.loc[valid & (x <= q33)] = "lower"
    out.loc[valid & (x > q33) & (x <= q67)] = "moderate"
    out.loc[valid & (x > q67)] = "higher"
    return out


def assign_bands_within(series, groups, mask, min_group_n):
    out = pd.Series("unknown", index=series.index, dtype=object)
    for group, idx in groups.groupby(groups).groups.items():
        idx = pd.Index(idx)
        valid = mask.loc[idx] & pd.to_numeric(series.loc[idx], errors="coerce").notna()
        valid_idx = idx[valid.values]
        if len(valid_idx) < min_group_n:
            continue
        s = pd.to_numeric(series.loc[valid_idx], errors="coerce")
        q33, q67 = s.quantile([1 / 3, 2 / 3]).tolist()
        out.loc[valid_idx[s <= q33]] = "lower"
        out.loc[valid_idx[(s > q33) & (s <= q67)]] = "moderate"
        out.loc[valid_idx[s > q67]] = "higher"
    return out


def normalized_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(abs(float(v)) for v in weights.values())
    if total <= 0:
        raise ValueError("Weight sum must be positive.")
    return {k: float(v) / total for k, v in weights.items()}


def score_from_z(z_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    w = normalized_weights(weights)
    score = pd.Series(0.0, index=z_df.index)
    valid_all = pd.Series(True, index=z_df.index)
    for feat in w:
        col = f"{feat}_z"
        valid_all &= z_df[col].notna()
        score = score + z_df[col].fillna(0.0) * w[feat]
    score.loc[~valid_all] = np.nan
    return score


def make_z_matrices(df, mask, min_within_n):
    pooled = pd.DataFrame(index=df.index)
    within = pd.DataFrame(index=df.index)
    groups = df["pitcher_id"].fillna("unknown_pitcher") if "pitcher_id" in df.columns else pd.Series("all", index=df.index)
    within_n = pd.Series(0, index=df.index, dtype=int)
    for feat in MECHANICAL_WEIGHTS:
        pooled[f"{feat}_z"] = zscore_on_mask(df[feat], mask)
        z, n = groupwise_zscore_on_mask(df[feat], groups, mask, min_within_n)
        within[f"{feat}_z"] = z
        within_n = pd.concat([within_n, n], axis=1).max(axis=1).astype(int)
    return pooled, within, within_n


def add_relative_mechanical_scores(df, mask, min_within_n):
    validate_columns(df, list(MECHANICAL_WEIGHTS))
    out = df.copy()
    pooled_z, within_z, within_n = make_z_matrices(out, mask, min_within_n)
    out["relative_mechanical_score_pooled"] = score_from_z(pooled_z, MECHANICAL_WEIGHTS)
    out["relative_mechanical_band_pooled"] = assign_bands(out["relative_mechanical_score_pooled"], mask)
    groups = out["pitcher_id"].fillna("unknown_pitcher") if "pitcher_id" in out.columns else pd.Series("all", index=out.index)
    within_mask = mask & (within_n >= min_within_n)
    out["relative_mechanical_score_within_pitcher"] = score_from_z(within_z, MECHANICAL_WEIGHTS)
    out["relative_mechanical_band_within_pitcher"] = assign_bands_within(
        out["relative_mechanical_score_within_pitcher"], groups, within_mask, min_within_n
    )
    out["within_pitcher_reference_n"] = within_n
    out["within_pitcher_reference_eligible"] = within_mask.astype(int)
    equal_weights = {k: 1.0 for k in MECHANICAL_WEIGHTS}
    out["relative_mechanical_score_pooled_equal_weight"] = score_from_z(pooled_z, equal_weights)
    out["relative_mechanical_score_within_pitcher_equal_weight"] = score_from_z(within_z, equal_weights)
    return out, pooled_z, within_z


def quantile_on_eligible(df, col, mask, q):
    s = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna()
    return float(s.quantile(q)) if len(s) else np.nan


def add_mechanical_flags(df, mask):
    validate_columns(df, MECHANICAL_COLUMNS)
    out = df.copy()
    q75 = {c: quantile_on_eligible(out, c, mask, 0.75) for c in [
        "elbow_ang_vel_max_ffc_to_release", "trunk_tilt_deg_at_release",
        "trunk_tilt_vel_max_ffc_to_release", "hip_shoulder_sep_vel_max_ffc_to_release"]}
    q25_knee = quantile_on_eligible(out, "lead_knee_extension_deg_ffc_to_release", mask, 0.25)
    stride = pd.to_numeric(out["stride_length_norm_at_ffc"], errors="coerce")
    stride_med = float(stride[mask & stride.notna()].median()) if (mask & stride.notna()).any() else np.nan
    stride_dev = (stride - stride_med).abs()
    stride_dev_q75 = float(stride_dev[mask & stride_dev.notna()].quantile(0.75)) if (mask & stride_dev.notna()).any() else np.nan

    def high_flag(col):
        x = pd.to_numeric(out[col], errors="coerce")
        threshold = q75[col]
        return (mask & x.notna() & (x > threshold)).astype(int) if np.isfinite(threshold) else pd.Series(0, index=out.index)

    out["flag_high_elbow_vel"] = high_flag("elbow_ang_vel_max_ffc_to_release")
    out["flag_high_trunk_tilt_release"] = high_flag("trunk_tilt_deg_at_release")
    out["flag_high_trunk_tilt_vel"] = high_flag("trunk_tilt_vel_max_ffc_to_release")
    out["flag_high_sep_vel"] = high_flag("hip_shoulder_sep_vel_max_ffc_to_release")
    out["flag_atypical_stride_ffc"] = (mask & stride_dev.notna() & (stride_dev > stride_dev_q75)).astype(int) if np.isfinite(stride_dev_q75) else 0
    knee = pd.to_numeric(out["lead_knee_extension_deg_ffc_to_release"], errors="coerce")
    out["flag_low_lead_knee_extension"] = (mask & knee.notna() & (knee < q25_knee)).astype(int) if np.isfinite(q25_knee) else 0
    mechanical_flags = ["flag_high_elbow_vel", "flag_high_trunk_tilt_release", "flag_high_trunk_tilt_vel", "flag_high_sep_vel", "flag_atypical_stride_ffc", "flag_low_lead_knee_extension"]
    out["mechanical_flag_count"] = out[mechanical_flags].sum(axis=1)
    out.loc[~mask, "mechanical_flag_count"] = np.nan

    def burden_label(v):
        if pd.isna(v): return "not_evaluable"
        v = int(v)
        if v <= 1: return "lower_flag_burden"
        if v <= 3: return "moderate_flag_burden"
        return "multiple_flag_burden"
    out["mechanical_flag_profile"] = out["mechanical_flag_count"].apply(burden_label)
    return out


def scenario_weights(base, perturb):
    scenarios = {"baseline_investigator_weights": dict(base), "equal_weights": {k: 1.0 for k in base}}
    for feat in base:
        up, down = dict(base), dict(base)
        up[feat] *= (1.0 + perturb)
        down[feat] *= max(0.0, 1.0 - perturb)
        scenarios[f"{feat}_plus_{int(round(perturb*100))}pct"] = up
        scenarios[f"{feat}_minus_{int(round(perturb*100))}pct"] = down
    return scenarios


def weight_sensitivity(df, pooled_z, within_z, mask, min_within_n, perturb):
    groups = df["pitcher_id"].fillna("unknown_pitcher") if "pitcher_id" in df.columns else pd.Series("all", index=df.index)
    within_mask = mask & (pd.to_numeric(df["within_pitcher_reference_n"], errors="coerce") >= min_within_n)
    base_pooled = pd.to_numeric(df["relative_mechanical_score_pooled"], errors="coerce")
    base_pool_band = df["relative_mechanical_band_pooled"].astype(str)
    base_within = pd.to_numeric(df["relative_mechanical_score_within_pitcher"], errors="coerce")
    base_within_band = df["relative_mechanical_band_within_pitcher"].astype(str)
    rows = []
    for name, weights in scenario_weights(MECHANICAL_WEIGHTS, perturb).items():
        pool_score = score_from_z(pooled_z, weights)
        pool_band = assign_bands(pool_score, mask)
        within_score = score_from_z(within_z, weights)
        within_band = assign_bands_within(within_score, groups, within_mask, min_within_n)
        pvalid = mask & base_pooled.notna() & pool_score.notna()
        wvalid = within_mask & base_within.notna() & within_score.notna()
        pool_rho = base_pooled[pvalid].corr(pool_score[pvalid], method="spearman") if pvalid.sum() >= 3 else np.nan
        within_rho = base_within[wvalid].corr(within_score[wvalid], method="spearman") if wvalid.sum() >= 3 else np.nan
        pool_agree = float((base_pool_band[pvalid] == pool_band[pvalid]).mean()) if pvalid.any() else np.nan
        within_agree = float((base_within_band[wvalid] == within_band[wvalid]).mean()) if wvalid.any() else np.nan
        rows.append({"scenario": name, "weights_normalized": "; ".join(f"{k}={v:.4f}" for k, v in normalized_weights(weights).items()), "pooled_spearman_vs_baseline": pool_rho, "pooled_band_agreement_vs_baseline": pool_agree, "within_pitcher_spearman_vs_baseline": within_rho, "within_pitcher_band_agreement_vs_baseline": within_agree, "n_pooled_compared": int(pvalid.sum()), "n_within_compared": int(wvalid.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    df = pd.read_csv(args.input)
    mask = eligible_mask(df)
    df, pooled_z, within_z = add_relative_mechanical_scores(df, mask, args.min_within_pitcher_n)
    df = add_mechanical_flags(df, mask)
    sensitivity = weight_sensitivity(df, pooled_z, within_z, mask, args.min_within_pitcher_n, args.weight_perturbation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_path = args.sensitivity_out or args.out.with_name(args.out.stem + "_weight_sensitivity.csv")
    sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)

    print("=" * 72)
    print("RELATIVE MECHANICAL DESCRIPTORS CREATED")
    print("=" * 72)
    print(f"Input                         : {args.input}")
    print(f"Output                        : {args.out}")
    print(f"Weight sensitivity            : {sensitivity_path}")
    print(f"Rows                          : {len(df)}")
    print(f"QC/mechanical-analysis eligible pitches : {int(mask.sum())}")
    print()
    print("Pooled relative mechanical bands:")
    print(df["relative_mechanical_band_pooled"].value_counts(dropna=False).to_string())
    print()
    print("Within-pitcher relative mechanical bands:")
    print(df["relative_mechanical_band_within_pitcher"].value_counts(dropna=False).to_string())
    print()
    print("Mechanical flag burden:")
    print(df["mechanical_flag_profile"].value_counts(dropna=False).to_string())
    print()
    print("Weight sensitivity summary:")
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
