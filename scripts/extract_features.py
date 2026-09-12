#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
CRITICAL_LANDMARKS = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle']

def get_xy(df: pd.DataFrame, base_name: str) -> np.ndarray:
    cols = [f'{base_name}_x', f'{base_name}_y']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return np.full((len(df), 2), np.nan, dtype=float)
    return df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

def get_vis(df: pd.DataFrame, base_name: str) -> np.ndarray:
    col = f'{base_name}_visibility'
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
    return np.full(len(df), np.nan, dtype=float)

def longest_nan_run_1d(x: np.ndarray) -> int:
    isnan = ~np.isfinite(np.asarray(x, dtype=float))
    best = run = 0
    for val in isnan:
        if val:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)

def longest_nan_run_2d(arr: np.ndarray) -> int:
    arr = np.asarray(arr, dtype=float)
    missing_any = ~np.isfinite(arr).all(axis=1)
    best = run = 0
    for val in missing_any:
        if val:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)

def interpolate_short_internal_gaps_1d(x: np.ndarray, max_gap_frames: int) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    n = len(out)
    i = 0
    while i < n:
        if np.isfinite(out[i]):
            i += 1
            continue
        start = i
        while i < n and (not np.isfinite(out[i])):
            i += 1
        end = i
        gap_len = end - start
        left = start - 1
        right = end
        if gap_len <= max_gap_frames and left >= 0 and (right < n) and np.isfinite(out[left]) and np.isfinite(out[right]):
            out[start:end] = np.linspace(out[left], out[right], gap_len + 2)[1:-1]
    return out

def interpolate_short_internal_gaps_2d(arr: np.ndarray, max_gap_frames: int) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    for axis in range(out.shape[1]):
        out[:, axis] = interpolate_short_internal_gaps_1d(out[:, axis], max_gap_frames)
    return out

def contiguous_finite_runs(x: np.ndarray) -> list[tuple[int, int]]:
    finite = np.isfinite(np.asarray(x, dtype=float))
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(finite)
    while i < n:
        if not finite[i]:
            i += 1
            continue
        start = i
        while i < n and finite[i]:
            i += 1
        runs.append((start, i))
    return runs

def odd_window_from_ms(window_ms: float, fps: float, minimum: int=5) -> int:
    raw = max(minimum, int(round(window_ms * fps / 1000.0)))
    if raw % 2 == 0:
        raw += 1
    return raw

def smooth_1d_preserve_long_gaps(x: np.ndarray, window_frames: int, poly: int=2) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    for start, end in contiguous_finite_runs(out):
        segment = out[start:end]
        if len(segment) < 5:
            continue
        w = min(window_frames, len(segment) if len(segment) % 2 == 1 else len(segment) - 1)
        if w < 5:
            continue
        p = min(poly, w - 1)
        out[start:end] = savgol_filter(segment, window_length=w, polyorder=p, mode='interp')
    return out

def preprocess_xy(arr: np.ndarray, max_gap_frames: int, smooth_window_frames: int, poly: int=2) -> tuple[np.ndarray, int]:
    raw_max_gap = longest_nan_run_2d(arr)
    out = interpolate_short_internal_gaps_2d(arr, max_gap_frames=max_gap_frames)
    for axis in range(out.shape[1]):
        out[:, axis] = smooth_1d_preserve_long_gaps(out[:, axis], window_frames=smooth_window_frames, poly=poly)
    return (out, raw_max_gap)

def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0

def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)

def angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    num = np.sum(ba * bc, axis=1)
    den = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    den = np.where(den == 0, np.nan, den)
    cosang = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def line_angle_deg(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    v = p2 - p1
    return np.degrees(np.arctan2(v[:, 1], v[:, 0]))

def wrap_signed_180(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x + 180.0) % 360.0 - 180.0

def wrap_abs_180(x: np.ndarray) -> np.ndarray:
    return np.abs(wrap_signed_180(x))

def signed_trunk_tilt_deg(hip_mid: np.ndarray, shoulder_mid: np.ndarray) -> np.ndarray:
    v = shoulder_mid - hip_mid
    return np.degrees(np.arctan2(v[:, 0], -v[:, 1]))

def speed_per_second(xy: np.ndarray, fps: float) -> np.ndarray:
    if not np.isfinite(fps) or fps <= 0:
        return np.full(len(xy), np.nan)
    delta = np.diff(xy, axis=0)
    good = np.isfinite(delta).all(axis=1)
    d = np.full(len(delta), np.nan)
    d[good] = np.linalg.norm(delta[good], axis=1) * fps
    return np.insert(d, 0, np.nan)

def speed_axis_per_second(xy: np.ndarray, fps: float, axis: int=0) -> np.ndarray:
    if not np.isfinite(fps) or fps <= 0:
        return np.full(len(xy), np.nan)
    delta = np.diff(xy[:, axis], axis=0)
    d = np.abs(delta) * fps
    return np.insert(d, 0, np.nan)

def angular_speed_abs_per_second(angle_deg: np.ndarray, fps: float) -> np.ndarray:
    if not np.isfinite(fps) or fps <= 0:
        return np.full(len(angle_deg), np.nan)
    a = np.asarray(angle_deg, dtype=float)
    d = np.diff(a)
    d = wrap_signed_180(d)
    return np.insert(np.abs(d) * fps, 0, np.nan)

def nanargmax_optional(x: np.ndarray) -> int | None:
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.all(~np.isfinite(x)):
        return None
    return int(np.nanargmax(x))

def nanargmin_optional(x: np.ndarray) -> int | None:
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.all(~np.isfinite(x)):
        return None
    return int(np.nanargmin(x))

def value_at(x: np.ndarray, idx: int | None) -> float:
    if idx is None or len(x) == 0:
        return np.nan
    idx = max(0, min(int(idx), len(x) - 1))
    val = x[idx]
    return float(val) if np.isfinite(val) else np.nan

def stat_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

def stat_max(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan

def stat_min(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan

def stat_range(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return np.nan
    return float(np.nanmax(x) - np.nanmin(x))

def percentile_safe(x: np.ndarray, q: float, fallback: float=1.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return fallback
    v = np.percentile(x, q)
    if not np.isfinite(v) or v <= 0:
        return fallback
    return float(v)

def mean_visibility_at_window(vis: np.ndarray, start: int, end: int) -> float:
    if len(vis) == 0:
        return np.nan
    sub = vis[max(0, start):min(len(vis), end)]
    return stat_mean(sub)

def choose_side_map(throwing_side: str) -> dict[str, str]:
    throwing_side = str(throwing_side).lower()
    if throwing_side == 'right':
        return {'throw_shoulder': 'right_shoulder', 'throw_elbow': 'right_elbow', 'throw_wrist': 'right_wrist', 'lead_hip': 'left_hip', 'lead_knee': 'left_knee', 'lead_ankle': 'left_ankle', 'lead_heel': 'left_heel', 'lead_foot_index': 'left_foot_index', 'trail_hip': 'right_hip', 'trail_knee': 'right_knee', 'trail_ankle': 'right_ankle'}
    if throwing_side == 'left':
        return {'throw_shoulder': 'left_shoulder', 'throw_elbow': 'left_elbow', 'throw_wrist': 'left_wrist', 'lead_hip': 'right_hip', 'lead_knee': 'right_knee', 'lead_ankle': 'right_ankle', 'lead_heel': 'right_heel', 'lead_foot_index': 'right_foot_index', 'trail_hip': 'left_hip', 'trail_knee': 'left_knee', 'trail_ankle': 'left_ankle'}
    raise ValueError("throwing_side must be 'right' or 'left'")

def resolve_fps(df: pd.DataFrame) -> tuple[float, bool, str]:
    if 'fps' not in df.columns:
        return (np.nan, False, 'missing_fps_column')
    vals = pd.to_numeric(df['fps'], errors='coerce')
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.empty:
        return (np.nan, False, 'invalid_or_missing_fps')
    fps = float(vals.median())
    if vals.max() - vals.min() > max(0.01, 0.01 * fps):
        return (fps, False, 'inconsistent_fps_within_pose_csv')
    for flag_col in ['fps_is_fallback', 'fps_defaulted']:
        if flag_col in df.columns:
            flagged = pd.to_numeric(df[flag_col], errors='coerce').fillna(0).astype(bool)
            if flagged.any():
                return (fps, False, f'known_fallback_from_{flag_col}')
    if 'fps_source' in df.columns:
        source = str(df['fps_source'].iloc[0]).strip().lower()
        if any((token in source for token in ['fallback', 'default', 'assumed', 'unknown', 'invalid'])):
            return (fps, False, f'fps_source={source}')
        return (fps, True, f"fps_source={source or 'metadata'}")
    return (fps, True, 'positive_pose_csv_fps_unverified_provenance')

def event_signal_confidence(signal: np.ndarray, idx: int, visibility: float=np.nan) -> float:
    finite = np.asarray(signal, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3 or not np.isfinite(signal[idx]):
        return np.nan
    peak = float(signal[idx])
    med = float(np.median(finite))
    denom = max(abs(peak), 1e-12)
    contrast = float(np.clip((peak - med) / denom, 0.0, 1.0))
    if np.isfinite(visibility):
        return float(np.clip(0.75 * contrast + 0.25 * visibility, 0.0, 1.0))
    return contrast

def detect_release_idx(wrist_speed_norm: np.ndarray, n_frames: int, fps: float, search_start_frac: float, wrist_visibility: np.ndarray) -> tuple[int | None, float, str, int | None, float, int]:
    if not np.isfinite(fps) or fps <= 0:
        return (None, np.nan, 'release_not_evaluable_invalid_fps', None, np.nan, 0)
    start = max(1, int(math.floor(search_start_frac * n_frames)))
    if start >= n_frames - 1:
        return (None, np.nan, 'insufficient_release_search_window', None, np.nan, 0)
    search_signal = np.asarray(wrist_speed_norm[start:], dtype=float)
    local_peak = nanargmax_optional(search_signal)
    if local_peak is None:
        return (None, np.nan, 'no_finite_wrist_speed_in_release_window', None, np.nan, 0)
    peak_idx = start + int(local_peak)
    peak_value = value_at(wrist_speed_norm, peak_idx)
    if not np.isfinite(peak_value) or peak_value <= 0:
        return (None, np.nan, 'invalid_release_peak', peak_idx, np.nan, 0)
    vis = value_at(wrist_visibility, peak_idx)
    conf = event_signal_confidence(wrist_speed_norm[start:], peak_idx - start, visibility=vis)
    return (int(peak_idx), conf, 'max_event_smoothed_normalized_wrist_speed', int(peak_idx), 1.0, 1)

def detect_ffc_idx(stride_length_norm: np.ndarray, lead_heel_xy: np.ndarray, lead_foot_index_xy: np.ndarray, fps: float, release_idx: int | None, lookback_ms: float, heel_visibility: np.ndarray, foot_index_visibility: np.ndarray, stability_hold_ms: float, precontact_window_ms: float, ground_tolerance_foot_length: float) -> tuple[int | None, float, str, float, float, float, int, float, float, float, float]:
    if release_idx is None or not np.isfinite(fps) or fps <= 0:
        return (None, np.nan, 'ffc_not_evaluable', np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan)
    lookback_frames = max(3, int(round(lookback_ms * fps / 1000.0)))
    hold_frames = max(2, int(math.ceil(stability_hold_ms * fps / 1000.0)))
    pre_frames = max(2, int(math.ceil(precontact_window_ms * fps / 1000.0)))
    window_start = max(1, release_idx - lookback_frames)
    idxs = np.arange(window_start, release_idx)
    if len(idxs) < max(5, hold_frames + 1):
        return (None, np.nan, 'insufficient_ffc_search_window', np.nan, np.nan, np.nan, hold_frames, np.nan, np.nan, np.nan, np.nan)
    heel_y = np.asarray(lead_heel_xy[:, 1], dtype=float)
    toe_y = np.asarray(lead_foot_index_xy[:, 1], dtype=float)
    distal_foot_y = np.fmax(heel_y, toe_y)
    foot_length_arr = dist(lead_heel_xy, lead_foot_index_xy)
    finite_foot_length = foot_length_arr[idxs]
    finite_foot_length = finite_foot_length[np.isfinite(finite_foot_length)]
    if len(finite_foot_length) == 0:
        return (None, np.nan, 'invalid_lead_foot_length_reference', np.nan, np.nan, np.nan, hold_frames, np.nan, np.nan, np.nan, np.nan)
    foot_length_ref = float(np.nanmedian(finite_foot_length))
    if not np.isfinite(foot_length_ref) or foot_length_ref <= 0:
        return (None, np.nan, 'invalid_lead_foot_length_reference', np.nan, np.nan, np.nan, hold_frames, np.nan, np.nan, np.nan, np.nan)
    tolerance = float(max(1e-06, ground_tolerance_foot_length * foot_length_ref))
    planted_ref_frames = max(2, int(math.ceil(0.1 * fps)))
    planted_start = max(window_start, release_idx - planted_ref_frames)
    planted_segment = distal_foot_y[planted_start:release_idx + 1]
    planted_segment = planted_segment[np.isfinite(planted_segment)]
    if len(planted_segment) == 0:
        return (None, np.nan, 'invalid_planted_ground_reference', np.nan, np.nan, np.nan, hold_frames, foot_length_ref, tolerance, np.nan, np.nan)
    planted_y_ref = float(np.nanpercentile(planted_segment, 90))
    ground_distance = np.abs(distal_foot_y - planted_y_ref)
    ground_proximity = 1.0 - np.clip(ground_distance / max(2.0 * tolerance, 1e-09), 0.0, 1.0)
    dy = np.diff(distal_foot_y)
    signed_vertical_velocity = np.insert(dy * fps, 0, np.nan)
    vertical_speed = np.abs(signed_vertical_velocity)
    finite_stride = stride_length_norm[idxs]
    stride_ref = np.nanmax(finite_stride) if np.isfinite(finite_stride).any() else np.nan
    if not np.isfinite(stride_ref) or stride_ref <= 0:
        return (None, np.nan, 'invalid_stride_reference', np.nan, np.nan, np.nan, hold_frames, foot_length_ref, tolerance, planted_y_ref, np.nan)
    stride_score_all = np.clip(stride_length_norm / stride_ref, 0.0, 1.0)
    candidates: list[dict] = []
    for candidate in idxs:
        candidate = int(candidate)
        post_end = min(release_idx + 1, candidate + hold_frames)
        if post_end - candidate < hold_frames:
            continue
        post_dist = ground_distance[candidate:post_end]
        post_speed = vertical_speed[candidate:post_end]
        post_stride = stride_score_all[candidate:post_end]
        if not np.isfinite(post_dist).all() or not np.isfinite(post_stride).all():
            continue
        sustained_ground = bool(np.all(post_dist <= tolerance))
        sustained_stride = bool(np.all(post_stride >= 0.9))
        if not (sustained_ground and sustained_stride):
            continue
        pre_start = max(window_start, candidate - pre_frames)
        pre_dist = ground_distance[pre_start:candidate]
        pre_speed = vertical_speed[pre_start:candidate]
        pre_v_signed = signed_vertical_velocity[pre_start:candidate]
        pre_dist_med = float(np.nanmedian(pre_dist)) if np.isfinite(pre_dist).any() else np.nan
        post_dist_med = float(np.nanmedian(post_dist)) if np.isfinite(post_dist).any() else np.nan
        pre_speed_med = float(np.nanmedian(pre_speed)) if np.isfinite(pre_speed).any() else np.nan
        post_speed_med = float(np.nanmedian(post_speed)) if np.isfinite(post_speed).any() else np.nan
        pre_downward_fraction = float(np.nanmean(pre_v_signed > 0)) if np.isfinite(pre_v_signed).any() else np.nan
        distance_transition = float(np.clip((pre_dist_med - post_dist_med) / tolerance, 0.0, 2.0) / 2.0) if np.isfinite(pre_dist_med) and np.isfinite(post_dist_med) else 0.0
        speed_transition = 0.0
        if np.isfinite(pre_speed_med) and np.isfinite(post_speed_med):
            denom = max(pre_speed_med, 1e-09)
            speed_transition = float(np.clip((pre_speed_med - post_speed_med) / denom, 0.0, 1.0))
        downward_transition = float(np.clip(pre_downward_fraction, 0.0, 1.0)) if np.isfinite(pre_downward_fraction) else 0.0
        proximity_at_candidate = value_at(ground_proximity, candidate)
        stride_at_candidate = value_at(stride_score_all, candidate)
        transition_strength = 0.45 * distance_transition + 0.35 * speed_transition + 0.2 * downward_transition
        candidate_score = 0.55 * transition_strength + 0.25 * float(np.clip(proximity_at_candidate, 0.0, 1.0)) + 0.2 * float(np.clip(stride_at_candidate, 0.0, 1.0))
        previous_distance = value_at(ground_distance, candidate - 1) if candidate > window_start else np.nan
        crossed_ground_band = np.isfinite(previous_distance) and previous_distance > tolerance and (value_at(ground_distance, candidate) <= tolerance)
        pre_was_above_ground_band = np.isfinite(pre_dist_med) and pre_dist_med > 1.05 * tolerance
        approach_evidence = distance_transition >= 0.1 or speed_transition >= 0.1 or (np.isfinite(pre_downward_fraction) and pre_downward_fraction >= 0.33)
        entry_ok = bool(crossed_ground_band or pre_was_above_ground_band or approach_evidence)
        if not entry_ok:
            continue
        candidates.append({'idx': candidate, 'score': candidate_score, 'transition': transition_strength, 'proximity': proximity_at_candidate, 'stride': stride_at_candidate})
    if candidates:
        chosen = min(candidates, key=lambda c: c['idx'])
        selected_idx = int(chosen['idx'])
        method = 'first_sustained_distal_foot_ground_contact'
        contact_score = float(chosen['score'])
        transition_strength = float(chosen['transition'])
        ground_prox_at_event = float(chosen['proximity'])
        stride_at_event = float(chosen['stride'])
    else:
        valid = []
        for candidate in idxs:
            candidate = int(candidate)
            prox = value_at(ground_proximity, candidate)
            stride = value_at(stride_score_all, candidate)
            if np.isfinite(prox) and np.isfinite(stride) and (stride >= 0.9):
                score = 0.65 * prox + 0.35 * stride
                valid.append((score, candidate, prox, stride))
        if not valid:
            return (None, np.nan, 'no_finite_ffc_ground_contact_candidate', np.nan, np.nan, np.nan, hold_frames, foot_length_ref, tolerance, planted_y_ref, np.nan)
        _, selected_idx, ground_prox_at_event, stride_at_event = max(valid, key=lambda x: x[0])
        selected_idx = int(selected_idx)
        contact_score = float(0.65 * ground_prox_at_event + 0.35 * stride_at_event)
        transition_strength = np.nan
        method = 'max_ground_proximity_fallback'
    heel_vis = value_at(heel_visibility, selected_idx)
    toe_vis = value_at(foot_index_visibility, selected_idx)
    visible = [v for v in [heel_vis, toe_vis] if np.isfinite(v)]
    foot_visibility = float(np.mean(visible)) if visible else np.nan
    conf_components = [x for x in [contact_score, ground_prox_at_event, stride_at_event, transition_strength, foot_visibility] if np.isfinite(x)]
    conf = float(np.mean(conf_components)) if conf_components else np.nan
    if method.endswith('fallback') and np.isfinite(conf):
        conf *= 0.85
    return (selected_idx, conf, method, contact_score, stride_at_event, ground_prox_at_event, hold_frames, foot_length_ref, tolerance, planted_y_ref, transition_strength)

def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    md = pd.read_csv(metadata_path)
    if 'video_file' not in md.columns:
        raise ValueError("Metadata CSV must contain a 'video_file' column.")
    return md

def safe_timestamp_ms(df: pd.DataFrame, idx: int | None, fps: float) -> float:
    if idx is None:
        return np.nan
    if 'timestamp_ms' in df.columns:
        ts = pd.to_numeric(df['timestamp_ms'], errors='coerce')
        if idx < len(ts) and np.isfinite(ts.iloc[idx]):
            return float(ts.iloc[idx])
    if np.isfinite(fps) and fps > 0:
        return float(idx * 1000.0 / fps)
    return np.nan

def process_file(csv_path: Path, metadata_row: dict, default_throwing_side: str, *, max_interp_gap_ms: float, savgol_window_ms: float, event_savgol_window_ms: float, ffc_event_savgol_window_ms: float, ffc_lookback_ms: float, ffc_stability_hold_ms: float, ffc_precontact_window_ms: float, ffc_ground_tolerance_foot_length: float, release_search_start_frac: float) -> dict:
    df = pd.read_csv(csv_path)
    n_frames = len(df)
    throwing_side = metadata_row.get('throwing_side', default_throwing_side)
    camera_view = metadata_row.get('camera_view', '')
    height_m = metadata_row.get('height_m', np.nan)
    side = choose_side_map(throwing_side)
    fps, fps_valid, fps_provenance = resolve_fps(df)
    max_gap_frames = max(0, int(round(max_interp_gap_ms * fps / 1000.0))) if fps_valid else 0
    smooth_window_frames = odd_window_from_ms(savgol_window_ms, fps) if fps_valid else 5
    event_smooth_window_frames = odd_window_from_ms(event_savgol_window_ms, fps, minimum=3) if fps_valid else 3
    ffc_event_smooth_window_frames = odd_window_from_ms(ffc_event_savgol_window_ms, fps, minimum=3) if fps_valid else 3
    raw_landmarks: dict[str, np.ndarray] = {}
    proc_landmarks: dict[str, np.ndarray] = {}
    gap_runs: dict[str, int] = {}
    needed_landmarks = set(['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle', *side.values()])
    for lm in sorted(needed_landmarks):
        raw = get_xy(df, lm)
        raw_landmarks[lm] = raw
        proc, gap = preprocess_xy(raw, max_gap_frames=max_gap_frames, smooth_window_frames=smooth_window_frames)
        proc_landmarks[lm] = proc
        gap_runs[lm] = gap
    ls, rs = (proc_landmarks['left_shoulder'], proc_landmarks['right_shoulder'])
    lh, rh = (proc_landmarks['left_hip'], proc_landmarks['right_hip'])
    la, ra = (proc_landmarks['left_ankle'], proc_landmarks['right_ankle'])
    ts = proc_landmarks[side['throw_shoulder']]
    te = proc_landmarks[side['throw_elbow']]
    tw = proc_landmarks[side['throw_wrist']]
    lead_hip = proc_landmarks[side['lead_hip']]
    lead_knee = proc_landmarks[side['lead_knee']]
    lead_ankle = proc_landmarks[side['lead_ankle']]
    lead_heel = proc_landmarks[side['lead_heel']]
    trail_hip = proc_landmarks[side['trail_hip']]
    trail_knee = proc_landmarks[side['trail_knee']]
    trail_ankle = proc_landmarks[side['trail_ankle']]
    shoulder_mid = midpoint(ls, rs)
    hip_mid = midpoint(lh, rh)
    ankle_mid = midpoint(la, ra)
    body_height_proxy = stat_mean(np.array([np.nanmedian(dist(shoulder_mid, ankle_mid))]))
    shoulder_width_proxy = stat_mean(np.array([np.nanmedian(dist(ls, rs))]))
    hip_width_proxy = stat_mean(np.array([np.nanmedian(dist(lh, rh))]))
    if not np.isfinite(body_height_proxy) or body_height_proxy <= 0:
        body_height_proxy = np.nan
    if not np.isfinite(shoulder_width_proxy) or shoulder_width_proxy <= 0:
        shoulder_width_proxy = np.nan
    if not np.isfinite(hip_width_proxy) or hip_width_proxy <= 0:
        hip_width_proxy = np.nan
    elbow_angle = angle_2d(ts, te, tw)
    lead_knee_angle = angle_2d(lead_hip, lead_knee, lead_ankle)
    trail_knee_angle = angle_2d(trail_hip, trail_knee, trail_ankle)
    elbow_ang_vel = angular_speed_abs_per_second(elbow_angle, fps) if fps_valid else np.full(n_frames, np.nan)
    wrist_speed_raw = speed_per_second(tw, fps) if fps_valid else np.full(n_frames, np.nan)
    if np.isfinite(body_height_proxy) and body_height_proxy > 0:
        wrist_speed_norm = wrist_speed_raw / body_height_proxy
    else:
        wrist_speed_norm = np.full(n_frames, np.nan)
    trunk_tilt = signed_trunk_tilt_deg(hip_mid, shoulder_mid)
    trunk_tilt_vel = angular_speed_abs_per_second(trunk_tilt, fps) if fps_valid else np.full(n_frames, np.nan)
    shoulder_line_angle = line_angle_deg(ls, rs)
    hip_line_angle = line_angle_deg(lh, rh)
    hip_shoulder_sep_signed = wrap_signed_180(shoulder_line_angle - hip_line_angle)
    hip_shoulder_sep = np.abs(hip_shoulder_sep_signed)
    hip_shoulder_sep_vel = angular_speed_abs_per_second(hip_shoulder_sep_signed, fps) if fps_valid else np.full(n_frames, np.nan)
    upper_arm_angle = line_angle_deg(ts, te)
    forearm_angle = line_angle_deg(te, tw)
    stride_length = dist(lead_ankle, trail_ankle)
    stride_length_norm = stride_length / body_height_proxy if np.isfinite(body_height_proxy) and body_height_proxy > 0 else np.full(n_frames, np.nan)
    throw_wrist_vis_arr = get_vis(df, side['throw_wrist'])
    lead_heel_vis_arr = get_vis(df, side['lead_heel'])
    lead_foot_index_vis_arr = get_vis(df, side['lead_foot_index'])

    def event_preprocess(lm_name: str, window_frames: int) -> np.ndarray:
        raw = raw_landmarks[lm_name]
        interp = interpolate_short_internal_gaps_2d(raw, max_gap_frames=max_gap_frames)
        out = interp.copy()
        for axis in range(out.shape[1]):
            out[:, axis] = smooth_1d_preserve_long_gaps(out[:, axis], window_frames=window_frames, poly=2)
        return out
    event_tw = event_preprocess(side['throw_wrist'], event_smooth_window_frames)
    event_lead_heel = event_preprocess(side['lead_heel'], ffc_event_smooth_window_frames)
    event_lead_foot_index = event_preprocess(side['lead_foot_index'], ffc_event_smooth_window_frames)
    event_lead_ankle = event_preprocess(side['lead_ankle'], ffc_event_smooth_window_frames)
    event_trail_ankle = event_preprocess(side['trail_ankle'], ffc_event_smooth_window_frames)
    event_wrist_speed_raw = speed_per_second(event_tw, fps) if fps_valid else np.full(n_frames, np.nan)
    event_wrist_speed_norm = event_wrist_speed_raw / body_height_proxy if np.isfinite(body_height_proxy) and body_height_proxy > 0 else np.full(n_frames, np.nan)
    event_stride_length = dist(event_lead_ankle, event_trail_ankle)
    event_stride_length_norm = event_stride_length / body_height_proxy if np.isfinite(body_height_proxy) and body_height_proxy > 0 else np.full(n_frames, np.nan)
    release_idx = None
    release_conf = np.nan
    release_method = 'not_evaluable_invalid_fps'
    release_peak_idx = None
    release_event_fraction = np.nan
    release_hold_frames = 0
    if fps_valid:
        release_idx, release_conf, release_method, release_peak_idx, release_event_fraction, release_hold_frames = detect_release_idx(event_wrist_speed_norm, n_frames, fps, release_search_start_frac, throw_wrist_vis_arr)
    ffc_idx, ffc_conf, ffc_method, ffc_score, ffc_stride_score, ffc_ground_proximity_score, ffc_hold_frames, ffc_foot_length_ref, ffc_ground_tolerance, ffc_planted_y_ref, ffc_transition_strength = detect_ffc_idx(event_stride_length_norm, event_lead_heel, event_lead_foot_index, fps, release_idx, ffc_lookback_ms, lead_heel_vis_arr, lead_foot_index_vis_arr, ffc_stability_hold_ms, ffc_precontact_window_ms, ffc_ground_tolerance_foot_length)
    sequence_ok = bool(ffc_idx is not None and release_idx is not None and (ffc_idx < release_idx))
    if sequence_ok:
        window = slice(ffc_idx, release_idx + 1)
        event_window_frames = int(release_idx - ffc_idx)
        ffc_time_ms = safe_timestamp_ms(df, ffc_idx, fps)
        release_time_ms = safe_timestamp_ms(df, release_idx, fps)
        ffc_to_release_ms = float(release_time_ms - ffc_time_ms) if np.isfinite(ffc_time_ms) and np.isfinite(release_time_ms) else float(event_window_frames * 1000.0 / fps)
    else:
        window = slice(0, 0)
        event_window_frames = np.nan
        ffc_time_ms = np.nan
        release_time_ms = np.nan
        ffc_to_release_ms = np.nan
    max_sep_local = nanargmax_optional(hip_shoulder_sep[window]) if sequence_ok else None
    max_elbow_flex_local = nanargmin_optional(elbow_angle[window]) if sequence_ok else None
    max_trunk_vel_local = nanargmax_optional(trunk_tilt_vel[window]) if sequence_ok else None
    max_sep_idx = ffc_idx + max_sep_local if sequence_ok and max_sep_local is not None else None
    max_elbow_flex_idx = ffc_idx + max_elbow_flex_local if sequence_ok and max_elbow_flex_local is not None else None
    max_trunk_tilt_vel_idx = ffc_idx + max_trunk_vel_local if sequence_ok and max_trunk_vel_local is not None else None

    def timing_to_release(event_idx: int | None) -> float:
        if release_idx is None or event_idx is None or (not fps_valid):
            return np.nan
        return float((release_idx - event_idx) * 1000.0 / fps)
    throw_shoulder_vis = stat_mean(get_vis(df, side['throw_shoulder']))
    throw_elbow_vis = stat_mean(get_vis(df, side['throw_elbow']))
    throw_wrist_vis = stat_mean(throw_wrist_vis_arr)
    lead_heel_vis = stat_mean(lead_heel_vis_arr)
    lead_ankle_vis = stat_mean(get_vis(df, side['lead_ankle']))
    trail_ankle_vis = stat_mean(get_vis(df, side['trail_ankle']))
    critical_landmarks = [side['throw_shoulder'], side['throw_elbow'], side['throw_wrist'], side['lead_hip'], side['lead_knee'], side['lead_ankle'], side['lead_heel'], side['trail_ankle'], 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    critical_max_gap_frames = max((gap_runs.get(lm, 0) for lm in critical_landmarks), default=0)
    critical_max_gap_ms = float(critical_max_gap_frames * 1000.0 / fps) if fps_valid else np.nan
    processing_status = 'ok' if fps_valid and sequence_ok else 'not_eligible'
    exclusion_reasons: list[str] = []
    if not fps_valid:
        exclusion_reasons.append('invalid_or_unverified_fallback_fps')
    if release_idx is None:
        exclusion_reasons.append('release_not_detected')
    if ffc_idx is None:
        exclusion_reasons.append('ffc_not_detected')
    if not sequence_ok:
        exclusion_reasons.append('invalid_event_sequence')
    base = {'video_file': df['video_file'].iloc[0] if 'video_file' in df.columns else csv_path.name, 'pitcher_id': metadata_row.get('pitcher_id', df['pitcher_id'].iloc[0] if 'pitcher_id' in df.columns else np.nan), 'pitch_id': df['pitch_id'].iloc[0] if 'pitch_id' in df.columns else np.nan, 'throwing_side': throwing_side, 'camera_view': camera_view, 'height_m': height_m, 'n_frames': int(n_frames), 'fps': fps, 'fps_valid_for_time_derivatives': int(bool(fps_valid)), 'fps_provenance': fps_provenance, 'pose_detected_rate': float(pd.to_numeric(df['pose_detected'], errors='coerce').mean()) if 'pose_detected' in df.columns else np.nan, 'processing_status': processing_status, 'processing_exclusion_reason': ';'.join(exclusion_reasons), 'max_interp_gap_ms_config': float(max_interp_gap_ms), 'max_interp_gap_frames_config': int(max_gap_frames), 'critical_landmark_longest_raw_gap_frames': int(critical_max_gap_frames), 'critical_landmark_longest_raw_gap_ms': critical_max_gap_ms, 'savgol_window_ms_config': float(savgol_window_ms), 'savgol_window_frames_used': int(smooth_window_frames), 'event_savgol_window_ms_config': float(event_savgol_window_ms), 'event_savgol_window_frames_used': int(event_smooth_window_frames), 'ffc_event_savgol_window_ms_config': float(ffc_event_savgol_window_ms), 'ffc_event_savgol_window_frames_used': int(ffc_event_smooth_window_frames), 'ffc_lookback_ms_config': float(ffc_lookback_ms), 'ffc_stability_hold_ms_config': float(ffc_stability_hold_ms), 'ffc_stability_hold_frames_used': int(ffc_hold_frames), 'ffc_precontact_window_ms_config': float(ffc_precontact_window_ms), 'ffc_ground_tolerance_foot_length_config': float(ffc_ground_tolerance_foot_length), 'release_search_start_frac': float(release_search_start_frac), 'release_hold_frames_used': int(release_hold_frames), 'body_height_proxy': body_height_proxy, 'shoulder_width_proxy': shoulder_width_proxy, 'hip_width_proxy': hip_width_proxy, 'ffc_frame_idx': float(ffc_idx) if ffc_idx is not None else np.nan, 'release_frame_idx': float(release_idx) if release_idx is not None else np.nan, 'ffc_time_ms': ffc_time_ms, 'release_time_ms': release_time_ms, 'ffc_to_release_ms': ffc_to_release_ms, 'event_window_frames': event_window_frames, 'event_window_ms': ffc_to_release_ms, 'sequence_ok': int(sequence_ok), 'ffc_confidence': ffc_conf, 'release_confidence': release_conf, 'ffc_detection_method': ffc_method, 'release_detection_method': release_method, 'ffc_composite_score': ffc_score, 'ffc_stride_score_at_event': ffc_stride_score, 'ffc_ground_proximity_score_at_event': ffc_ground_proximity_score, 'ffc_transition_strength': ffc_transition_strength, 'ffc_foot_length_reference': ffc_foot_length_ref, 'ffc_ground_tolerance_image_units': ffc_ground_tolerance, 'ffc_planted_y_reference': ffc_planted_y_ref, 'release_peak_frame_idx': float(release_peak_idx) if release_peak_idx is not None else np.nan, 'release_event_to_peak_offset_frames': float(release_peak_idx - release_idx) if release_peak_idx is not None and release_idx is not None else np.nan, 'release_event_speed_fraction_of_peak': release_event_fraction}
    if not sequence_ok:
        feature_values = {'stride_length_norm_at_ffc': np.nan, 'stride_length_norm_at_release': np.nan, 'stride_length_norm_max_ffc_to_release': np.nan, 'elbow_angle_deg_at_ffc': np.nan, 'elbow_angle_deg_at_release': np.nan, 'elbow_angle_deg_mean_ffc_to_release': np.nan, 'elbow_angle_deg_range_ffc_to_release': np.nan, 'elbow_ang_vel_max_ffc_to_release': np.nan, 'elbow_ang_vel_mean_ffc_to_release': np.nan, 'wrist_speed_image_per_s_max_ffc_to_release': np.nan, 'wrist_speed_image_per_s_mean_ffc_to_release': np.nan, 'wrist_speed_norm_bh_per_s_max_ffc_to_release': np.nan, 'wrist_speed_norm_bh_per_s_mean_ffc_to_release': np.nan, 'wrist_speed_norm_bh_per_s_at_release': np.nan, 'wrist_speed_max_ffc_to_release': np.nan, 'wrist_speed_mean_ffc_to_release': np.nan, 'wrist_speed_at_release': np.nan, 'upper_arm_angle_deg_at_release': np.nan, 'forearm_angle_deg_at_release': np.nan, 'trunk_tilt_deg_at_ffc': np.nan, 'trunk_tilt_deg_at_release': np.nan, 'trunk_tilt_deg_mean_ffc_to_release': np.nan, 'trunk_tilt_deg_range_ffc_to_release': np.nan, 'trunk_tilt_vel_max_ffc_to_release': np.nan, 'hip_shoulder_sep_deg_at_ffc': np.nan, 'hip_shoulder_sep_deg_at_release': np.nan, 'hip_shoulder_sep_deg_max_ffc_to_release': np.nan, 'hip_shoulder_sep_deg_mean_ffc_to_release': np.nan, 'hip_shoulder_sep_vel_max_ffc_to_release': np.nan, 'lead_knee_angle_deg_at_ffc': np.nan, 'lead_knee_angle_deg_at_release': np.nan, 'lead_knee_extension_deg_ffc_to_release': np.nan, 'trail_knee_angle_deg_at_ffc': np.nan, 'trail_knee_angle_deg_at_release': np.nan, 'max_sep_frame_idx': np.nan, 'max_sep_to_release_ms': np.nan, 'max_elbow_flex_frame_idx': np.nan, 'max_elbow_flex_to_release_ms': np.nan, 'max_trunk_tilt_vel_frame_idx': np.nan, 'max_trunk_tilt_vel_to_release_ms': np.nan}
    else:
        feature_values = {'stride_length_norm_at_ffc': value_at(stride_length_norm, ffc_idx), 'stride_length_norm_at_release': value_at(stride_length_norm, release_idx), 'stride_length_norm_max_ffc_to_release': stat_max(stride_length_norm[window]), 'elbow_angle_deg_at_ffc': value_at(elbow_angle, ffc_idx), 'elbow_angle_deg_at_release': value_at(elbow_angle, release_idx), 'elbow_angle_deg_mean_ffc_to_release': stat_mean(elbow_angle[window]), 'elbow_angle_deg_range_ffc_to_release': stat_range(elbow_angle[window]), 'elbow_ang_vel_max_ffc_to_release': stat_max(elbow_ang_vel[window]), 'elbow_ang_vel_mean_ffc_to_release': stat_mean(elbow_ang_vel[window]), 'wrist_speed_image_per_s_max_ffc_to_release': stat_max(wrist_speed_raw[window]), 'wrist_speed_image_per_s_mean_ffc_to_release': stat_mean(wrist_speed_raw[window]), 'wrist_speed_norm_bh_per_s_max_ffc_to_release': stat_max(wrist_speed_norm[window]), 'wrist_speed_norm_bh_per_s_mean_ffc_to_release': stat_mean(wrist_speed_norm[window]), 'wrist_speed_norm_bh_per_s_at_release': value_at(wrist_speed_norm, release_idx), 'wrist_speed_max_ffc_to_release': stat_max(wrist_speed_norm[window]), 'wrist_speed_mean_ffc_to_release': stat_mean(wrist_speed_norm[window]), 'wrist_speed_at_release': value_at(wrist_speed_norm, release_idx), 'upper_arm_angle_deg_at_release': value_at(upper_arm_angle, release_idx), 'forearm_angle_deg_at_release': value_at(forearm_angle, release_idx), 'trunk_tilt_deg_at_ffc': value_at(trunk_tilt, ffc_idx), 'trunk_tilt_deg_at_release': value_at(trunk_tilt, release_idx), 'trunk_tilt_deg_mean_ffc_to_release': stat_mean(trunk_tilt[window]), 'trunk_tilt_deg_range_ffc_to_release': stat_range(trunk_tilt[window]), 'trunk_tilt_vel_max_ffc_to_release': stat_max(trunk_tilt_vel[window]), 'hip_shoulder_sep_deg_at_ffc': value_at(hip_shoulder_sep, ffc_idx), 'hip_shoulder_sep_deg_at_release': value_at(hip_shoulder_sep, release_idx), 'hip_shoulder_sep_deg_max_ffc_to_release': stat_max(hip_shoulder_sep[window]), 'hip_shoulder_sep_deg_mean_ffc_to_release': stat_mean(hip_shoulder_sep[window]), 'hip_shoulder_sep_vel_max_ffc_to_release': stat_max(hip_shoulder_sep_vel[window]), 'lead_knee_angle_deg_at_ffc': value_at(lead_knee_angle, ffc_idx), 'lead_knee_angle_deg_at_release': value_at(lead_knee_angle, release_idx), 'lead_knee_extension_deg_ffc_to_release': value_at(lead_knee_angle, release_idx) - value_at(lead_knee_angle, ffc_idx), 'trail_knee_angle_deg_at_ffc': value_at(trail_knee_angle, ffc_idx), 'trail_knee_angle_deg_at_release': value_at(trail_knee_angle, release_idx), 'max_sep_frame_idx': float(max_sep_idx) if max_sep_idx is not None else np.nan, 'max_sep_to_release_ms': timing_to_release(max_sep_idx), 'max_elbow_flex_frame_idx': float(max_elbow_flex_idx) if max_elbow_flex_idx is not None else np.nan, 'max_elbow_flex_to_release_ms': timing_to_release(max_elbow_flex_idx), 'max_trunk_tilt_vel_frame_idx': float(max_trunk_tilt_vel_idx) if max_trunk_tilt_vel_idx is not None else np.nan, 'max_trunk_tilt_vel_to_release_ms': timing_to_release(max_trunk_tilt_vel_idx)}
    visibility = {'throw_shoulder_visibility_mean': throw_shoulder_vis, 'throw_elbow_visibility_mean': throw_elbow_vis, 'throw_wrist_visibility_mean': throw_wrist_vis, 'lead_heel_visibility_mean': lead_heel_vis, 'lead_ankle_visibility_mean': lead_ankle_vis, 'trail_ankle_visibility_mean': trail_ankle_vis}
    return {**base, **feature_values, **visibility}

def main() -> None:
    parser = argparse.ArgumentParser(description='Extract pitch-level 2D biomechanical features with time-normalized processing.')
    parser.add_argument('--pose-dir', required=True, help='Directory containing *_pose.csv files')
    parser.add_argument('--out', required=True, help='Output feature CSV')
    parser.add_argument('--metadata', default='', help='Optional metadata CSV with video_file column')
    parser.add_argument('--default-throwing-side', default='right', choices=['right', 'left'])
    parser.add_argument('--max-interp-gap-ms', type=float, default=100.0, help='Maximum internal landmark gap to interpolate, in milliseconds. Default: 100 ms.')
    parser.add_argument('--savgol-window-ms', type=float, default=200.0, help='Kinematic Savitzky-Golay smoothing duration. Default: 200 ms.')
    parser.add_argument('--event-savgol-window-ms', type=float, default=100.0, help='Event smoothing duration used for release localization. Default: 100 ms.')
    parser.add_argument('--ffc-event-savgol-window-ms', type=float, default=50.0, help='Dedicated shorter smoothing duration used for FFC localization. Default: 50 ms.')
    parser.add_argument('--ffc-lookback-ms', type=float, default=600.0, help='Time before estimated release searched for FFC. Default: 600 ms.')
    parser.add_argument('--ffc-stability-hold-ms', type=float, default=50.0, help='Required duration of sustained near-ground lead-foot state. Default: 50 ms.')
    parser.add_argument('--ffc-precontact-window-ms', type=float, default=100.0, help='Pre-candidate window used to quantify the foot-to-ground transition. Default: 100 ms.')
    parser.add_argument('--ffc-ground-tolerance-foot-length', type=float, default=0.15, help='Ground-proximity tolerance as a fraction of lead-foot length. Default: 0.15.')
    parser.add_argument('--release-search-start-frac', type=float, default=0.45, help='Fraction of clip before which release is not searched. Default: 0.45.')
    args = parser.parse_args()
    if not 0.0 <= args.release_search_start_frac < 1.0:
        raise ValueError('--release-search-start-frac must be in [0,1).')
    if args.ffc_ground_tolerance_foot_length <= 0:
        raise ValueError('--ffc-ground-tolerance-foot-length must be > 0.')
    positive_args = [args.savgol_window_ms, args.event_savgol_window_ms, args.ffc_event_savgol_window_ms, args.ffc_lookback_ms, args.ffc_stability_hold_ms, args.ffc_precontact_window_ms]
    if args.max_interp_gap_ms < 0 or any((x <= 0 for x in positive_args)):
        raise ValueError('Time-window arguments must be positive (max interp gap may be 0).')
    pose_dir = Path(args.pose_dir)
    out_path = Path(args.out)
    files = sorted(pose_dir.glob('*_pose.csv'))
    if not files:
        raise FileNotFoundError(f'No *_pose.csv files found in {pose_dir}')
    metadata_df = load_metadata(args.metadata) if args.metadata else None
    rows = []
    for f in files:
        df_head = pd.read_csv(f, nrows=1)
        real_video_file = df_head['video_file'].iloc[0] if 'video_file' in df_head.columns else f.name.replace('_pose.csv', '')
        if metadata_df is not None:
            matches = metadata_df[metadata_df['video_file'] == real_video_file]
            meta = matches.iloc[0].to_dict() if len(matches) else {'video_file': real_video_file, 'throwing_side': args.default_throwing_side}
        else:
            meta = {'video_file': real_video_file, 'throwing_side': args.default_throwing_side}
        rows.append(process_file(f, meta, args.default_throwing_side, max_interp_gap_ms=args.max_interp_gap_ms, savgol_window_ms=args.savgol_window_ms, event_savgol_window_ms=args.event_savgol_window_ms, ffc_event_savgol_window_ms=args.ffc_event_savgol_window_ms, ffc_lookback_ms=args.ffc_lookback_ms, ffc_stability_hold_ms=args.ffc_stability_hold_ms, ffc_precontact_window_ms=args.ffc_precontact_window_ms, ffc_ground_tolerance_foot_length=args.ffc_ground_tolerance_foot_length, release_search_start_frac=args.release_search_start_frac))
    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')
    print(f'Processed {len(out_df)} pitch files.')
    cols = ['video_file', 'fps', 'fps_provenance', 'processing_status', 'ffc_frame_idx', 'release_frame_idx', 'ffc_to_release_ms', 'ffc_confidence', 'release_confidence', 'wrist_speed_norm_bh_per_s_max_ffc_to_release']
    print(out_df[[c for c in cols if c in out_df.columns]].head().to_string(index=False))
if __name__ == '__main__':
    main()
