#!/usr/bin/env python
"""
run_pipeline.py

Validated pitching-biomechanics pipeline.

Runs acquisition -> feature extraction -> QC -> relative mechanical characterization
using event-localization rules evaluated after blinded development and an independent
holdout evaluation.

Validated event-processing defaults:
- maximum interpolated internal gap: 100 ms
- kinematic Savitzky-Golay smoothing: 200 ms
- release-event smoothing: 100 ms
- FFC-event smoothing: 50 ms
- FFC lookback: 600 ms
- FFC sustained-contact hold: 50 ms
- FFC pre-contact transition window: 100 ms
- FFC ground tolerance: 0.15 x lead-foot length
- release search start: 45% of clip

Do not tune these values after the independent holdout evaluation when attempting
to reproduce the associated study.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
POSE_SCRIPT = SCRIPT_DIR / "extract_pose.py"
FEATURE_SCRIPT = SCRIPT_DIR / "extract_features.py"
QC_SCRIPT = SCRIPT_DIR / "quality_control.py"
RELATIVE_MECHANICS_SCRIPT = SCRIPT_DIR / "relative_mechanics.py"
VIDEO_DIR = PROJECT_ROOT / "videos"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"
METADATA_CSV = PROJECT_ROOT / "metadata" / "pitch_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
POSE_DIR = OUTPUT_DIR / "pose"
FEATURES_DIR = OUTPUT_DIR / "features"
LOGS_DIR = OUTPUT_DIR / "logs"
FEATURES_CSV = FEATURES_DIR / "pitch_features_master.csv"
QC_ANNOTATED_CSV = FEATURES_DIR / "pitch_features_qc_annotated.csv"
RELATIVE_MECHANICS_CSV = FEATURES_DIR / "pitch_features_with_relative_mechanics.csv"
MANIFEST_CSV = LOGS_DIR / "pipeline_manifest.csv"
QC_REPORT = LOGS_DIR / "features_qc_report.txt"
QC_FLOW = LOGS_DIR / "features_qc_flow.csv"
PARAMETERS_JSON = LOGS_DIR / "pipeline_parameters.json"
ENVIRONMENT_TXT = LOGS_DIR / "software_environment.txt"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}

VALIDATED_DEFAULTS = {
    "max_interp_gap_ms": 100.0,
    "savgol_window_ms": 200.0,
    "event_savgol_window_ms": 100.0,
    "ffc_event_savgol_window_ms": 50.0,
    "ffc_lookback_ms": 600.0,
    "ffc_stability_hold_ms": 50.0,
    "ffc_precontact_window_ms": 100.0,
    "ffc_ground_tolerance_foot_length": 0.15,
    "release_search_start_frac": 0.45,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the validated 2D pitching-biomechanics pipeline through relative mechanical characterization.")
    p.add_argument("--reuse-pose", action="store_true", help="Reuse existing outputs/pose/*_pose.csv files instead of rerunning MediaPipe.")
    p.add_argument("--max-interp-gap-ms", type=float, default=VALIDATED_DEFAULTS["max_interp_gap_ms"])
    p.add_argument("--savgol-window-ms", type=float, default=VALIDATED_DEFAULTS["savgol_window_ms"])
    p.add_argument("--event-savgol-window-ms", type=float, default=VALIDATED_DEFAULTS["event_savgol_window_ms"])
    p.add_argument("--ffc-event-savgol-window-ms", type=float, default=VALIDATED_DEFAULTS["ffc_event_savgol_window_ms"])
    p.add_argument("--ffc-lookback-ms", type=float, default=VALIDATED_DEFAULTS["ffc_lookback_ms"])
    p.add_argument("--ffc-stability-hold-ms", type=float, default=VALIDATED_DEFAULTS["ffc_stability_hold_ms"])
    p.add_argument("--ffc-precontact-window-ms", type=float, default=VALIDATED_DEFAULTS["ffc_precontact_window_ms"])
    p.add_argument("--ffc-ground-tolerance-foot-length", type=float, default=VALIDATED_DEFAULTS["ffc_ground_tolerance_foot_length"])
    p.add_argument("--release-search-start-frac", type=float, default=VALIDATED_DEFAULTS["release_search_start_frac"])
    return p.parse_args()


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def validate_required_paths(reuse_pose: bool) -> None:
    required = {"Feature script": FEATURE_SCRIPT, "QC script": QC_SCRIPT, "Relative mechanics script": RELATIVE_MECHANICS_SCRIPT, "Video directory": VIDEO_DIR, "Metadata CSV": METADATA_CSV}
    if not reuse_pose:
        required["Pose script"] = POSE_SCRIPT
        required["MediaPipe model"] = MODEL_PATH
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required validated-pipeline paths are missing:\n- " + "\n- ".join(missing))


def find_videos() -> List[Path]:
    return sorted(p for p in VIDEO_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, shell=False)


def command_string(cmd: List[str]) -> str:
    return " ".join(f'"{x}"' if " " in x else x for x in cmd)


def write_manifest(rows: List[Dict[str, object]]) -> None:
    fields = ["video_name", "video_path", "pose_csv", "status", "return_code", "duration_sec", "stdout", "stderr"]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "not detected"


def write_environment_archive() -> None:
    packages = ["mediapipe", "opencv-python", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
    lines = ["VALIDATED COMPUTATIONAL ENVIRONMENT", "=" * 48, f"Python executable: {sys.executable}", f"Python version   : {platform.python_version()}", f"Platform         : {platform.platform()}", "", "Package versions:"]
    for pkg in packages:
        lines.append(f"{pkg}: {package_version(pkg)}")
    ENVIRONMENT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parameter_record(args: argparse.Namespace) -> dict:
    return {
        "pipeline_status": "validated_after_holdout_evaluation",
        "pose_script": POSE_SCRIPT.name,
        "feature_script": FEATURE_SCRIPT.name,
        "qc_script": QC_SCRIPT.name,
        "relative_mechanics_script": RELATIVE_MECHANICS_SCRIPT.name,
        "reuse_pose": bool(args.reuse_pose),
        "parameters": {
            "max_interp_gap_ms": args.max_interp_gap_ms,
            "savgol_window_ms": args.savgol_window_ms,
            "event_savgol_window_ms": args.event_savgol_window_ms,
            "ffc_event_savgol_window_ms": args.ffc_event_savgol_window_ms,
            "ffc_lookback_ms": args.ffc_lookback_ms,
            "ffc_stability_hold_ms": args.ffc_stability_hold_ms,
            "ffc_precontact_window_ms": args.ffc_precontact_window_ms,
            "ffc_ground_tolerance_foot_length": args.ffc_ground_tolerance_foot_length,
            "release_search_start_frac": args.release_search_start_frac,
        },
        "holdout_validation_reference": {
            "n_pitches": 25,
            "ffc_mae_frames": 0.80,
            "ffc_mae_ms": 23.33,
            "release_mae_frames": 0.72,
            "release_mae_ms": 24.00,
            "ffc_to_release_mae_frames": 1.12,
            "ffc_to_release_mae_ms": 34.00,
        },
    }


def warn_if_parameters_changed(args: argparse.Namespace) -> None:
    current = {k: getattr(args, k) for k in ["max_interp_gap_ms", "savgol_window_ms", "event_savgol_window_ms", "ffc_event_savgol_window_ms", "ffc_lookback_ms", "ffc_stability_hold_ms", "ffc_precontact_window_ms", "ffc_ground_tolerance_foot_length", "release_search_start_frac"]}
    changed = {k: (VALIDATED_DEFAULTS[k], current[k]) for k in VALIDATED_DEFAULTS if current[k] != VALIDATED_DEFAULTS[k]}
    if changed:
        print("\n" + "!" * 72)
        print("WARNING: VALIDATED PARAMETERS WERE OVERRIDDEN")
        for key, (validated, supplied) in changed.items():
            print(f"  {key}: validated={validated} supplied={supplied}")
        print("This should not be used as the validated study run unless the methodological change is intentionally documented.")
        print("!" * 72 + "\n")


def main() -> None:
    args = parse_args()
    ensure_directories(OUTPUT_DIR, POSE_DIR, FEATURES_DIR, LOGS_DIR)
    validate_required_paths(args.reuse_pose)
    warn_if_parameters_changed(args)
    videos = find_videos()
    if not videos:
        raise FileNotFoundError(f"No supported videos found in {VIDEO_DIR}")

    manifest_rows: List[Dict[str, object]] = []
    print("=" * 72)
    print("VALIDATED PITCHING BIOMECHANICS PIPELINE")
    print("=" * 72)
    print(f"Videos found: {len(videos)}")
    print(f"Feature extractor: {FEATURE_SCRIPT.name}")
    print(f"Reuse pose: {args.reuse_pose}")
    print("=" * 72)

    if args.reuse_pose:
        missing_pose = []
        for video in videos:
            pose_csv = POSE_DIR / f"{video.stem}_pose.csv"
            if not pose_csv.exists():
                missing_pose.append(str(pose_csv)); status = "missing_existing_pose"; return_code = -1
            else:
                status = "reused_existing_pose"; return_code = 0
            manifest_rows.append({"video_name": video.name, "video_path": str(video), "pose_csv": str(pose_csv), "status": status, "return_code": return_code, "duration_sec": 0.0, "stdout": "", "stderr": ""})
        if missing_pose:
            write_manifest(manifest_rows)
            raise FileNotFoundError("Cannot use --reuse-pose because pose CSVs are missing:\n- " + "\n- ".join(missing_pose))
    else:
        for i, video in enumerate(videos, start=1):
            pose_csv = POSE_DIR / f"{video.stem}_pose.csv"
            cmd = [sys.executable, str(POSE_SCRIPT), "--video", str(video), "--model", str(MODEL_PATH), "--out", str(pose_csv)]
            print(f"[{i}/{len(videos)}] {video.name}")
            t0 = time.time(); result = run_command(cmd); elapsed = round(time.time() - t0, 2)
            status = "success" if result.returncode == 0 and pose_csv.exists() else "failed"
            manifest_rows.append({"video_name": video.name, "video_path": str(video), "pose_csv": str(pose_csv), "status": status, "return_code": result.returncode, "duration_sec": elapsed, "stdout": (result.stdout or "").strip(), "stderr": (result.stderr or "").strip()})
            if result.stdout: print(result.stdout)
            if result.stderr: print(result.stderr)
    write_manifest(manifest_rows)
    successful_pose_files = [Path(row["pose_csv"]) for row in manifest_rows if row["status"] in {"success", "reused_existing_pose"}]
    if not successful_pose_files:
        raise RuntimeError("No valid pose outputs are available.")

    feature_cmd = [sys.executable, str(FEATURE_SCRIPT), "--pose-dir", str(POSE_DIR), "--metadata", str(METADATA_CSV), "--out", str(FEATURES_CSV), "--max-interp-gap-ms", str(args.max_interp_gap_ms), "--savgol-window-ms", str(args.savgol_window_ms), "--event-savgol-window-ms", str(args.event_savgol_window_ms), "--ffc-event-savgol-window-ms", str(args.ffc_event_savgol_window_ms), "--ffc-lookback-ms", str(args.ffc_lookback_ms), "--ffc-stability-hold-ms", str(args.ffc_stability_hold_ms), "--ffc-precontact-window-ms", str(args.ffc_precontact_window_ms), "--ffc-ground-tolerance-foot-length", str(args.ffc_ground_tolerance_foot_length), "--release-search-start-frac", str(args.release_search_start_frac)]
    print(command_string(feature_cmd)); result = run_command(feature_cmd)
    if result.stdout: print(result.stdout)
    if result.returncode != 0:
        if result.stderr: print(result.stderr)
        raise RuntimeError("Pitch-level feature extraction failed.")

    qc_cmd = [sys.executable, str(QC_SCRIPT), "--input", str(FEATURES_CSV), "--annotated-out", str(QC_ANNOTATED_CSV), "--report-out", str(QC_REPORT), "--flow-out", str(QC_FLOW)]
    result = run_command(qc_cmd)
    if result.stdout: print(result.stdout)
    if result.returncode != 0:
        if result.stderr: print(result.stderr)
        raise RuntimeError("Measurement QC failed.")

    mechanics_cmd = [sys.executable, str(RELATIVE_MECHANICS_SCRIPT), "--input", str(QC_ANNOTATED_CSV), "--out", str(RELATIVE_MECHANICS_CSV)]
    result = run_command(mechanics_cmd)
    if result.stdout: print(result.stdout)
    if result.returncode != 0:
        if result.stderr: print(result.stderr)
        raise RuntimeError("Relative mechanical characterization failed.")

    PARAMETERS_JSON.write_text(json.dumps(parameter_record(args), indent=2), encoding="utf-8")
    write_environment_archive()
    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Pose manifest       : {MANIFEST_CSV}")
    print(f"Master features     : {FEATURES_CSV}")
    print(f"QC annotated        : {QC_ANNOTATED_CSV}")
    print(f"QC report           : {QC_REPORT}")
    print(f"QC flow             : {QC_FLOW}")
    print(f"Relative mechanics  : {RELATIVE_MECHANICS_CSV}")
    print(f"Pipeline parameters : {PARAMETERS_JSON}")
    print(f"Software environment: {ENVIRONMENT_TXT}")
    print("\nNEXT: Run prepare_analysis_data.py on outputs/features/pitch_features_with_relative_mechanics.csv")


if __name__ == "__main__":
    main()
