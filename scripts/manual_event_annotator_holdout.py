#!/usr/bin/env python
"""Independent blinded holdout event-annotation utility.

This script reuses the interactive display from `manual_event_annotator.py` but
excludes every pitch already present in the development annotation file before
selecting a new pitcher-balanced holdout subset. Automated event estimates are
not displayed during annotation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from manual_event_annotator import (
    annotate_video,
    balanced_sample,
    load_candidates,
    load_existing,
    save_annotation,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_FEATURES = PROJECT_ROOT / "outputs" / "features" / "pitch_features_master.csv"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "videos"
DEFAULT_OUT = PROJECT_ROOT / "metadata" / "manual_event_annotations_holdout.csv"
DEFAULT_EXCLUDE = PROJECT_ROOT / "metadata" / "manual_event_annotations.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an independent blinded manual event holdout set.")
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE,
                   help="Development annotation CSV whose pitch IDs are excluded from holdout selection.")
    p.add_argument("--n-per-pitcher", type=int, default=5)
    p.add_argument("--seed", type=int, default=29,
                   help="Independent random seed used for holdout pitch selection.")
    p.add_argument("--overwrite-selection", action="store_true")
    return p.parse_args()


def excluded_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Development annotation file not found: {path}. "
            "The holdout annotator requires the development set so overlap can be prevented."
        )
    dev = load_existing(path)
    return set(dev["unique_pitch_id"].dropna().astype(str))


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.features)
    dev_ids = excluded_ids(args.exclude)
    eligible = candidates[~candidates["unique_pitch_id"].astype(str).isin(dev_ids)].copy()
    if eligible.empty:
        raise RuntimeError("No pitches remain after excluding the development annotation set.")

    existing = load_existing(args.out)
    sample = balanced_sample(eligible, args.n_per_pitcher, args.seed)

    overlap = set(sample["unique_pitch_id"].astype(str)) & dev_ids
    if overlap:
        raise RuntimeError(f"Development/holdout overlap detected before annotation: {sorted(overlap)[:10]}")

    print("=" * 72)
    print("BLINDED HOLDOUT EVENT ANNOTATOR")
    print("=" * 72)
    print(f"Feature table       : {args.features}")
    print(f"Development exclude : {args.exclude}")
    print(f"Holdout output      : {args.out}")
    print(f"Independent seed    : {args.seed}")
    print(f"Development pitches excluded: {len(dev_ids)}")
    print(f"Holdout pitches selected    : {len(sample)}")
    print("Automated event estimates are NOT shown.")
    print("=" * 72)

    completed = 0
    for i, row in sample.iterrows():
        uid = str(row["unique_pitch_id"])
        prior = None
        if not existing.empty:
            match = existing[existing["unique_pitch_id"].astype(str) == uid]
            if not match.empty:
                prior = match.iloc[0]
                complete = (
                    pd.notna(prior.get("manual_ffc_frame_idx", np.nan))
                    and pd.notna(prior.get("manual_release_frame_idx", np.nan))
                )
                if complete and not args.overwrite_selection:
                    print(f"Skipping already annotated holdout pitch: {uid}")
                    completed += 1
                    continue

        video_path = args.video_dir / str(row["video_file"])
        if not video_path.exists():
            print(f"WARNING: video not found, skipping: {video_path}")
            continue

        result = annotate_video(video_path, row, i, len(sample), prior)
        if result["ffc"] is not None or result["release"] is not None:
            record = {
                "unique_pitch_id": uid,
                "video_file": row["video_file"],
                "pitcher_id": row["pitcher_id"],
                "pitch_id": row["pitch_id"],
                "manual_ffc_frame_idx": result["ffc"],
                "manual_release_frame_idx": result["release"],
                "annotator": "",
                "notes": "",
            }
            existing = save_annotation(args.out, existing, record)

        if result["action"] == "save":
            completed += 1
            print(f"Saved holdout {uid}: FFC={result['ffc']}, Release={result['release']}")
        if result["action"] == "quit":
            print("Progress saved. Exiting.")
            break

    cv2.destroyAllWindows()

    if args.out.exists():
        saved = load_existing(args.out)
        saved_ids = set(saved["unique_pitch_id"].dropna().astype(str))
        contamination = saved_ids & dev_ids
        if contamination:
            raise RuntimeError(
                "Development/holdout contamination detected in saved annotations: "
                + ", ".join(sorted(contamination)[:10])
            )

    print(f"Completed holdout subset: {completed}/{len(sample)}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
