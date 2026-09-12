#!/usr/bin/env python
"""
extract_pose.py

Frame-level MediaPipe Pose Landmarker extraction.

This script does not substitute a default frame rate when video FPS metadata are
invalid. Time-dependent and derivative features depend directly on FPS, so an
invalid value is treated as an acquisition/metadata failure until the true frame
rate is recovered.

The output records FPS provenance for downstream verification.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import re

import cv2
import mediapipe as mp
import pandas as pd


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def parse_filename(video_path: Path):
    stem = video_path.stem
    m = re.match(r"^(pitcher\d+)_(pitch\d+)$", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    return "unknown_pitcher", stem


def valid_fps(value: float) -> bool:
    return bool(math.isfinite(value) and value > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Full path to video")
    parser.add_argument("--model", required=True, help="Full path to pose_landmarker .task model")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pitcher_id, pitch_id = parse_filename(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not valid_fps(fps):
        cap.release()
        raise RuntimeError(
            f"Video FPS metadata are invalid for {video_path.name} (OpenCV reported {fps}). "
            "The pipeline will not assume a default FPS because event timing and derivatives depend on the true FPS. "
            "Recover or correct the frame rate before processing this pitch."
        )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    rows = []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(round((frame_idx / fps) * 1000.0))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = {
                "video_file": video_path.name,
                "pitcher_id": pitcher_id,
                "pitch_id": pitch_id,
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "fps": fps,
                "fps_source": "opencv_video_metadata",
                "fps_is_fallback": 0,
                "pose_detected": 1 if result.pose_landmarks else 0,
            }

            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                for i, lm in enumerate(pose):
                    name = LANDMARK_NAMES[i]
                    row[f"{name}_x"] = lm.x
                    row[f"{name}_y"] = lm.y
                    row[f"{name}_z"] = lm.z
                    row[f"{name}_visibility"] = getattr(lm, "visibility", None)

                if result.pose_world_landmarks:
                    world_pose = result.pose_world_landmarks[0]
                    for i, lm in enumerate(world_pose):
                        name = LANDMARK_NAMES[i]
                        row[f"{name}_world_x"] = lm.x
                        row[f"{name}_world_y"] = lm.y
                        row[f"{name}_world_z"] = lm.z
                        row[f"{name}_world_visibility"] = getattr(lm, "visibility", None)

            rows.append(row)
            frame_idx += 1

    cap.release()

    if not rows:
        raise RuntimeError(f"No frames were read from {video_path}")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"FPS: {fps:.6f} (source: opencv_video_metadata; fallback: no)")
    print(f"Frames processed: {len(df)}")
    print(f"Frames with detected pose: {int(df['pose_detected'].sum())}")


if __name__ == "__main__":
    main()
