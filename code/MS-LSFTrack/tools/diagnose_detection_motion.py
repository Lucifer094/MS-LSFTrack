from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.config import load_dataset_config
from mslsftrack.mot_io import read_mot_array, read_split, infer_scenes, resolve_detection_file, resolve_gt_file


def q(values) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {k: float("nan") for k in ["mean", "p50", "p75", "p90", "p95", "max"]}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def centers(rows: np.ndarray) -> np.ndarray:
    return rows[:, 2:4] + rows[:, 4:6] * 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose detection/GT matching and GT motion smoothness.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    split_file = ds.split_root / f"{args.split}.txt"
    scenes = read_split(split_file) if split_file.exists() else infer_scenes(ds.data_root, args.split)

    scene_rows = []
    center_errors = []
    gt_step = []
    gt_accel = []
    det_counts = []
    gt_counts = []
    matched_counts = []
    for scene in scenes:
        det_path = resolve_detection_file(ds.detection_root, args.split, scene)
        gt_path = resolve_gt_file(ds.gt_root, scene, args.split)
        det = read_mot_array(det_path)
        gt = read_mot_array(gt_path)
        det_counts.append(len(det))
        gt_counts.append(len(gt))
        matched = 0
        if len(det) and len(gt):
            det_frames = det[:, 0].astype(np.int32)
            gt_frames = gt[:, 0].astype(np.int32)
            for frame in np.intersect1d(np.unique(det_frames), np.unique(gt_frames)):
                di = np.flatnonzero(det_frames == frame)
                gi = np.flatnonzero(gt_frames == frame)
                dxy = centers(det[di])
                gxy = centers(gt[gi])
                dist = np.linalg.norm(dxy[:, None] - gxy[None], axis=2)
                row, col = linear_sum_assignment(dist)
                valid = dist[row, col] <= args.match_radius
                matched += int(valid.sum())
                center_errors.extend(dist[row[valid], col[valid]].astype(np.float64).tolist())

        if len(gt):
            for track_id in np.unique(gt[:, 1].astype(np.int32)):
                idx = np.flatnonzero(gt[:, 1].astype(np.int32) == track_id)
                idx = idx[np.argsort(gt[idx, 0])]
                if len(idx) < 2:
                    continue
                frames = gt[idx, 0].astype(np.float32)
                xy = centers(gt[idx]).astype(np.float32)
                steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(1.0, np.diff(frames))
                gt_step.extend(steps.astype(np.float64).tolist())
                if len(xy) >= 3:
                    vel = np.diff(xy, axis=0) / np.maximum(1.0, np.diff(frames))[:, None]
                    accel = np.linalg.norm(np.diff(vel, axis=0), axis=1)
                    gt_accel.extend(accel.astype(np.float64).tolist())
        matched_counts.append(matched)
        scene_rows.append(
            {
                "dataset": ds.name,
                "det_version": ds.det_version,
                "split": args.split,
                "scene": scene,
                "det_count": len(det),
                "gt_count": len(gt),
                "matched_count": matched,
                "det_per_gt": len(det) / max(len(gt), 1),
                "gt_match_rate": matched / max(len(gt), 1),
                "det_match_rate": matched / max(len(det), 1),
            }
        )

    summary = {
        "dataset": ds.name,
        "det_version": ds.det_version,
        "split": args.split,
        "num_scenes": len(scenes),
        "total_dets": int(sum(det_counts)),
        "total_gt": int(sum(gt_counts)),
        "total_matched": int(sum(matched_counts)),
        "gt_match_rate": float(sum(matched_counts) / max(sum(gt_counts), 1)),
        "det_match_rate": float(sum(matched_counts) / max(sum(det_counts), 1)),
        **{f"center_error_px_{k}": v for k, v in q(center_errors).items()},
        **{f"gt_step_px_per_frame_{k}": v for k, v in q(gt_step).items()},
        **{f"gt_accel_px_per_frame2_{k}": v for k, v in q(gt_accel).items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    scene_path = args.output.with_name(args.output.stem + "_scenes.csv")
    with scene_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scene_rows)
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
