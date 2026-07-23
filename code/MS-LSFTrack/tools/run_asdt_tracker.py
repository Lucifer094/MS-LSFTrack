from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.config import load_dataset_config
from mslsftrack.mot_io import infer_scenes, read_split, resolve_detection_file

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Detection:
    frame: int
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def center(self) -> np.ndarray:
        return np.asarray([self.x + self.w * 0.5, self.y + self.h * 0.5], dtype=np.float32)

    def as_row(self, track_id: int, score: float | None = None) -> list[float]:
        return [
            float(self.frame),
            float(track_id),
            float(self.x),
            float(self.y),
            float(max(0.0, self.w)),
            float(max(0.0, self.h)),
            float(self.score if score is None else score),
            -1.0,
            -1.0,
            -1.0,
        ]


@dataclass
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    missing: int = 0

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    def velocity(self) -> np.ndarray:
        if len(self.detections) < 2:
            return np.zeros(2, dtype=np.float32)
        previous = self.detections[-2]
        current = self.detections[-1]
        dt = max(1, current.frame - previous.frame)
        return (current.center - previous.center) / float(dt)

    def predicted_center(self, frame: int) -> np.ndarray:
        dt = max(1, frame - self.last.frame)
        return self.last.center + self.velocity() * float(dt)

    def update(self, detection: Detection) -> None:
        self.detections.append(detection)
        self.missing = 0


def parse_detection_file(path: Path, score_threshold: float, max_bboxes_per_frame: int) -> dict[int, list[Detection]]:
    grouped: dict[int, list[Detection]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return grouped
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part for part in line.replace(" ", ",").split(",") if part]
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) >= 7:
                frame, x, y, w, h, score = values[0], values[2], values[3], values[4], values[5], values[6]
            elif len(values) >= 6:
                frame, x, y, w, h, score = values[:6]
            else:
                continue
            if score < score_threshold or w <= 0 or h <= 0:
                continue
            grouped.setdefault(int(round(frame)), []).append(
                Detection(int(round(frame)), float(x), float(y), float(w), float(h), float(score))
            )
    if max_bboxes_per_frame > 0:
        for frame, detections in grouped.items():
            grouped[frame] = sorted(detections, key=lambda det: det.score, reverse=True)[:max_bboxes_per_frame]
    return grouped


def resolve_image_dir(data_root: Path, scene: str, split: str | None = None) -> Path | None:
    candidates = [
        data_root / "sequence" / scene,
        data_root / scene / "img1",
        data_root / scene / "images",
        data_root / "images" / scene,
    ]
    if split:
        candidates.extend(
            [
                data_root / split / scene / "img1",
                data_root / split / scene / "images",
                data_root / split / scene,
            ]
        )
    for path in candidates:
        if path.is_dir() and any(child.suffix.lower() in IMG_EXTS for child in path.iterdir() if child.is_file()):
            return path
    return None


def infer_image_shape(data_root: Path, scene: str, split: str | None, detections: dict[int, list[Detection]]) -> tuple[int, int]:
    image_dir = resolve_image_dir(data_root, scene, split)
    if image_dir is not None:
        try:
            import cv2

            first_image = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMG_EXTS)[0]
            image = cv2.imread(str(first_image), cv2.IMREAD_UNCHANGED)
            if image is not None:
                return int(image.shape[1]), int(image.shape[0])
        except Exception:
            pass
    max_x = 1.0
    max_y = 1.0
    for frame_dets in detections.values():
        for det in frame_dets:
            max_x = max(max_x, det.x + det.w)
            max_y = max(max_y, det.y + det.h)
    return int(math.ceil(max_x)), int(math.ceil(max_y))


def iou_xywh(a: Detection, b: Detection) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return float(inter / union) if union > 0 else 0.0


def movement_gate_px(width: int, height: int, normalized_gate: float, fixed_gate_px: float | None) -> float:
    if fixed_gate_px is not None and fixed_gate_px > 0:
        return float(fixed_gate_px)
    return float(normalized_gate * max(width, height))


def assign_tracks(
    tracks: list[Track],
    detections: list[Detection],
    frame: int,
    gate_px: float,
    iou_weight: float,
    score_weight: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    if not tracks or not detections:
        return [], set(), set()
    cost = np.full((len(tracks), len(detections)), 1e6, dtype=np.float32)
    for i, track in enumerate(tracks):
        predicted = track.predicted_center(frame)
        for j, det in enumerate(detections):
            distance = float(np.linalg.norm(det.center - predicted))
            allowed_gate = gate_px * max(1.0, 1.0 + 0.25 * track.missing)
            if distance > allowed_gate:
                continue
            distance_cost = distance / max(allowed_gate, 1e-6)
            iou_cost = 1.0 - iou_xywh(track.last, det)
            score_cost = 1.0 - det.score
            cost[i, j] = distance_cost + iou_weight * iou_cost + score_weight * score_cost
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    matched_tracks = set()
    matched_dets = set()
    for row, col in zip(row_ind, col_ind):
        if cost[row, col] >= 1e5:
            continue
        matches.append((int(row), int(col)))
        matched_tracks.add(int(row))
        matched_dets.add(int(col))
    return matches, matched_tracks, matched_dets


def interpolate_track(track: Track, max_gap: int, score: float) -> list[list[float]]:
    rows: list[list[float]] = []
    detections = sorted(track.detections, key=lambda det: det.frame)
    for index, det in enumerate(detections):
        if index > 0:
            previous = detections[index - 1]
            gap = det.frame - previous.frame
            if 1 < gap <= max_gap + 1:
                for frame in range(previous.frame + 1, det.frame):
                    alpha = (frame - previous.frame) / float(gap)
                    interp = Detection(
                        frame=frame,
                        x=previous.x * (1.0 - alpha) + det.x * alpha,
                        y=previous.y * (1.0 - alpha) + det.y * alpha,
                        w=previous.w * (1.0 - alpha) + det.w * alpha,
                        h=previous.h * (1.0 - alpha) + det.h * alpha,
                        score=score,
                    )
                    rows.append(interp.as_row(track.track_id, score=score))
        rows.append(det.as_row(track.track_id))
    return rows


def run_asdt_scene(
    detections: dict[int, list[Detection]],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> tuple[list[list[float]], dict[str, Any]]:
    gate_px = movement_gate_px(width, height, args.max_movement, args.max_movement_px)
    frames = sorted(detections)
    if not frames:
        return [], {"num_frames": 0, "num_dets": 0, "num_outputs": 0, "gate_px": f"{gate_px:.6f}"}

    tracks: dict[int, Track] = {}
    next_id = 1
    for frame in range(min(frames), max(frames) + 1):
        frame_dets = detections.get(frame, [])
        active_ids = [track_id for track_id, track in tracks.items() if track.missing <= args.max_missing]
        active_tracks = [tracks[track_id] for track_id in active_ids]
        matches, matched_tracks, matched_dets = assign_tracks(
            active_tracks,
            frame_dets,
            frame,
            gate_px,
            args.iou_weight,
            args.score_weight,
        )
        for active_index, det_index in matches:
            tracks[active_ids[active_index]].update(frame_dets[det_index])
        for active_index, track in enumerate(active_tracks):
            if active_index not in matched_tracks:
                track.missing += 1
        for det_index, det in enumerate(frame_dets):
            if det_index in matched_dets:
                continue
            tracks[next_id] = Track(track_id=next_id, detections=[det])
            next_id += 1

    rows: list[list[float]] = []
    kept_tracks = 0
    for track in tracks.values():
        if len(track.detections) < args.min_track_length:
            continue
        kept_tracks += 1
        rows.extend(interpolate_track(track, args.interpolate_gap, args.interpolate_score))
    rows.sort(key=lambda row: (int(row[0]), int(row[1])))
    return rows, {
        "num_frames": max(frames) - min(frames) + 1,
        "num_dets": sum(len(value) for value in detections.values()),
        "num_outputs": len(rows),
        "num_tracks": kept_tracks,
        "gate_px": f"{gate_px:.6f}",
        "width": width,
        "height": height,
    }


def write_mot(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".partial")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                ",".join(
                    [
                        str(int(row[0])),
                        str(int(row[1])),
                        *[f"{float(value):.6f}" for value in row[2:]],
                    ]
                )
                + "\n"
            )
    tmp_path.replace(path)


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_runtime_summary(rows: list[dict[str, Any]], path: Path, tracker: str, det_version: str, split: str) -> None:
    total_frames = sum(int(float(row.get("num_frames", 0))) for row in rows)
    total_dets = sum(int(float(row.get("num_dets", 0))) for row in rows)
    total_outputs = sum(int(float(row.get("num_outputs", 0))) for row in rows)
    total_tracks = sum(int(float(row.get("num_tracks", 0))) for row in rows)
    total_time = sum(float(row.get("time_sec", 0.0)) for row in rows)
    summary = {
        "tracker": tracker,
        "det_version": det_version,
        "split": split,
        "num_scenes": len(rows),
        "total_frames": total_frames,
        "total_dets": total_dets,
        "total_outputs": total_outputs,
        "total_tracks": total_tracks,
        "total_time_sec": f"{total_time:.6f}",
        "update_time_sec": f"{total_time:.6f}",
        "fps_total": f"{total_frames / max(total_time, 1e-12):.6f}" if total_frames else "0.000000",
        "fps_update": f"{total_frames / max(total_time, 1e-12):.6f}" if total_frames else "0.000000",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a weak MOT baseline inspired by A-Simple-Detector-is-a-Strong-Tracker "
            "TC-filtering on clean-layout MOT detections. This is not the official ASDT "
            "multi-object tracker, because the released ASDT code does not include one."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--tracker-name", default="asdt_tc_filter")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-bboxes-per-frame", type=int, default=0, help="0 keeps all detections; the ASDT example used 3.")
    parser.add_argument("--max-movement", type=float, default=0.09, help="Frame-to-frame motion gate normalized by max(image width,height).")
    parser.add_argument("--max-movement-px", type=float, default=None, help="Absolute pixel gate; overrides --max-movement when set.")
    parser.add_argument("--max-missing", type=int, default=3)
    parser.add_argument("--interpolate-gap", type=int, default=3)
    parser.add_argument("--interpolate-score", type=float, default=0.5)
    parser.add_argument("--min-track-length", type=int, default=1)
    parser.add_argument("--iou-weight", type=float, default=0.15)
    parser.add_argument("--score-weight", type=float, default=0.05)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    split = args.split or ds.default_split
    split_file = ds.split_root / f"{split}.txt"
    scenes = read_split(split_file) if split_file.exists() else infer_scenes(ds.data_root, split)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    log_dir = ds.unified_root / "logs" / ds.det_version
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = log_dir / f"run_{args.tracker_name}_{split}_config.json"
    config_path.write_text(
        json.dumps(
            {
                **vars(args),
                "dataset_name": ds.name,
                "det_version": ds.det_version,
                "split": split,
                "num_scenes": len(scenes),
                "source_project": "A-Simple-Detector-is-a-Strong-Tracker",
                "source_component": "TC_Filtering idea adapted to multi-object MOT detections",
                "official_mot_reproduction": False,
                "mot_baseline_note": (
                    "The public ASDT repository exposes frame-dynamics detection preparation "
                    "and TC-filtering/post-processing code, but no official multi-object "
                    "association module with stable target IDs. Treat this script as a weak "
                    "heuristic baseline, not as a main 2025 MOT comparison method."
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"dataset": ds.name, "split": split, "num_scenes": len(scenes), "config": str(config_path)}, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("[DONE] dry-run")
        return

    output_root = ds.unified_root / "tracks" / ds.det_version / args.tracker_name / split
    runtime_dir = ds.unified_root / "runtime" / ds.det_version / args.tracker_name
    per_scene_csv = runtime_dir / f"{split}_per_scene.csv"
    summary_csv = runtime_dir / f"{split}_summary.csv"
    progress_csv = log_dir / f"{args.tracker_name}_{split}_progress.csv"
    if args.overwrite:
        for path in [per_scene_csv, summary_csv, progress_csv]:
            if path.exists():
                path.unlink()

    per_scene_fields = [
        "scene",
        "num_frames",
        "num_dets",
        "num_outputs",
        "num_tracks",
        "width",
        "height",
        "gate_px",
        "time_sec",
        "fps",
        "out_file",
    ]
    progress_fields = ["time", "tracker", "det_version", "split", "scene", "status", "message"]
    rows_done = read_csv_rows(per_scene_csv)
    done_scenes = {row.get("scene") for row in rows_done}

    for scene in scenes:
        out_file = output_root / f"{scene}.txt"
        if (not args.overwrite) and scene in done_scenes and out_file.exists():
            print(f"[SKIP] {split}/{scene}: already done")
            continue
        det_file = resolve_detection_file(ds.detection_root, split, scene)
        append_csv(
            progress_csv,
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tracker": args.tracker_name,
                "det_version": ds.det_version,
                "split": split,
                "scene": scene,
                "status": "START",
                "message": str(det_file),
            },
            progress_fields,
        )
        start = time.perf_counter()
        try:
            detections = parse_detection_file(det_file, args.score_threshold, args.max_bboxes_per_frame)
            width, height = infer_image_shape(ds.data_root, scene, split, detections)
            track_rows, stats = run_asdt_scene(detections, width, height, args)
            write_mot(out_file, track_rows)
        except Exception as exc:
            append_csv(
                progress_csv,
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tracker": args.tracker_name,
                    "det_version": ds.det_version,
                    "split": split,
                    "scene": scene,
                    "status": "ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                },
                progress_fields,
            )
            raise
        elapsed = time.perf_counter() - start
        stats = {
            "scene": scene,
            **stats,
            "time_sec": f"{elapsed:.6f}",
            "fps": f"{int(stats.get('num_frames', 0)) / max(elapsed, 1e-12):.6f}",
            "out_file": str(out_file),
        }
        append_csv(per_scene_csv, stats, per_scene_fields)
        append_csv(
            progress_csv,
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tracker": args.tracker_name,
                "det_version": ds.det_version,
                "split": split,
                "scene": scene,
                "status": "DONE",
                "message": f"tracks={stats['num_tracks']} outputs={stats['num_outputs']} fps={stats['fps']}",
            },
            progress_fields,
        )
        print(f"[OK] {split}/{scene}: tracks={stats['num_tracks']} outputs={stats['num_outputs']} fps={stats['fps']}")

    write_runtime_summary(read_csv_rows(per_scene_csv), summary_csv, args.tracker_name, ds.det_version, split)
    print(f"[DONE] tracks={output_root}")
    print(f"[DONE] runtime={summary_csv}")


if __name__ == "__main__":
    main()
