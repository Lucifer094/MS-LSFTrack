from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
from typing import Dict, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.modules.setdefault("smia_v4", types.ModuleType("smia_v4"))
sys.modules.setdefault("smia_v4.mot_io", sys.modules[__name__])


@dataclass
class SceneDetections:
    scene: str
    frames: np.ndarray
    xy: np.ndarray
    wh: np.ndarray
    scores: np.ndarray
    pseudo_ids: np.ndarray
    frame_to_indices: Dict[int, np.ndarray]


def read_mot_array(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 10), dtype=np.float32)
    data = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if data.shape[1] < 6:
        raise ValueError(f"{path} has only {data.shape[1]} columns")
    return data


def assign_detection_ids(det_rows: np.ndarray, gt_rows: np.ndarray, radius: float = 3.0) -> np.ndarray:
    pseudo_ids = np.full(len(det_rows), -1, dtype=np.int32)
    if len(det_rows) == 0 or len(gt_rows) == 0:
        return pseudo_ids
    det_frames = det_rows[:, 0].astype(np.int32)
    gt_frames = gt_rows[:, 0].astype(np.int32)
    for frame in np.intersect1d(np.unique(det_frames), np.unique(gt_frames)):
        di = np.flatnonzero(det_frames == frame)
        gi = np.flatnonzero(gt_frames == frame)
        det_xy = det_rows[di, 2:4] + det_rows[di, 4:6] * 0.5
        gt_xy = gt_rows[gi, 2:4] + gt_rows[gi, 4:6] * 0.5
        distance = np.linalg.norm(det_xy[:, None] - gt_xy[None], axis=2)
        row, col = linear_sum_assignment(distance)
        valid = distance[row, col] <= radius
        pseudo_ids[di[row[valid]]] = gt_rows[gi[col[valid]], 1].astype(np.int32)
    return pseudo_ids


def load_scene(scene: str, det_file: Path, gt_file: Path, match_radius: float = 3.0) -> SceneDetections:
    det = read_mot_array(det_file)
    gt = read_mot_array(gt_file)
    frames = det[:, 0].astype(np.int32)
    xy = det[:, 2:4] + det[:, 4:6] * 0.5
    wh = det[:, 4:6].astype(np.float32)
    scores = det[:, 6].astype(np.float32) if det.shape[1] > 6 else np.ones(len(det), np.float32)
    pseudo_ids = assign_detection_ids(det, gt, radius=match_radius)
    frame_to_indices = {
        int(frame): np.flatnonzero(frames == frame).astype(np.int32)
        for frame in np.unique(frames)
    }
    return SceneDetections(
        scene=scene,
        frames=frames,
        xy=xy.astype(np.float32),
        wh=wh,
        scores=scores,
        pseudo_ids=pseudo_ids,
        frame_to_indices=frame_to_indices,
    )


def load_detection_scene(scene: str, det_file: Path) -> SceneDetections:
    det = read_mot_array(det_file)
    frames = det[:, 0].astype(np.int32)
    xy = det[:, 2:4] + det[:, 4:6] * 0.5
    frame_to_indices = {
        int(frame): np.flatnonzero(frames == frame).astype(np.int32)
        for frame in np.unique(frames)
    }
    return SceneDetections(
        scene=scene,
        frames=frames,
        xy=xy.astype(np.float32),
        wh=det[:, 4:6].astype(np.float32),
        scores=(det[:, 6] if det.shape[1] > 6 else np.ones(len(det))).astype(np.float32),
        pseudo_ids=np.full(len(det), -1, dtype=np.int32),
        frame_to_indices=frame_to_indices,
    )


def normalize_scene_name(name: str) -> str:
    """Normalize numeric IR-DMSTrack scene names while keeping GMOT/LEO names intact."""
    name = name.strip()
    return name.zfill(5) if name.isdigit() else name


def read_split(path: Path) -> list[str]:
    return [
        normalize_scene_name(line.strip().split()[0])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def infer_scenes(data_root: Path, split: str | None = None) -> list[str]:
    """Read a split file if available; otherwise infer scenes from supported dataset layouts."""
    candidates = []
    if split:
        candidates.extend([
            data_root / "splits" / f"{split}.txt",
            data_root / "splits" / split,
            data_root / f"{split}.txt",
        ])
    for path in candidates:
        if path.exists():
            return read_split(path)
    if not data_root.exists():
        return []
    track_label_root = data_root / "track_label"
    if track_label_root.exists():
        return sorted(normalize_scene_name(path.stem) for path in track_label_root.glob("*.txt"))
    sequence_root = data_root / "sequence"
    if sequence_root.exists():
        return sorted(normalize_scene_name(child.name) for child in sequence_root.iterdir() if child.is_dir())
    return sorted(
        child.name for child in data_root.iterdir()
        if child.is_dir() and (child / "gt" / "gt.txt").exists()
    )


def resolve_detection_file(detection_root: Path, split: str, scene: str) -> Path:
    """Support both unified/detector/split/scene.txt and unified/detector/scene.txt layouts."""
    candidates = [
        detection_root / split / f"{scene}.txt",
        detection_root / f"{scene}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_gt_file(gt_root: Path, scene: str, split: str | None = None) -> Path:
    """Support both MOT style {scene}/gt/gt.txt and clean style track_label/{scene}.txt."""
    candidates = [
        gt_root / scene / "gt" / "gt.txt",
        gt_root / "track_label" / f"{scene}.txt",
    ]
    if split:
        candidates.append(gt_root / "track_label" / split / f"{scene}.txt")
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def iter_scene_paths(
    scenes: Iterable[str],
    detection_root: Path,
    gt_root: Path,
    split: str,
):
    for scene in scenes:
        det_file = resolve_detection_file(detection_root, split, scene)
        gt_file = resolve_gt_file(gt_root, scene, split)
        if det_file.exists() and gt_file.exists():
            yield scene, det_file, gt_file
