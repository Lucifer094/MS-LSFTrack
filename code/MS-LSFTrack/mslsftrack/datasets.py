from __future__ import annotations

from dataclasses import dataclass
import pickle
import sys
import types
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .mot_io import SceneDetections, infer_scenes, iter_scene_paths, load_scene, read_split

# Backward compatibility for caches created by BoxMOT_V4/V6/V8 prototypes.
sys.modules.setdefault("smia_v4", types.ModuleType("smia_v4"))
sys.modules.setdefault("smia_v4.real_dataset", sys.modules[__name__])


@dataclass(frozen=True)
class Query:
    scene_index: int
    history_det_index: int
    target_frame: int
    candidate_indices: tuple[int, ...]
    positive_index: int
    predicted_xy: tuple[float, float]
    gap: int


@dataclass
class RealCache:
    split: str
    scenes: list[SceneDetections]
    queries: list[Query]
    stats: dict


def _track_indices(scene: SceneDetections) -> dict[int, np.ndarray]:
    result = {}
    for track_id in np.unique(scene.pseudo_ids):
        if track_id < 0:
            continue
        idx = np.flatnonzero(scene.pseudo_ids == track_id)
        idx = idx[np.argsort(scene.frames[idx])]
        result[int(track_id)] = idx
    return result


def build_queries_for_scene(
    scene_index: int,
    scene: SceneDetections,
    gate_radius: float,
    max_gap: int,
    max_candidates: int,
    min_candidates: int,
) -> list[Query]:
    queries: list[Query] = []
    for track_id, indices in _track_indices(scene).items():
        for pos in range(1, len(indices)):
            current_idx = int(indices[pos])
            current_frame = int(scene.frames[current_idx])
            history_idx = int(indices[pos - 1])
            history_frame = int(scene.frames[history_idx])
            gap = current_frame - history_frame
            if gap <= 0 or gap > max_gap:
                continue

            if pos >= 2:
                older_idx = int(indices[pos - 2])
                dt = max(1, history_frame - int(scene.frames[older_idx]))
                velocity = (scene.xy[history_idx] - scene.xy[older_idx]) / float(dt)
                predicted = scene.xy[history_idx] + velocity * float(gap)
            else:
                predicted = scene.xy[history_idx].copy()

            frame_indices = scene.frame_to_indices.get(current_frame)
            if frame_indices is None:
                continue
            distances = np.linalg.norm(scene.xy[frame_indices] - predicted[None], axis=1)
            valid = frame_indices[distances <= gate_radius]
            valid_distances = distances[distances <= gate_radius]
            if current_idx not in valid:
                continue
            order = np.argsort(valid_distances)
            candidates = valid[order][:max_candidates].astype(np.int32)
            if current_idx not in candidates:
                continue
            if len(candidates) < min_candidates:
                continue
            positive = int(np.flatnonzero(candidates == current_idx)[0])
            queries.append(
                Query(
                    scene_index=scene_index,
                    history_det_index=history_idx,
                    target_frame=current_frame,
                    candidate_indices=tuple(int(v) for v in candidates),
                    positive_index=positive,
                    predicted_xy=(float(predicted[0]), float(predicted[1])),
                    gap=gap,
                )
            )
    return queries


def build_real_cache(
    split: str,
    detection_root: Path,
    gt_root: Path,
    split_file: Path | None,
    output: Path,
    match_radius: float = 3.0,
    gate_radius: float = 10.0,
    max_gap: int = 3,
    max_candidates: int = 8,
    min_candidates: int = 2,
    max_queries: int | None = None,
    seed: int = 42,
) -> RealCache:
    if split_file is not None and split_file.exists():
        scene_names = read_split(split_file)
    else:
        scene_names = infer_scenes(gt_root, split)
    scenes: list[SceneDetections] = []
    queries: list[Query] = []
    for scene, det_file, gt_file in iter_scene_paths(scene_names, detection_root, gt_root, split):
        loaded = load_scene(scene, det_file, gt_file, match_radius=match_radius)
        scene_index = len(scenes)
        scene_queries = build_queries_for_scene(
            scene_index,
            loaded,
            gate_radius=gate_radius,
            max_gap=max_gap,
            max_candidates=max_candidates,
            min_candidates=min_candidates,
        )
        scenes.append(loaded)
        queries.extend(scene_queries)
        print(
            f"[CACHE] {split}/{scene}: detections={len(loaded.frames)} "
            f"matched={(loaded.pseudo_ids >= 0).sum()} queries={len(scene_queries)}"
        )

    if max_queries is not None and len(queries) > max_queries:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(queries), size=max_queries, replace=False))
        queries = [queries[int(i)] for i in keep]

    candidate_counts = np.asarray([len(q.candidate_indices) for q in queries], dtype=np.int32)
    stats = {
        "split": split,
        "num_scenes": len(scenes),
        "num_queries": len(queries),
        "mean_candidates": float(candidate_counts.mean()) if len(candidate_counts) else 0.0,
        "ambiguous_queries": int((candidate_counts >= 2).sum()),
        "gate_radius": gate_radius,
        "match_radius": match_radius,
        "max_gap": max_gap,
        "max_candidates": max_candidates,
    }
    cache = RealCache(split=split, scenes=scenes, queries=queries, stats=stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[DONE] cache={output} stats={stats}")
    return cache


def load_real_cache(path: Path) -> RealCache:
    with path.open("rb") as handle:
        return pickle.load(handle)


class RealCandidateGroupDataset(Dataset):
    def __init__(
        self,
        cache_path: Path,
        radius: float = 20.0,
        max_neighbors: int = 32,
        max_candidates: int = 8,
        neighbor_mode: str = "radius",
    ):
        self.cache = load_real_cache(cache_path)
        self.radius = float(radius)
        self.max_neighbors = int(max_neighbors)
        self.max_candidates = int(max_candidates)
        self.neighbor_mode = neighbor_mode

    def __len__(self) -> int:
        return len(self.cache.queries)

    def _neighbors(self, scene: SceneDetections, center_idx: int) -> tuple[np.ndarray, np.ndarray]:
        return extract_neighbors(
            scene, center_idx, self.radius, self.max_neighbors, neighbor_mode=self.neighbor_mode
        )

    def __getitem__(self, index: int):
        query = self.cache.queries[index]
        scene = self.cache.scenes[query.scene_index]
        history_points, history_mask = self._neighbors(scene, query.history_det_index)

        candidate_points = np.zeros(
            (self.max_candidates, self.max_neighbors, 3), dtype=np.float32
        )
        candidate_neighbor_mask = np.zeros(
            (self.max_candidates, self.max_neighbors), dtype=np.bool_
        )
        geometry = np.zeros((self.max_candidates, 5), dtype=np.float32)
        candidate_mask = np.zeros(self.max_candidates, dtype=np.bool_)
        predicted = np.asarray(query.predicted_xy, dtype=np.float32)
        for slot, candidate_idx in enumerate(query.candidate_indices):
            points, mask = self._neighbors(scene, candidate_idx)
            candidate_points[slot] = points
            candidate_neighbor_mask[slot] = mask
            delta = scene.xy[candidate_idx] - predicted
            distance = float(np.linalg.norm(delta))
            geometry[slot] = np.asarray(
                [
                    delta[0] / 10.0,
                    delta[1] / 10.0,
                    distance / 10.0,
                    scene.scores[candidate_idx],
                    min(query.gap / 3.0, 1.0),
                ],
                dtype=np.float32,
            )
            candidate_mask[slot] = True

        return {
            "history_points": torch.from_numpy(history_points),
            "history_mask": torch.from_numpy(history_mask),
            "candidate_points": torch.from_numpy(candidate_points),
            "candidate_neighbor_mask": torch.from_numpy(candidate_neighbor_mask),
            "geometry": torch.from_numpy(geometry),
            "candidate_mask": torch.from_numpy(candidate_mask),
            "target": torch.tensor(query.positive_index, dtype=torch.long),
            "query_index": torch.tensor(index, dtype=torch.long),
        }


class MultiScaleRealCandidateGroupDataset(Dataset):
    """Multi-scale local-structure-field dataset for candidate association.

    Instead of using a single fixed neighborhood radius, the dataset extracts
    neighbor sets under several radii for each history point and candidate.
    Each scale is normalized to [-1, 1], so the rasterizer can share the same
    output grid across all scales.
    """

    def __init__(
        self,
        cache_path: Path,
        radii: Sequence[float] = (10.0, 15.0, 20.0),
        max_neighbors: int = 32,
        max_candidates: int = 8,
        neighbor_mode: str = "radius",
        knn: Sequence[int] | None = None,
    ):
        self.cache = load_real_cache(cache_path)
        self.radii = tuple(float(v) for v in radii)
        self.max_neighbors = int(max_neighbors)
        self.max_candidates = int(max_candidates)
        self.neighbor_mode = neighbor_mode
        self.knn = tuple(int(v) for v in knn) if knn is not None else None
        if self.knn is not None and len(self.knn) != len(self.radii):
            raise ValueError("knn and radii must have the same length")

    def __len__(self) -> int:
        return len(self.cache.queries)

    def _neighbors_multi(self, scene: SceneDetections, center_idx: int) -> tuple[np.ndarray, np.ndarray]:
        points = np.zeros((len(self.radii), self.max_neighbors, 3), dtype=np.float32)
        masks = np.zeros((len(self.radii), self.max_neighbors), dtype=np.bool_)
        for scale_index, radius in enumerate(self.radii):
            max_neighbors = (
                min(self.max_neighbors, self.knn[scale_index])
                if self.neighbor_mode == "knn" and self.knn is not None
                else self.max_neighbors
            )
            scale_points, scale_mask = extract_neighbors(
                scene,
                center_idx,
                radius,
                self.max_neighbors,
                neighbor_mode=self.neighbor_mode,
                selection_limit=max_neighbors,
            )
            points[scale_index] = scale_points
            masks[scale_index] = scale_mask
        return points, masks

    def __getitem__(self, index: int):
        query = self.cache.queries[index]
        scene = self.cache.scenes[query.scene_index]
        history_points, history_mask = self._neighbors_multi(scene, query.history_det_index)

        candidate_points = np.zeros(
            (self.max_candidates, len(self.radii), self.max_neighbors, 3),
            dtype=np.float32,
        )
        candidate_neighbor_mask = np.zeros(
            (self.max_candidates, len(self.radii), self.max_neighbors),
            dtype=np.bool_,
        )
        geometry = np.zeros((self.max_candidates, 5), dtype=np.float32)
        candidate_mask = np.zeros(self.max_candidates, dtype=np.bool_)
        predicted = np.asarray(query.predicted_xy, dtype=np.float32)
        for slot, candidate_idx in enumerate(query.candidate_indices):
            points, mask = self._neighbors_multi(scene, candidate_idx)
            candidate_points[slot] = points
            candidate_neighbor_mask[slot] = mask
            delta = scene.xy[candidate_idx] - predicted
            distance = float(np.linalg.norm(delta))
            geometry[slot] = np.asarray(
                [
                    delta[0] / 10.0,
                    delta[1] / 10.0,
                    distance / 10.0,
                    scene.scores[candidate_idx],
                    min(query.gap / 3.0, 1.0),
                ],
                dtype=np.float32,
            )
            candidate_mask[slot] = True

        return {
            "history_points": torch.from_numpy(history_points),
            "history_mask": torch.from_numpy(history_mask),
            "candidate_points": torch.from_numpy(candidate_points),
            "candidate_neighbor_mask": torch.from_numpy(candidate_neighbor_mask),
            "geometry": torch.from_numpy(geometry),
            "candidate_mask": torch.from_numpy(candidate_mask),
            "target": torch.tensor(query.positive_index, dtype=torch.long),
            "query_index": torch.tensor(index, dtype=torch.long),
        }


def extract_neighbors(
    scene: SceneDetections,
    center_idx: int,
    radius: float = 20.0,
    max_neighbors: int = 32,
    neighbor_mode: str = "radius",
    selection_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = int(scene.frames[center_idx])
    frame_indices = scene.frame_to_indices[frame]
    relative = scene.xy[frame_indices] - scene.xy[center_idx][None]
    distances = np.linalg.norm(relative, axis=1)
    if neighbor_mode == "radius":
        valid = (distances > 1e-5) & (distances <= radius)
    elif neighbor_mode == "knn":
        valid = distances > 1e-5
    else:
        raise ValueError(f"unknown neighbor_mode: {neighbor_mode}")
    selected = frame_indices[valid]
    relative = relative[valid]
    distances = distances[valid]
    limit = int(selection_limit) if selection_limit is not None else max_neighbors
    if len(selected) > limit:
        order = np.argsort(distances)[:limit]
        selected = selected[order]
        relative = relative[order]
        distances = distances[order]
    points = np.zeros((max_neighbors, 3), dtype=np.float32)
    mask = np.zeros(max_neighbors, dtype=np.bool_)
    count = len(selected)
    if count:
        points[:count, :2] = np.clip(relative / radius, -1.0, 1.0)
        points[:count, 2] = scene.scores[selected]
        mask[:count] = True
    return points, mask
