from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.datasets import RealCache


def q(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {name: float("nan") for name in ["mean", "p50", "p75", "p90", "p95", "max"]}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def load_cache(path: Path) -> RealCache:
    with path.open("rb") as handle:
        return pickle.load(handle)


def frame_density_stats(cache: RealCache) -> dict[str, float]:
    dets_per_frame = []
    matched_per_frame = []
    nearest_neighbor = []
    for scene in cache.scenes:
        for indices in scene.frame_to_indices.values():
            dets_per_frame.append(float(len(indices)))
            matched = indices[scene.pseudo_ids[indices] >= 0]
            matched_per_frame.append(float(len(matched)))
            xy = scene.xy[indices]
            if len(xy) >= 2:
                diff = xy[:, None, :] - xy[None, :, :]
                dist = np.linalg.norm(diff, axis=-1)
                dist += np.eye(len(xy), dtype=np.float32) * 1e6
                nearest_neighbor.extend(dist.min(axis=1).astype(np.float64).tolist())
    det_stats = q(dets_per_frame)
    matched_stats = q(matched_per_frame)
    nn_stats = q(nearest_neighbor)
    return {
        "frames": int(len(dets_per_frame)),
        **{f"dets_per_frame_{k}": v for k, v in det_stats.items()},
        **{f"matched_per_frame_{k}": v for k, v in matched_stats.items()},
        **{f"nearest_neighbor_px_{k}": v for k, v in nn_stats.items()},
    }


def query_stats(cache: RealCache) -> dict[str, float]:
    candidate_counts = []
    positive_ranks = []
    positive_distances = []
    nearest_correct = 0
    hard_02 = 0
    hard_05 = 0
    hard_10 = 0
    ambiguous_by_margin = []
    frame_query_counts = defaultdict(int)
    track_query_counts = defaultdict(int)
    density_at_query = []
    candidate_density_at_query = []
    same_frame_candidates = []
    for query in cache.queries:
        scene = cache.scenes[query.scene_index]
        predicted = np.asarray(query.predicted_xy, dtype=np.float32)
        candidates = np.asarray(query.candidate_indices, dtype=np.int32)
        distances = np.linalg.norm(scene.xy[candidates] - predicted[None], axis=1)
        order = np.argsort(distances)
        rank = int(np.flatnonzero(order == query.positive_index)[0]) + 1
        positive_distance = float(distances[query.positive_index])
        candidate_counts.append(float(len(candidates)))
        positive_ranks.append(float(rank))
        positive_distances.append(positive_distance)
        nearest_correct += int(rank == 1)
        if len(distances) >= 2:
            sorted_dist = np.sort(distances)
            margin = float((sorted_dist[1] - sorted_dist[0]) / 10.0)
            ambiguous_by_margin.append(margin)
            hard_02 += int(margin <= 0.2)
            hard_05 += int(margin <= 0.5)
            hard_10 += int(margin <= 1.0)
        key = (query.scene_index, int(query.target_frame))
        frame_query_counts[key] += 1
        track_query_counts[(query.scene_index, int(scene.pseudo_ids[query.history_det_index]))] += 1
        frame_indices = scene.frame_to_indices.get(int(query.target_frame), np.zeros(0, dtype=np.int32))
        density_at_query.append(float(len(frame_indices)))
        candidate_density_at_query.append(float(len(candidates)))
        same_frame_candidates.append(float(max(0, len(frame_indices) - 1)))
    n = max(1, len(cache.queries))
    rank_arr = np.asarray(positive_ranks, dtype=np.float64)
    margin_stats = q(ambiguous_by_margin)
    return {
        "queries": int(len(cache.queries)),
        "mean_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "candidate_p90": q(candidate_counts)["p90"],
        "nearest_distance_top1": nearest_correct / n,
        "positive_rank_mean": float(rank_arr.mean()) if rank_arr.size else float("nan"),
        "positive_rank_gt1_rate": float(np.mean(rank_arr > 1)) if rank_arr.size else float("nan"),
        "hard_margin_le_0p2_rate": hard_02 / n,
        "hard_margin_le_0p5_rate": hard_05 / n,
        "hard_margin_le_1p0_rate": hard_10 / n,
        "distance_margin_mean": margin_stats["mean"],
        "distance_margin_p50": margin_stats["p50"],
        "distance_margin_p90": margin_stats["p90"],
        **{f"positive_distance_px_{k}": v for k, v in q(positive_distances).items()},
        **{f"query_frame_dets_{k}": v for k, v in q(density_at_query).items()},
        **{f"same_frame_other_dets_{k}": v for k, v in q(same_frame_candidates).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose dataset density and candidate ambiguity.")
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.cache) != len(args.name):
        raise ValueError("--cache and --name must have the same count")

    rows = []
    for name, path in zip(args.name, args.cache):
        cache = load_cache(path)
        row = {
            "dataset": name,
            "cache": str(path),
            "split": cache.split,
            **cache.stats,
            **frame_density_stats(cache),
            **query_stats(cache),
        }
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
