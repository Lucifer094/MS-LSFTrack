from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.config import load_dataset_config
from mslsftrack.mot_io import infer_scenes, load_detection_scene, read_split, resolve_detection_file
from mslsftrack.tracker import (
    LocalStructureFieldTracker,
    frame_box_density_rho,
    load_model,
    load_neighborhood_config,
)


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def estimate_density_stats(
    dataset_path: Path,
    split: str,
    quantiles: tuple[float, float],
) -> tuple[float, float, float]:
    ds = load_dataset_config(dataset_path)
    split_file = ds.split_root / f"{split}.txt"
    scenes = read_split(split_file) if split_file.exists() else infer_scenes(ds.data_root, split)
    values: list[float] = []
    for scene_name in scenes:
        det_file = resolve_detection_file(ds.detection_root, split, scene_name)
        scene = load_detection_scene(scene_name, det_file)
        for detections in scene.frame_to_indices.values():
            rho = frame_box_density_rho(scene, detections)
            if rho is not None:
                values.append(float(rho))
    if not values:
        return 0.05, 0.20, 0.10
    low_q, high_q = quantiles
    arr = np.asarray(values, dtype=np.float32)
    low = float(max(1e-6, np.quantile(arr, low_q)))
    high = float(max(low + 1e-6, np.quantile(arr, high_q)))
    init = float(max(1e-6, np.median(arr)))
    return low, high, init


def write_mot(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            values = [
                str(int(row[0])),
                str(int(row[1])),
                *[f"{float(value):.6f}" for value in row[2:]],
            ]
            handle.write(",".join(values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online MS-LSFTrack association.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--score-mode",
        choices=[
            "full",
            "residual",
            "competitive",
            "fixed_fusion",
            "structure",
            "structure_zscore",
            "geometry",
            "field_l1",
            "distance",
            "last_distance",
            "agrf_lite",
            "agrf_full",
            "rank_agrf",
            "safe_fixed",
            "expert_select",
            "expert_pool",
            "ambiguity_rerank_strict",
            "ambiguity_rerank",
            "ambiguity_rerank_loose",
        ],
        default="full",
    )
    parser.add_argument(
        "--scale-mode",
        choices=[
            "learned",
            "uniform",
            "small",
            "middle",
            "large",
            "density",
            "density_soft",
            "density_only",
            "learned_density",
        ],
        default="learned",
    )
    parser.add_argument("--fixed-motion-weight", type=float, default=0.5)
    parser.add_argument(
        "--structure-radius-override",
        type=float,
        default=None,
        help=(
            "Override the local-structure radius at inference time. For multi-scale checkpoints, "
            "the value is repeated for every scale so scale-mode=small can be used as a single-radius probe."
        ),
    )
    parser.add_argument(
        "--structure-radii-override",
        default=None,
        help="Comma-separated inference-time structure radii override, e.g. 5,10,15.",
    )
    parser.add_argument("--gate-radius", type=float, default=10.0)
    parser.add_argument("--max-gate-radius", type=float, default=16.0)
    parser.add_argument("--missing-gate-growth", type=float, default=0.75)
    parser.add_argument("--missing-score-penalty", type=float, default=0.05)
    parser.add_argument("--distance-score-weight", type=float, default=0.0)
    parser.add_argument("--distance-prior-mode", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--adaptive-distance-margin-low", type=float, default=0.02)
    parser.add_argument("--adaptive-distance-margin-high", type=float, default=0.20)
    parser.add_argument("--adaptive-distance-min-factor", type=float, default=0.0)
    parser.add_argument("--density-tau-low", type=float, default=None)
    parser.add_argument("--density-tau-high", type=float, default=None)
    parser.add_argument("--density-init", type=float, default=None)
    parser.add_argument("--density-threshold-split", default="train")
    parser.add_argument("--density-quantiles", default="0.33,0.66")
    parser.add_argument("--density-alpha", type=float, default=0.8)
    parser.add_argument("--density-beta", type=float, default=2.0)
    parser.add_argument("--density-delta", type=float, default=0.05)
    parser.add_argument("--density-prior-lambda", type=float, default=1.0)
    parser.add_argument("--density-prior-direction", choices=["normal", "reverse"], default="normal")
    parser.add_argument("--asr-density-gain", type=float, default=1.0)
    parser.add_argument("--score-threshold", type=float, default=-17.0)
    parser.add_argument("--max-missing", type=int, default=20)
    parser.add_argument("--min-track-length", type=int, default=25)
    parser.add_argument("--mature-birth-length", type=int, default=25)
    parser.add_argument("--assignment", choices=["hungarian", "greedy"], default="hungarian")
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    split = args.split or ds.default_split
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    neigh = load_neighborhood_config(args.checkpoint)
    structure_radius = neigh["structure_radius"]
    structure_radii = neigh["structure_radii"]
    if args.structure_radii_override is not None:
        structure_radii = parse_float_tuple(args.structure_radii_override)
        structure_radius = structure_radii[0] if structure_radii else structure_radius
    elif args.structure_radius_override is not None:
        structure_radius = float(args.structure_radius_override)
        if structure_radii is not None:
            structure_radii = tuple(structure_radius for _ in structure_radii)
    density_tau_low = args.density_tau_low
    density_tau_high = args.density_tau_high
    density_init = args.density_init
    if args.scale_mode in {"density", "density_soft", "density_only", "learned_density"} and (
        density_tau_low is None or density_tau_high is None or density_init is None
    ):
        quantiles = parse_float_tuple(args.density_quantiles)
        if len(quantiles) != 2:
            raise ValueError("--density-quantiles must contain two comma-separated values")
        estimated_low, estimated_high, estimated_init = estimate_density_stats(
            args.dataset,
            args.density_threshold_split,
            (quantiles[0], quantiles[1]),
        )
        density_tau_low = estimated_low if density_tau_low is None else density_tau_low
        density_tau_high = estimated_high if density_tau_high is None else density_tau_high
        density_init = estimated_init if density_init is None else density_init
        print(
            "[DENSITY]",
            f"split={args.density_threshold_split}",
            f"tau_low={density_tau_low:.6f}",
            f"tau_high={density_tau_high:.6f}",
            f"init={density_init:.6f}",
            flush=True,
        )
    tracker = LocalStructureFieldTracker(
        model=model,
        device=device,
        gate_radius=args.gate_radius,
        structure_radius=structure_radius,
        structure_radii=structure_radii,
        neighbor_mode=neigh["neighbor_mode"],
        knn=neigh["knn"],
        max_gate_radius=args.max_gate_radius,
        missing_gate_growth=args.missing_gate_growth,
        missing_score_penalty=args.missing_score_penalty,
        distance_score_weight=args.distance_score_weight,
        distance_prior_mode=args.distance_prior_mode,
        adaptive_distance_margin_low=args.adaptive_distance_margin_low,
        adaptive_distance_margin_high=args.adaptive_distance_margin_high,
        adaptive_distance_min_factor=args.adaptive_distance_min_factor,
        density_tau_low=float(density_tau_low) if density_tau_low is not None else 0.05,
        density_tau_high=float(density_tau_high) if density_tau_high is not None else 0.20,
        density_init=float(density_init) if density_init is not None else None,
        density_alpha=args.density_alpha,
        density_beta=args.density_beta,
        density_delta=args.density_delta,
        density_prior_lambda=args.density_prior_lambda,
        density_prior_direction=args.density_prior_direction,
        asr_density_gain=args.asr_density_gain,
        score_threshold=args.score_threshold,
        max_missing=args.max_missing,
        min_track_length=args.min_track_length,
        mature_birth_length=args.mature_birth_length,
        score_mode=args.score_mode,
        scale_mode=args.scale_mode,
        fixed_motion_weight=args.fixed_motion_weight,
        assignment=args.assignment,
    )
    split_file = ds.split_root / f"{split}.txt"
    scenes = read_split(split_file) if split_file.exists() else infer_scenes(ds.data_root, split)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    output_root = ds.unified_root / "tracks" / ds.det_version / args.tracker_name / split
    runtime_rows = []
    for scene_name in scenes:
        det_file = resolve_detection_file(ds.detection_root, split, scene_name)
        scene = load_detection_scene(scene_name, det_file)
        start = time.perf_counter()
        rows = tracker.run(scene)
        elapsed = time.perf_counter() - start
        write_mot(output_root / f"{scene_name}.txt", rows)
        frames = max(1, len(scene.frame_to_indices))
        runtime_rows.append(
            {
                "scene": scene_name,
                "num_frames": frames,
                "time_sec": elapsed,
                "fps": frames / max(elapsed, 1e-9),
                "num_rows": len(rows),
            }
        )
        print(f"[TRACK] {scene_name}: rows={len(rows)} time={elapsed:.2f}s fps={frames/max(elapsed,1e-9):.1f}", flush=True)

    runtime_dir = ds.unified_root / "runtime" / ds.det_version / args.tracker_name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with (runtime_dir / f"{split}_per_scene.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "num_frames", "time_sec", "fps", "num_rows"])
        writer.writeheader()
        writer.writerows(runtime_rows)
    total_frames = sum(int(row["num_frames"]) for row in runtime_rows)
    total_time = sum(float(row["time_sec"]) for row in runtime_rows)
    summary = {
        "tracker": args.tracker_name,
        "det_version": ds.det_version,
        "split": split,
        "num_scenes": len(runtime_rows),
        "total_frames": total_frames,
        "total_outputs": sum(int(row["num_rows"]) for row in runtime_rows),
        "total_time_sec": f"{total_time:.6f}",
        "update_time_sec": f"{total_time:.6f}",
        "fps_total": f"{total_frames / max(total_time, 1e-12):.6f}" if total_frames else "0.000000",
        "fps_update": f"{total_frames / max(total_time, 1e-12):.6f}" if total_frames else "0.000000",
    }
    with (runtime_dir / f"{split}_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"[DONE] tracks={output_root}")


if __name__ == "__main__":
    main()
