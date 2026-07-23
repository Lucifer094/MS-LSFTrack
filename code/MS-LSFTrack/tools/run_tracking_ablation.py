from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.config import load_dataset_config


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def make_variants(args) -> list[dict]:
    variants: list[dict] = []
    if getattr(args, "include_modules", False):
        variants.extend(
            [
                {"suffix": "motion_prior", "score_mode": "geometry", "scale_mode": "small"},
                {"suffix": "structure", "score_mode": "structure_zscore", "scale_mode": "small"},
                {
                    "suffix": "fixed_m0p5",
                    "score_mode": "fixed_fusion",
                    "scale_mode": "small",
                    "fixed_motion_weight": 0.5,
                },
                {"suffix": "residual_full", "score_mode": "residual", "scale_mode": "small"},
            ]
        )
        return variants
    if getattr(args, "include_fusion", False):
        if args.competitive_checkpoint is None:
            raise ValueError("--only fusion requires --competitive-checkpoint for the competitive gate row")
        for weight in parse_floats(args.fixed_motion_weights):
            label = str(weight).replace(".", "p")
            variants.append(
                {
                    "suffix": f"fixed_m{label}",
                    "score_mode": "fixed_fusion",
                    "scale_mode": args.fusion_scale_mode,
                    "fixed_motion_weight": weight,
                    "checkpoint": args.checkpoint,
                }
            )
        variants.append(
            {
                "suffix": "residual",
                "score_mode": "residual",
                "scale_mode": args.fusion_scale_mode,
                "checkpoint": args.checkpoint,
            }
        )
        variants.append(
            {
                "suffix": "competitive",
                "score_mode": "competitive",
                "scale_mode": args.fusion_scale_mode,
                "checkpoint": args.competitive_checkpoint,
            }
        )
        return variants
    if getattr(args, "include_competitive", False):
        variants.append(
            {
                "suffix": "competitive",
                "score_mode": "competitive",
                "scale_mode": args.fusion_scale_mode,
                "checkpoint": args.competitive_checkpoint or args.checkpoint,
            }
        )
        return variants
    if args.include_sources:
        variants.extend(
            [
                {"suffix": "motion", "score_mode": "distance", "scale_mode": "learned"},
                {"suffix": "structure", "score_mode": "structure", "scale_mode": "learned"},
                {"suffix": "residual", "score_mode": "residual", "scale_mode": "learned"},
                {"suffix": "competitive", "score_mode": "competitive", "scale_mode": "learned"},
            ]
        )
    if args.include_scales:
        variants.extend(
            [
                {"suffix": "scale_small", "score_mode": args.scale_score_mode, "scale_mode": "small"},
                {"suffix": "scale_middle", "score_mode": args.scale_score_mode, "scale_mode": "middle"},
                {"suffix": "scale_large", "score_mode": args.scale_score_mode, "scale_mode": "large"},
                {"suffix": "scale_uniform", "score_mode": args.scale_score_mode, "scale_mode": "uniform"},
                {"suffix": "scale_learned", "score_mode": args.scale_score_mode, "scale_mode": "learned"},
            ]
        )
    if args.include_fixed_fusion:
        for weight in parse_floats(args.fixed_motion_weights):
            label = str(weight).replace(".", "p")
            variants.append(
                {
                    "suffix": f"fixed_m{label}",
                    "score_mode": "fixed_fusion",
                    "scale_mode": "learned",
                    "fixed_motion_weight": weight,
                }
            )
    seen = set()
    deduped = []
    for variant in variants:
        key = (
            variant["suffix"],
            variant["score_mode"],
            variant["scale_mode"],
            float(variant.get("fixed_motion_weight", args.fixed_motion_weight)),
            str(variant.get("checkpoint", args.checkpoint)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tracking-level MS-LSFTrack ablations.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--competitive-checkpoint", type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--tracker-prefix", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-radius", type=float, default=10.0)
    parser.add_argument("--score-threshold", type=float, default=-17.0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--fixed-motion-weight", type=float, default=0.5)
    parser.add_argument("--fixed-motion-weights", default="0.75,0.5,0.25")
    parser.add_argument("--fusion-scale-mode", choices=["learned", "uniform", "small", "middle", "large"], default="learned")
    parser.add_argument("--scale-score-mode", choices=["residual", "competitive", "full"], default="residual")
    parser.add_argument(
        "--only",
        choices=["all", "modules", "sources", "scales", "fixed_fusion", "fusion", "competitive"],
        default="all",
    )
    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    args.include_modules = args.only == "modules"
    args.include_fusion = args.only == "fusion"
    args.include_competitive = args.only == "competitive"
    args.include_sources = args.only in {"all", "sources"}
    args.include_scales = args.only in {"all", "scales"}
    args.include_fixed_fusion = args.only in {"all", "fixed_fusion"}

    ds = load_dataset_config(args.dataset)
    split = args.split or ds.default_split
    variants = make_variants(args)
    tracker_names = [f"{args.tracker_prefix}_{variant['suffix']}" for variant in variants]

    if not args.skip_tracking:
        for tracker_name, variant in zip(tracker_names, variants):
            cmd = [
                sys.executable,
                "tools/run_tracker.py",
                "--dataset",
                str(args.dataset),
                "--checkpoint",
                str(variant.get("checkpoint", args.checkpoint)),
                "--split",
                split,
                "--tracker-name",
                tracker_name,
                "--device",
                args.device,
                "--score-mode",
                variant["score_mode"],
                "--scale-mode",
                variant["scale_mode"],
                "--fixed-motion-weight",
                str(variant.get("fixed_motion_weight", args.fixed_motion_weight)),
                "--gate-radius",
                str(args.gate_radius),
                "--score-threshold",
                str(args.score_threshold),
            ]
            if args.max_scenes:
                cmd.extend(["--max-scenes", str(args.max_scenes)])
            run(cmd)

    if not args.skip_eval:
        for mode in ("point", "iou"):
            cmd = [
                sys.executable,
                "tools/evaluate_tracks.py",
                "--dataset",
                str(args.dataset),
                "--trackers",
                *tracker_names,
                "--split",
                split,
                "--mode",
                mode,
            ]
            if args.max_scenes:
                cmd.extend(["--max-scenes", str(args.max_scenes)])
            run(cmd)

        out_dir = ds.unified_root / "comparison" / "ablations"
        out_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "tools/compare_trackers.py",
                "--dataset",
                str(args.dataset),
                "--trackers",
                *tracker_names,
                "--split",
                split,
                "--output",
                str(out_dir / f"{args.tracker_prefix}_{split}_tracking_ablation.csv"),
            ]
        )

    print("[DONE] variants:")
    for tracker_name, variant in zip(tracker_names, variants):
        print(f"  {tracker_name}: {variant}")


if __name__ == "__main__":
    main()
