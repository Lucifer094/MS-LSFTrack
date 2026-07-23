from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.config import load_dataset_config


def split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def run(cmd: list[str]) -> None:
    print("\n[RUN] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def add_if(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def cache_path(cache_root: Path, dataset_name: str, det_version: str, split: str) -> Path:
    return cache_root / dataset_name / det_version / f"{split}.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end association-only MS-LSFTrack pipeline.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset YAML/JSON config.")
    parser.add_argument("--exp-name", default="ms_lsf_assoc", help="Experiment/tracker name.")
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("MS_LSF_CACHE_ROOT", "./cache")))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("MS_LSF_RUN_ROOT", "./runs")))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--build-splits", default="train,val,test")

    parser.add_argument("--model", choices=["single_lsf", "multiscale_lsf"], default="multiscale_lsf")
    parser.add_argument("--fusion-mode", choices=["residual", "competitive"], default="residual")
    parser.add_argument("--neighbor-mode", choices=["radius", "knn"], default="radius")
    parser.add_argument("--radii", default="10,15,20")
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--knn", default=None)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--gate-radius", type=float, default=10.0)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--max-cache-queries", type=int)
    parser.add_argument("--max-train-queries", type=int)
    parser.add_argument("--max-val-queries", type=int)
    parser.add_argument("--max-test-queries", type=int)

    parser.add_argument("--gate-modes", default="learned,uniform,small,middle,large")
    parser.add_argument("--tracker-name", default=None)
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
            "ambiguity_rerank",
        ],
        default="ambiguity_rerank",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["learned", "uniform", "small", "middle", "large", "density", "density_soft", "density_only", "learned_density"],
        default="learned_density",
    )
    parser.add_argument("--fixed-motion-weight", type=float, default=0.5)
    parser.add_argument("--density-prior-direction", choices=["normal", "reverse"], default="normal")
    parser.add_argument("--density-prior-lambda", type=float, default=0.5)
    parser.add_argument("--asr-density-gain", type=float, default=0.0)
    parser.add_argument("--eval-mode", choices=["point", "iou"], default="point")
    parser.add_argument("--max-scenes", type=int)

    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    exp_dir = args.output_root / args.exp_name
    checkpoint = exp_dir / "best.pt"
    tracker_name = args.tracker_name or args.exp_name

    if not args.skip_cache:
        for split in split_csv(args.build_splits):
            cmd = [
                sys.executable,
                "tools/build_cache.py",
                "--dataset",
                str(args.dataset),
                "--split",
                split,
                "--output-dir",
                str(args.cache_root),
                "--match-radius",
                str(args.match_radius),
                "--gate-radius",
                str(args.gate_radius),
                "--max-gap",
                str(args.max_gap),
                "--max-candidates",
                str(args.max_candidates),
                "--min-candidates",
                str(args.min_candidates),
                "--seed",
                str(args.seed),
            ]
            add_if(cmd, "--max-queries", args.max_cache_queries)
            run(cmd)

    train_cache = cache_path(args.cache_root, ds.name, ds.det_version, args.train_split)
    val_cache = cache_path(args.cache_root, ds.name, ds.det_version, args.val_split)
    test_cache = cache_path(args.cache_root, ds.name, ds.det_version, args.test_split)

    if not args.skip_train:
        cmd = [
            sys.executable,
            "tools/train_assoc.py",
            "--model",
            args.model,
            "--fusion-mode",
            args.fusion_mode,
            "--neighbor-mode",
            args.neighbor_mode,
            "--radii",
            args.radii,
            "--radius",
            str(args.radius),
            "--max-neighbors",
            str(args.max_neighbors),
            "--max-candidates",
            str(args.max_candidates),
            "--train-cache",
            str(train_cache),
            "--val-cache",
            str(val_cache),
            "--test-cache",
            str(test_cache),
            "--output",
            str(exp_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        add_if(cmd, "--knn", args.knn)
        add_if(cmd, "--max-train-queries", args.max_train_queries)
        add_if(cmd, "--max-val-queries", args.max_val_queries)
        add_if(cmd, "--max-test-queries", args.max_test_queries)
        run(cmd)

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    if not args.skip_ablation:
        cmd = [
            sys.executable,
            "tools/evaluate_assoc_ablation.py",
            "--checkpoint",
            str(checkpoint),
            "--test-cache",
            str(test_cache),
            "--output",
            str(exp_dir / "ablation"),
            "--batch-size",
            str(max(args.batch_size, 128)),
            "--workers",
            str(args.workers),
            "--device",
            args.device,
            "--gate-modes",
            args.gate_modes,
        ]
        add_if(cmd, "--max-test-queries", args.max_test_queries)
        run(cmd)

    if not args.skip_tracking:
        cmd = [
            sys.executable,
            "tools/run_tracker.py",
            "--dataset",
            str(args.dataset),
            "--checkpoint",
            str(checkpoint),
            "--split",
            args.test_split,
            "--tracker-name",
            tracker_name,
            "--device",
            args.device,
            "--score-mode",
            args.score_mode,
            "--scale-mode",
            args.scale_mode,
            "--structure-radii-override",
            args.radii,
            "--density-prior-direction",
            args.density_prior_direction,
            "--density-prior-lambda",
            str(args.density_prior_lambda),
            "--asr-density-gain",
            str(args.asr_density_gain),
            "--fixed-motion-weight",
            str(args.fixed_motion_weight),
            "--gate-radius",
            str(args.gate_radius),
        ]
        add_if(cmd, "--max-scenes", args.max_scenes)
        # run_tracker.py intentionally owns most online-tracking defaults; the
        # pipeline only forwards the core experiment controls.
        run(cmd)

    if not args.skip_eval:
        cmd = [
            sys.executable,
            "tools/evaluate_tracks.py",
            "--dataset",
            str(args.dataset),
            "--trackers",
            tracker_name,
            "--split",
            args.test_split,
            "--mode",
            args.eval_mode,
        ]
        run(cmd)

    print(f"\n[DONE] experiment={exp_dir}")
    print(f"[DONE] checkpoint={checkpoint}")
    print(f"[DONE] tracker={tracker_name}")


if __name__ == "__main__":
    main()
