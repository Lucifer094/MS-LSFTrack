from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.config import load_dataset_config
from mslsftrack.datasets import build_real_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build association candidate cache.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset YAML/JSON config.")
    parser.add_argument("--split", default=None, help="train/val/test or dataset-specific split.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--gate-radius", type=float, default=10.0)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    split = args.split or ds.default_split
    output = args.output_dir / ds.name / ds.det_version / f"{split}.pkl"
    split_file = ds.split_root / f"{split}.txt"
    cache = build_real_cache(
        split=split,
        detection_root=ds.detection_root,
        gt_root=ds.gt_root,
        split_file=split_file if split_file.exists() else None,
        output=output,
        match_radius=args.match_radius,
        gate_radius=args.gate_radius,
        max_gap=args.max_gap,
        max_candidates=args.max_candidates,
        min_candidates=args.min_candidates,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    print(f"[CACHE] dataset={ds.name} det={ds.det_version} split={split}")
    print(f"[CACHE] output={output}")
    print(f"[CACHE] stats={cache.stats}")


if __name__ == "__main__":
    main()
