from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_ROOT = ROOT / "third_party"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from mslsftrack.config import load_dataset_config
from unified_eval_suite.io import read_gt, read_predictions, write_csv_dicts, ensure_dir
from unified_eval_suite.metrics import evaluate, prefix_metrics
from mslsftrack.mot_io import infer_scenes, read_split


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    total_gt = sum(float(r.get("num_gt_dets", 0)) for r in rows)
    total_pred = sum(float(r.get("num_pred_dets", 0)) for r in rows)
    weights = [float(r.get("num_gt_dets", 0)) for r in rows]

    def wavg(key: str) -> float:
        return sum(float(r.get(key, 0)) * w for r, w in zip(rows, weights)) / total_gt if total_gt else 0.0

    out = {"num_scenes": len(rows), "num_gt_dets": total_gt, "num_pred_dets": total_pred}
    sum_keys = [
        "Pt-TP",
        "Pt-FP",
        "Pt-FN",
        "Pt-IDs",
        "Pt-Frag",
        "Pt-IDTP",
        "Pt-IDFP",
        "Pt-IDFN",
        "IoU-TP",
        "IoU-FP",
        "IoU-FN",
        "IoU-IDs",
        "IoU-Frag",
        "IoU-IDTP",
        "IoU-IDFP",
        "IoU-IDFN",
        "MT",
        "PT",
        "ML",
        "Pred_tracks",
        "GT_tracks",
    ]
    sum_key_set = set(sum_keys)
    for key in rows[0].keys():
        if key in sum_key_set:
            continue
        if key.startswith("Pt-") or key.startswith("IoU-") or key in ["PredPurity", "PurePredRate", "AvgPredLen", "AvgGTLen"]:
            out[key] = wavg(key)
    for key in sum_keys:
        if key in rows[0]:
            out[key] = sum(float(r.get(key, 0)) for r in rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified point/IoU evaluation.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trackers", nargs="+", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--mode", choices=["point", "iou"], default="point")
    parser.add_argument("--dist-th", type=float, default=None)
    parser.add_argument("--iou-th", type=float, default=None)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    split = args.split or ds.default_split
    dist_th = ds.point_dist_th if args.dist_th is None else args.dist_th
    iou_th = ds.iou_th if args.iou_th is None else args.iou_th
    split_file = ds.split_root / f"{split}.txt"
    scenes = read_split(split_file) if split_file.exists() else infer_scenes(ds.data_root, split)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    for tracker_name in args.trackers:
        rows = []
        for scene in scenes:
            gt = read_gt(ds.gt_root, scene, split)
            pred = read_predictions(ds.unified_root, ds.det_version, tracker_name, split, scene)
            raw = evaluate(gt, pred, mode=args.mode, dist_th=dist_th, iou_th=iou_th)
            if args.mode == "point":
                metrics = prefix_metrics(raw, "Pt", "Pt-MOTP_dist")
                eval_name = f"point_{dist_th:g}px"
            else:
                metrics = prefix_metrics(raw, "IoU", "IoU-MOTP")
                eval_name = f"iou_{iou_th:g}"
            metrics.update(
                {
                    "tracker": tracker_name,
                    "det_version": ds.det_version,
                    "split": split,
                    "scene": scene,
                    "eval_mode": args.mode,
                    "dist_th": dist_th,
                    "iou_th": iou_th,
                }
            )
            rows.append(metrics)
        out_dir = ensure_dir(ds.unified_root / "eval" / eval_name / ds.det_version / tracker_name)
        write_csv_dicts(out_dir / f"{split}_per_scene.csv", rows)
        summary = aggregate(rows)
        summary.update({"tracker": tracker_name, "det_version": ds.det_version, "split": split, "eval_mode": args.mode})
        write_csv_dicts(out_dir / f"{split}_summary.csv", [summary])
        print(f"[DONE] {args.mode} eval {tracker_name}: {out_dir / f'{split}_summary.csv'}")


if __name__ == "__main__":
    main()
