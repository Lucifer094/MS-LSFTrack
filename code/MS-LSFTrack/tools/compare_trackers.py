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
from unified_eval_suite.io import read_csv_dicts, write_csv_dicts, ensure_dir
from unified_eval_suite.runtime import merge_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Create unified comparison table.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trackers", nargs="+", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--dist-th", type=float, default=None)
    parser.add_argument("--iou-th", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    split = args.split or ds.default_split
    dist_th = ds.point_dist_th if args.dist_th is None else args.dist_th
    iou_th = ds.iou_th if args.iou_th is None else args.iou_th
    rows = []
    for tracker_name in args.trackers:
        point_path = ds.unified_root / "eval" / f"point_{dist_th:g}px" / ds.det_version / tracker_name / f"{split}_summary.csv"
        iou_path = ds.unified_root / "eval" / f"iou_{iou_th:g}" / ds.det_version / tracker_name / f"{split}_summary.csv"
        point_rows = read_csv_dicts(point_path)
        iou_rows = read_csv_dicts(iou_path)
        runtime_rows = read_csv_dicts(ds.unified_root / "runtime" / ds.det_version / tracker_name / f"{split}_summary.csv")
        row = {
            "dataset": ds.name,
            "det_version": ds.det_version,
            "split": split,
            "tracker": tracker_name,
        }
        if point_rows:
            for key in [
                "num_scenes",
                "num_gt_dets",
                "num_pred_dets",
                "Pt-HOTA",
                "Pt-DetA",
                "Pt-AssA",
                "Pt-AssR",
                "Pt-MOTA",
                "Pt-IDF1",
                "Pt-IDTP",
                "Pt-IDFP",
                "Pt-IDFN",
                "Pt-IDP",
                "Pt-IDR",
                "Pt-TP",
                "Pt-FP",
                "Pt-FN",
                "Pt-IDs",
                "Pt-Frag",
                "Pt-MOTP_dist",
                "Pred_tracks",
                "GT_tracks",
                "AvgPredLen",
                "AvgGTLen",
                "PredPurity",
            ]:
                row[key] = point_rows[0].get(key, "")
        else:
            print(f"[WARN] missing point summary: {point_path}")
        if iou_rows:
            for key in [
                "IoU-HOTA",
                "IoU-DetA",
                "IoU-AssA",
                "IoU-AssR",
                "IoU-MOTA",
                "IoU-IDF1",
                "IoU-IDTP",
                "IoU-IDFP",
                "IoU-IDFN",
                "IoU-IDP",
                "IoU-IDR",
                "IoU-TP",
                "IoU-FP",
                "IoU-FN",
                "IoU-IDs",
                "IoU-Frag",
                "IoU-MOTP",
            ]:
                row[key] = iou_rows[0].get(key, "")
        row = merge_runtime(row, runtime_rows)
        rows.append(row)
    out_path = args.output or ds.unified_root / "comparison" / "paper_tables" / f"{ds.det_version}_{split}_paper_table.csv"
    ensure_dir(out_path.parent)
    write_csv_dicts(out_path, rows)
    print(f"[DONE] table={out_path}")


if __name__ == "__main__":
    main()
