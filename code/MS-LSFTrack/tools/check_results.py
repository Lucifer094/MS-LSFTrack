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
from mslsftrack.mot_io import infer_scenes, read_split, resolve_detection_file
from unified_eval_suite.io import read_mot_txt, write_csv_dicts, ensure_dir


def check_txt(path: Path, require_id: bool = False) -> tuple[str, int]:
    if not path.exists():
        return "missing", 0
    arr = read_mot_txt(path)
    if arr.size == 0:
        return "empty_or_bad_format", 0
    if require_id and (arr[:, 1] < 0).all():
        return "no_valid_id", len(arr)
    return "ok", len(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check detection and tracking result completeness.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--trackers", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    split = args.split or ds.default_split
    split_file = ds.split_root / f"{split}.txt"
    scenes = read_split(split_file) if split_file.exists() else infer_scenes(ds.data_root, split)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    rows = []
    for scene in scenes:
        det_path = resolve_detection_file(ds.detection_root, split, scene)
        status, count = check_txt(det_path)
        rows.append({
            "type": "detection",
            "dataset": ds.name,
            "det_version": ds.det_version,
            "tracker": "",
            "split": split,
            "scene": scene,
            "path": str(det_path),
            "status": status,
            "rows": count,
        })
        for tracker in args.trackers:
            track_path = ds.unified_root / "tracks" / ds.det_version / tracker / split / f"{scene}.txt"
            status, count = check_txt(track_path, require_id=True)
            rows.append({
                "type": "track",
                "dataset": ds.name,
                "det_version": ds.det_version,
                "tracker": tracker,
                "split": split,
                "scene": scene,
                "path": str(track_path),
                "status": status,
                "rows": count,
            })

    out_dir = ensure_dir(args.output_dir or ds.unified_root / "analysis" / "check" / ds.det_version)
    write_csv_dicts(out_dir / f"{split}_check.csv", rows)
    summary = []
    keys = sorted({(row["type"], row["tracker"]) for row in rows})
    for typ, tracker in keys:
        sub = [row for row in rows if row["type"] == typ and row["tracker"] == tracker]
        summary.append({
            "type": typ,
            "tracker": tracker,
            "split": split,
            "total": len(sub),
            "ok": sum(row["status"] == "ok" for row in sub),
            "missing": sum(row["status"] == "missing" for row in sub),
            "bad": sum(row["status"] not in {"ok", "missing"} for row in sub),
        })
    write_csv_dicts(out_dir / f"{split}_check_summary.csv", summary)
    print(f"[DONE] check={out_dir / f'{split}_check_summary.csv'}")


if __name__ == "__main__":
    main()
