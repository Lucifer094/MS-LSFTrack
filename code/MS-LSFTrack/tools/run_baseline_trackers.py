from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.config import load_dataset_config
from mslsftrack.mot_io import infer_scenes, read_split, resolve_detection_file

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALL_TRACKERS = [
    "bytetrack",
    "ocsort",
    "sfsort",
    "strongsort",
    "botsort",
    "deepocsort",
    "hybridsort",
    "boosttrack",
    "occluboost",
]
REID_TRACKERS = {"strongsort", "botsort", "deepocsort", "hybridsort", "occluboost"}
PUBLIC_REID_MODEL_NAMES = {
    "osnet_x0_25_msmt17.pt",
    "osnet_x0_5_msmt17.pt",
    "osnet_x1_0_msmt17.pt",
    "lmbn_n_duke.pt",
    "clip_market1501.pt",
}
TRACKER_ALIASES = {
    "occluboost": "occluboost",
    "occlu_boost": "occluboost",
}


def normalize_tracker_name(name: str) -> str:
    lowered = name.lower()
    return TRACKER_ALIASES.get(lowered, lowered)


def import_boxmot() -> dict[str, Any]:
    create_tracker = None
    tracker_mapping = {}
    reid_trackers = set(REID_TRACKERS)
    try:
        from boxmot.trackers.tracker_zoo import create_tracker as create_tracker
        from boxmot.trackers.tracker_zoo import REID_TRACKERS as zoo_reid_trackers
        from boxmot.trackers.tracker_zoo import TRACKER_MAPPING as tracker_mapping

        reid_trackers.update(str(name).lower() for name in zoo_reid_trackers)
        tracker_mapping = {str(key).lower(): value for key, value in tracker_mapping.items()}
    except Exception:
        pass

    try:
        from boxmot import ByteTrack, OcSort, StrongSort, BotSort, DeepOcSort, HybridSort, BoostTrack
    except Exception as exc:
        if create_tracker is None:
            raise ImportError("boxmot is not installed or cannot be imported") from exc
        ByteTrack = OcSort = StrongSort = BotSort = DeepOcSort = HybridSort = BoostTrack = None

    sfsort_cls = None
    sfsort_error = None
    try:
        from boxmot.trackers.sfsort.sfsort import SFSORT as sfsort_cls
    except Exception as exc1:
        try:
            from boxmot.trackers.sfsort.sfsort import SfSort as sfsort_cls
        except Exception as exc2:
            sfsort_error = f"{type(exc1).__name__}: {exc1} | {type(exc2).__name__}: {exc2}"

    return {
        "_create_tracker": create_tracker,
        "_tracker_mapping": tracker_mapping,
        "_reid_trackers": reid_trackers,
        "bytetrack": ByteTrack,
        "ocsort": OcSort,
        "sfsort": sfsort_cls,
        "strongsort": StrongSort,
        "botsort": BotSort,
        "deepocsort": DeepOcSort,
        "hybridsort": HybridSort,
        "boosttrack": BoostTrack,
        "occluboost": None,
        "_sfsort_error": sfsort_error,
    }


def resolve_reid_weights(value: str) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_file():
        return str(path)
    cached = Path(os.environ.get("MS_LSF_WEIGHT_ROOT", "./weights")) / "reid" / value
    if cached.is_file():
        return str(cached)
    return value


def boxmot_tracker_defaults(name: str) -> dict[str, Any]:
    try:
        from boxmot.trackers.tracker_zoo import get_tracker_config
    except Exception:
        return {}
    config_path = get_tracker_config(name)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    defaults = {}
    for param, details in config.items():
        if isinstance(details, dict) and "default" in details:
            defaults[param] = details["default"]
    return defaults


def boxmot_evolve_params(name: str, args: argparse.Namespace) -> dict[str, Any] | None:
    params: dict[str, Any] | None = None
    if name == "deepocsort" and args.deepocsort_cmc_off:
        params = boxmot_tracker_defaults(name)
        params["cmc_off"] = True
    if name == "boosttrack" and args.boosttrack_no_reid:
        params = boxmot_tracker_defaults(name)
        params["with_reid"] = False
    return params


def create_tracker_from_zoo(name: str, classes: dict[str, Any], args: argparse.Namespace):
    create_tracker = classes.get("_create_tracker")
    if create_tracker is None:
        return None
    if name not in classes.get("_tracker_mapping", {}):
        return None
    kwargs = {
        "tracker_type": name,
        "device": args.device,
        "half": args.half,
    }
    if tracker_needs_reid(name, args.boosttrack_no_reid, classes):
        kwargs["reid_weights"] = resolve_reid_weights(args.reid_weights)
    evolve_param_dict = boxmot_evolve_params(name, args)
    if evolve_param_dict is not None:
        kwargs["evolve_param_dict"] = evolve_param_dict
    signature = inspect.signature(create_tracker)
    if "tracker_backend" in signature.parameters:
        kwargs["tracker_backend"] = "python"
    return create_tracker(**kwargs)


def make_tracker(name: str, classes: dict[str, Any], args: argparse.Namespace):
    name = normalize_tracker_name(name)
    zoo_tracker = create_tracker_from_zoo(name, classes, args)
    if zoo_tracker is not None:
        return zoo_tracker
    if name == "occluboost":
        raise ImportError("occluboost is not available in the installed BoxMOT; upgrade boxmot to a version that includes OccluBoost.")
    if name == "sfsort" and classes["sfsort"] is None:
        raise ImportError(f"SFSORT import failed: {classes['_sfsort_error']}")
    if name in {"bytetrack", "ocsort", "sfsort"}:
        return classes[name]()
    if name in REID_TRACKERS:
        if not args.reid_weights:
            raise ValueError(f"{name} requires --reid-weights")
        if name == "deepocsort":
            return classes[name](
                reid_weights=resolve_reid_weights(args.reid_weights),
                device=args.device,
                half=args.half,
                cmc_off=args.deepocsort_cmc_off,
            )
        return classes[name](reid_weights=resolve_reid_weights(args.reid_weights), device=args.device, half=args.half)
    if name == "boosttrack":
        if args.boosttrack_no_reid:
            return classes[name](with_reid=False)
        if not args.reid_weights:
            raise ValueError("boosttrack with ReID requires --reid-weights; use --boosttrack-no-reid to disable")
        return classes[name](reid_weights=resolve_reid_weights(args.reid_weights), device=args.device, half=args.half, with_reid=True)
    raise ValueError(f"unknown tracker: {name}")


def tracker_needs_reid(name: str, boosttrack_no_reid: bool, classes: dict[str, Any] | None = None) -> bool:
    reid_trackers = set(REID_TRACKERS)
    if classes is not None:
        reid_trackers.update(classes.get("_reid_trackers", set()))
    return name in reid_trackers or (name == "boosttrack" and not boosttrack_no_reid)


def precheck_tracker(name: str, args: argparse.Namespace, classes: dict[str, Any] | None) -> tuple[bool, str]:
    name = normalize_tracker_name(name)
    if name not in ALL_TRACKERS:
        return False, f"unknown tracker: {name}"
    if classes is None:
        return False, "boxmot import failed"
    has_zoo_tracker = name in classes.get("_tracker_mapping", {})
    if name == "occluboost" and not has_zoo_tracker and classes.get(name) is None:
        return False, "occluboost is not available in the installed BoxMOT; upgrade boxmot to a version that includes OccluBoost"
    if name == "sfsort" and not has_zoo_tracker and classes["sfsort"] is None:
        return False, f"SFSORT import failed: {classes['_sfsort_error']}"
    if not has_zoo_tracker and classes.get(name) is None:
        return False, f"{name} is not available in the installed BoxMOT"
    if tracker_needs_reid(name, args.boosttrack_no_reid, classes):
        if not args.reid_weights:
            return False, "missing --reid-weights"
        resolved = resolve_reid_weights(args.reid_weights)
        is_model_name = args.reid_weights in PUBLIC_REID_MODEL_NAMES
        looks_like_path = "/" in args.reid_weights or "\\" in args.reid_weights or args.reid_weights.startswith(".")
        if looks_like_path and not is_model_name and not Path(resolved).is_file():
            return False, f"ReID weights not found: {args.reid_weights}"
    return True, "ok"


def resolve_image_dir(data_root: Path, scene: str, split: str | None = None) -> Path:
    candidates = [
        data_root / "sequence" / scene,
        data_root / scene / "img1",
        data_root / scene / "images",
        data_root / "images" / scene,
    ]
    if split:
        candidates.extend([
            data_root / split / scene / "img1",
            data_root / split / scene / "images",
            data_root / split / scene,
        ])
    for path in candidates:
        if path.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in path.iterdir() if p.is_file()):
            return path
    return candidates[0]


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMG_EXTS)
    if not images:
        raise ValueError(f"no images found in {image_dir}")
    return images


def read_detection_rows(det_file: Path) -> np.ndarray:
    if not det_file.exists() or det_file.stat().st_size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    rows = []
    with det_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part for part in line.replace(" ", ",").split(",") if part != ""]
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) >= 7:
                frame, x, y, w, h, score = values[0], values[2], values[3], values[4], values[5], values[6]
            elif len(values) >= 6:
                frame, x, y, w, h, score = values[:6]
            else:
                continue
            rows.append([frame, x, y, w, h, score, 0.0])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 7), dtype=np.float32)


def detections_by_frame(det_file: Path) -> dict[int, np.ndarray]:
    rows = read_detection_rows(det_file)
    grouped: dict[int, list[list[float]]] = {}
    for frame, x, y, w, h, score, cls in rows:
        grouped.setdefault(int(frame), []).append([x, y, x + w, y + h, score, cls])
    return {frame: np.asarray(values, dtype=np.float32) for frame, values in grouped.items()}


def clip_detections_to_frame(dets: np.ndarray, frame_shape: tuple[int, ...], min_box_size: float) -> tuple[np.ndarray, int]:
    """Clip xyxy detections to the image and drop boxes that cannot produce a crop."""
    if dets.size == 0:
        return np.zeros((0, 6), dtype=np.float32), 0

    clipped = np.asarray(dets, dtype=np.float32).copy()
    if clipped.ndim == 1:
        clipped = clipped[None, :]
    if clipped.shape[1] < 6:
        raise ValueError(f"expected detections with at least 6 columns, got shape={clipped.shape}")

    height, width = frame_shape[:2]
    finite_mask = np.isfinite(clipped[:, :6]).all(axis=1)
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width))
    clipped[:, 2] = np.clip(clipped[:, 2], 0.0, float(width))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height))
    clipped[:, 3] = np.clip(clipped[:, 3], 0.0, float(height))

    box_widths = clipped[:, 2] - clipped[:, 0]
    box_heights = clipped[:, 3] - clipped[:, 1]
    valid_mask = finite_mask & (box_widths >= min_box_size) & (box_heights >= min_box_size)
    dropped = int(clipped.shape[0] - int(valid_mask.sum()))
    if not valid_mask.any():
        return np.zeros((0, 6), dtype=np.float32), dropped
    return clipped[valid_mask].astype(np.float32, copy=False), dropped


def tracks_to_mot_lines(tracks: Any, frame_id: int) -> list[str]:
    arr = np.asarray(tracks)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr[None, :]
    lines = []
    for row in arr:
        if len(row) < 5:
            continue
        x1, y1, x2, y2 = [float(value) for value in row[:4]]
        track_id = int(round(float(row[4])))
        score = float(row[5]) if len(row) >= 6 else 1.0
        lines.append(
            f"{frame_id},{track_id},{x1:.6f},{y1:.6f},{max(0.0, x2 - x1):.6f},{max(0.0, y2 - y1):.6f},{score:.6f},-1,-1,-1"
        )
    return lines


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_runtime_summary(rows: list[dict[str, Any]], path: Path, tracker: str, det_version: str, split: str) -> None:
    total_frames = sum(int(float(row.get("num_frames", 0))) for row in rows)
    total_dets = sum(int(float(row.get("num_dets", 0))) for row in rows)
    total_valid_dets = sum(int(float(row.get("num_valid_dets", row.get("num_dets", 0)))) for row in rows)
    total_dropped_dets = sum(int(float(row.get("num_dropped_dets", 0))) for row in rows)
    total_outputs = sum(int(float(row.get("num_outputs", 0))) for row in rows)
    total_time = sum(float(row.get("total_time_sec", 0.0)) for row in rows)
    update_time = sum(float(row.get("update_time_sec", 0.0)) for row in rows)
    summary = {
        "tracker": tracker,
        "det_version": det_version,
        "split": split,
        "num_scenes": len(rows),
        "total_frames": total_frames,
        "total_dets": total_dets,
        "total_valid_dets": total_valid_dets,
        "total_dropped_dets": total_dropped_dets,
        "total_outputs": total_outputs,
        "total_time_sec": f"{total_time:.6f}",
        "update_time_sec": f"{update_time:.6f}",
        "fps_total": f"{total_frames / max(total_time, 1e-12):.6f}" if total_frames else "0.000000",
        "fps_update": f"{total_frames / max(update_time, 1e-12):.6f}" if total_frames else "0.000000",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def run_scene(
    args: argparse.Namespace,
    tracker_name: str,
    tracker: Any,
    scene: str,
    image_files: list[Path],
    det_file: Path,
    out_file: Path,
) -> dict[str, Any]:
    dets = detections_by_frame(det_file)
    num_dets = sum(len(values) for values in dets.values())
    tmp_file = out_file.with_suffix(out_file.suffix + ".partial")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if tmp_file.exists():
        tmp_file.unlink()

    update_time = 0.0
    num_outputs = 0
    num_valid_dets = 0
    num_dropped_dets = 0
    t0_total = time.perf_counter()
    with tmp_file.open("w", encoding="utf-8") as handle:
        for frame_id, image_path in enumerate(image_files, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"failed to read image: {image_path}")
            frame_dets = dets.get(frame_id, np.zeros((0, 6), dtype=np.float32))
            frame_dets, dropped_dets = clip_detections_to_frame(frame_dets, frame.shape, args.min_box_size)
            num_valid_dets += len(frame_dets)
            num_dropped_dets += dropped_dets
            t0_update = time.perf_counter()
            tracks = tracker.update(frame_dets, frame)
            update_time += time.perf_counter() - t0_update
            lines = tracks_to_mot_lines(tracks, frame_id)
            num_outputs += len(lines)
            if lines:
                handle.write("\n".join(lines) + "\n")
            if args.flush_every > 0 and frame_id % args.flush_every == 0:
                handle.flush()
                os.fsync(handle.fileno())
    tmp_file.replace(out_file)
    total_time = time.perf_counter() - t0_total
    return {
        "scene": scene,
        "num_frames": len(image_files),
        "num_dets": num_dets,
        "num_valid_dets": num_valid_dets,
        "num_dropped_dets": num_dropped_dets,
        "num_outputs": num_outputs,
        "total_time_sec": f"{total_time:.6f}",
        "update_time_sec": f"{update_time:.6f}",
        "fps_total": f"{len(image_files) / max(total_time, 1e-12):.6f}",
        "fps_update": f"{len(image_files) / max(update_time, 1e-12):.6f}",
        "out_file": str(out_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BoxMOT baseline trackers on clean-layout datasets.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--trackers", nargs="+", default=ALL_TRACKERS)
    parser.add_argument("--reid-weights", default="")
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--boosttrack-no-reid", action="store_true")
    parser.add_argument(
        "--deepocsort-cmc-off",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable DeepOCSORT camera motion compensation; avoids OpenCV cornerSubPix failures on tiny low-texture frames.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--min-box-size", type=float, default=1.0, help="Drop clipped detections smaller than this many pixels.")
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Only validate imports, paths and output locations.")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping unavailable trackers.")
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    split = args.split or ds.default_split
    split_file = ds.split_root / f"{split}.txt"
    scenes = read_split(split_file) if split_file.exists() else infer_scenes(ds.data_root, split)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    try:
        classes = import_boxmot()
        boxmot_error = ""
    except Exception as exc:
        classes = None
        boxmot_error = f"{type(exc).__name__}: {exc}"
        print(f"[WARN] {boxmot_error}")

    trackers = [normalize_tracker_name(name) for name in args.trackers]
    checks = []
    for name in trackers:
        ok, message = precheck_tracker(name, args, classes)
        checks.append({"tracker": name, "status": "ok" if ok else "unavailable", "message": message})
        if not ok and args.strict:
            raise RuntimeError(f"{name}: {message}")

    for scene in scenes:
        image_dir = resolve_image_dir(ds.data_root, scene, split)
        det_file = resolve_detection_file(ds.detection_root, split, scene)
        if not image_dir.exists() or not det_file.exists():
            msg = f"scene precheck failed: scene={scene}, image_dir={image_dir}, det_file={det_file}"
            if args.strict:
                raise FileNotFoundError(msg)
            print(f"[WARN] {msg}")

    run_config = {
        **vars(args),
        "dataset_name": ds.name,
        "det_version": ds.det_version,
        "split": split,
        "num_scenes": len(scenes),
        "boxmot_error": boxmot_error,
        "tracker_checks": checks,
    }
    log_dir = ds.unified_root / "logs" / ds.det_version
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = log_dir / f"run_baseline_trackers_{split}_config.json"
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"split": split, "num_scenes": len(scenes), "tracker_checks": checks}, ensure_ascii=False, indent=2))

    if args.dry_run:
        print(f"[DONE] dry-run config={config_path}")
        return

    if classes is None:
        raise ImportError("Cannot run baselines without boxmot. Install the optional baseline dependency first.")

    per_scene_fields = [
        "scene",
        "num_frames",
        "num_dets",
        "num_valid_dets",
        "num_dropped_dets",
        "num_outputs",
        "total_time_sec",
        "update_time_sec",
        "fps_total",
        "fps_update",
        "out_file",
    ]
    progress_fields = ["time", "tracker", "det_version", "split", "scene", "status", "message"]

    for tracker_name in trackers:
        ok, message = precheck_tracker(tracker_name, args, classes)
        runtime_dir = ds.unified_root / "runtime" / ds.det_version / tracker_name
        per_scene_csv = runtime_dir / f"{split}_per_scene.csv"
        summary_csv = runtime_dir / f"{split}_summary.csv"
        progress_csv = log_dir / f"{tracker_name}_{split}_progress.csv"
        if args.overwrite:
            for path in [per_scene_csv, summary_csv, progress_csv]:
                if path.exists():
                    path.unlink()
        if not ok:
            append_csv(progress_csv, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tracker": tracker_name,
                "det_version": ds.det_version,
                "split": split,
                "scene": "*",
                "status": "SKIP_TRACKER",
                "message": message,
            }, progress_fields)
            print(f"[SKIP] {tracker_name}: {message}")
            continue

        rows_done = read_csv_rows(per_scene_csv)
        done_scenes = {row.get("scene") for row in rows_done}
        for scene in scenes:
            out_file = ds.unified_root / "tracks" / ds.det_version / tracker_name / split / f"{scene}.txt"
            if (not args.overwrite) and scene in done_scenes and out_file.exists() and out_file.stat().st_size > 0:
                print(f"[SKIP] {tracker_name} {split}/{scene}: already done")
                continue
            image_files = list_images(resolve_image_dir(ds.data_root, scene, split))
            det_file = resolve_detection_file(ds.detection_root, split, scene)
            tracker = make_tracker(tracker_name, classes, args)
            append_csv(progress_csv, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tracker": tracker_name,
                "det_version": ds.det_version,
                "split": split,
                "scene": scene,
                "status": "START",
                "message": str(out_file),
            }, progress_fields)
            try:
                row = run_scene(args, tracker_name, tracker, scene, image_files, det_file, out_file)
            except Exception as exc:
                append_csv(progress_csv, {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tracker": tracker_name,
                    "det_version": ds.det_version,
                    "split": split,
                    "scene": scene,
                    "status": "ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                }, progress_fields)
                if args.strict:
                    raise
                print(f"[ERROR] {tracker_name} {split}/{scene}: {type(exc).__name__}: {exc}")
                continue
            append_csv(per_scene_csv, row, per_scene_fields)
            rows_done.append(row)
            append_csv(progress_csv, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tracker": tracker_name,
                "det_version": ds.det_version,
                "split": split,
                "scene": scene,
                "status": "DONE",
                "message": (
                    f"frames={row['num_frames']} dets={row['num_dets']} valid_dets={row['num_valid_dets']} "
                    f"dropped_dets={row['num_dropped_dets']} outputs={row['num_outputs']} fps_update={row['fps_update']}"
                ),
            }, progress_fields)
            print(f"[OK] {tracker_name} {split}/{scene}: outputs={row['num_outputs']} fps_update={row['fps_update']}")
        write_runtime_summary(read_csv_rows(per_scene_csv), summary_csv, tracker_name, ds.det_version, split)
        print(f"[DONE] {tracker_name}: runtime={summary_csv}")


if __name__ == "__main__":
    main()
