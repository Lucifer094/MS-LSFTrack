from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_DATA_ROOT = Path(os.environ.get("MS_LSF_DATA_ROOT", "./data"))


def dataset_catalog(source_root: Path) -> dict[str, dict]:
    return {
        "IR-DMSTrack-v3": {
            "detector": "det_label",
            "source": source_root / "IR-DMSTrack-v3",
        },
        "GMOT-40-small-target": {
            "detector": "det_label",
            "source": source_root / "GMOT-40-small-target",
        },
        "IRSatVideo-LEO": {
            "detector": "ResUNet_RFR",
            "source": source_root / "IRSatVideo-LEO",
        },
    }


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_split(path: Path) -> list[str]:
    return [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def robocopy_dir(src: Path, dst: Path) -> str:
    """Copy directory contents, following junctions into real files when possible."""
    if not src.exists():
        return "missing_source"
    ensure_dir(dst)
    if os.name != "nt":
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return "copied"
    result = subprocess.run(
        [
            "robocopy",
            str(src),
            str(dst),
            "/E",
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
            "/NP",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Robocopy exit codes 0-7 are non-fatal.
    if result.returncode <= 7:
        return "copied"
    raise RuntimeError(f"robocopy failed: {src} -> {dst}\n{result.stderr}\n{result.stdout}")


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def copy_splits(src: Path, dst: Path) -> dict[str, list[str]]:
    ensure_dir(dst)
    splits: dict[str, list[str]] = {}
    for name in ["train", "val", "test", "all"]:
        source = src / f"{name}.txt"
        if source.exists():
            scenes = read_split(source)
            splits[name] = scenes
            copy_file(source, dst / f"{name}.txt")
    # Keep GMOT TS-AMID balance metadata if present.
    for extra in ["split_assignment.csv", "split_summary.csv", "split_metadata.json"]:
        copy_file(src / extra, dst / extra)
    return splits


def clean_existing_dataset(dst: Path, output_root: Path) -> None:
    """Remove only the four generated folders for a single clean dataset."""
    resolved = dst.resolve()
    allowed_root = output_root.resolve()
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise RuntimeError(f"Refusing to clean outside {allowed_root}: {resolved}")
    for name in ["sequence", "track_label", "detecion_label", "splits"]:
        target = dst / name
        if target.exists():
            shutil.rmtree(target)


def prepare_dataset(name: str, info: dict, output_root: Path, overwrite: bool) -> dict:
    source = Path(info["source"])
    detector = str(info["detector"])
    dst = output_root / name
    if overwrite and dst.exists():
        clean_existing_dataset(dst, output_root)
    ensure_dir(dst)

    splits = copy_splits(source / "splits", dst / "splits")
    all_scenes = sorted(set(scene for scenes in splits.values() for scene in scenes))
    if not all_scenes:
        raise RuntimeError(f"No scenes found in splits: {source / 'splits'}")

    summary: dict = {
        "dataset": name,
        "detector": detector,
        "num_scenes": len(all_scenes),
        "layout": {
            "sequence": "sequence/{scene}/image files",
            "track_label": "track_label/{scene}.txt",
            "detecion_label": "detecion_label/{split}/{scene}.txt",
            "splits": "splits/{train,val,test,all}.txt",
        },
        "missing": [],
        "split_counts": {k: len(v) for k, v in splits.items()},
        "num_images": 0,
    }

    for index, scene in enumerate(all_scenes, start=1):
        src_img = source / scene / "img1"
        dst_img = dst / "sequence" / scene
        status = robocopy_dir(src_img, dst_img)
        if status == "missing_source":
            summary["missing"].append(f"sequence:{scene}")
        img_count = count_images(dst_img)
        summary["num_images"] += img_count

        gt_src = source / scene / "gt" / "gt.txt"
        if not copy_file(gt_src, dst / "track_label" / f"{scene}.txt"):
            summary["missing"].append(f"track_label:{scene}")

        if index % 50 == 0 or index == len(all_scenes):
            print(f"[{name}] sequences {index}/{len(all_scenes)} copied", flush=True)

    det_source_root = source / "detections" / detector
    for split, scenes in splits.items():
        if split == "all":
            continue
        for scene in scenes:
            det_src = det_source_root / split / f"{scene}.txt"
            if not copy_file(det_src, dst / "detecion_label" / split / f"{scene}.txt"):
                summary["missing"].append(f"detecion_label:{split}/{scene}")

    return summary


def write_summary(output_root: Path, summaries: list[dict]) -> None:
    write_json(output_root / "clean_layout_summary.json", {"datasets": summaries})
    with (output_root / "clean_layout_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "detector",
                "num_scenes",
                "num_images",
                "missing_count",
                "missing",
            ],
        )
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    "detector": item["detector"],
                    "num_scenes": item["num_scenes"],
                    "num_images": item["num_images"],
                    "missing_count": len(item.get("missing", [])),
                    "missing": " | ".join(item.get("missing", [])),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean dataset layout with copied images.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(dataset_catalog(DEFAULT_DATA_ROOT).keys()),
        default=list(dataset_catalog(DEFAULT_DATA_ROOT).keys()),
    )
    args = parser.parse_args()

    catalog = dataset_catalog(args.source_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in args.datasets:
        summaries.append(prepare_dataset(dataset, catalog[dataset], args.output_root, args.overwrite))
    write_summary(args.output_root, summaries)
    print(json.dumps({"output_root": str(args.output_root), "datasets": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
