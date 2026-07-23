from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.config import load_dataset_config


SPLITS = ("train", "val", "test")
DIFFICULTY_3_ORDER = (
    "dense_hard_lt_0p75x",
    "medium_0p75_1p25x",
    "sparse_relative_ge_1p25x",
)
DIFFICULTY_6_ORDER = (
    "extremely_hard_lt_0p5x",
    "very_hard_0p5_0p75x",
    "hard_0p75_1p0x",
    "medium_1p0_1p25x",
    "easyish_1p25_1p5x",
    "sparse_easy_ge_1p5x",
)


def read_split(split_root: Path, split: str) -> list[str]:
    path = split_root / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_scene(gt_root: Path, scene: str) -> list[tuple[int, int, float, float, float, float]]:
    rows = []
    path = gt_root / "track_label" / f"{scene}.txt"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = [float(value) for value in line.replace(",", " ").split()]
        frame = int(parts[0])
        track_id = int(parts[1])
        x, y, w, h = parts[2], parts[3], parts[4], parts[5]
        if w > 0 and h > 0:
            rows.append((frame, track_id, x, y, w, h))
    return rows


def rect_min_distances(boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    n = arr.shape[0]
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 0] + arr[:, 2]
    y2 = arr[:, 1] + arr[:, 3]
    dx = np.maximum.reduce(
        [
            x1[None, :] - x2[:, None],
            x1[:, None] - x2[None, :],
            np.zeros((n, n), dtype=np.float64),
        ]
    )
    dy = np.maximum.reduce(
        [
            y1[None, :] - y2[:, None],
            y1[:, None] - y2[None, :],
            np.zeros((n, n), dtype=np.float64),
        ]
    )
    dist = np.hypot(dx, dy)
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def qstats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            key: float("nan")
            for key in ("n", "mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "p95", "max")
        }
    qs = np.percentile(arr, [0, 10, 25, 50, 75, 90, 95, 100])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(qs[0]),
        "p10": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p90": float(qs[5]),
        "p95": float(qs[6]),
        "max": float(qs[7]),
    }


def difficulty_3(ratio: float) -> str:
    if ratio < 0.75:
        return "dense_hard_lt_0p75x"
    if ratio < 1.25:
        return "medium_0p75_1p25x"
    return "sparse_relative_ge_1p25x"


def difficulty_6(ratio: float) -> str:
    if ratio < 0.5:
        return "extremely_hard_lt_0p5x"
    if ratio < 0.75:
        return "very_hard_0p5_0p75x"
    if ratio < 1.0:
        return "hard_0p75_1p0x"
    if ratio < 1.25:
        return "medium_1p0_1p25x"
    if ratio < 1.5:
        return "easyish_1p25_1p5x"
    return "sparse_easy_ge_1p5x"


def scene_stats(gt_root: Path, split: str, scene: str) -> dict[str, float | int | str]:
    rows = load_scene(gt_root, scene)
    by_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    widths = []
    heights = []
    avg_sides = []
    sqrt_areas = []
    diagonals = []
    areas = []
    for frame, _, x, y, w, h in rows:
        by_frame[frame].append((x, y, w, h))
        widths.append(w)
        heights.append(h)
        avg_sides.append((w + h) / 2.0)
        area = w * h
        areas.append(area)
        sqrt_areas.append(math.sqrt(area))
        diagonals.append(math.hypot(w, h))

    frame_amids = []
    frame_min_dists = []
    zero_min_dist_frames = 0
    for boxes in by_frame.values():
        if len(boxes) < 2:
            continue
        nearest = rect_min_distances(boxes)
        frame_amids.append(float(nearest.mean()))
        min_dist = float(nearest.min())
        frame_min_dists.append(min_dist)
        if min_dist <= 1e-9:
            zero_min_dist_frames += 1

    return {
        "split": split,
        "scene": scene,
        "frames": len(by_frame),
        "multi_target_frames": len(frame_amids),
        "gt_boxes": len(rows),
        "mean_targets_per_frame": len(rows) / len(by_frame) if by_frame else float("nan"),
        "mean_width": float(np.mean(widths)) if widths else float("nan"),
        "mean_height": float(np.mean(heights)) if heights else float("nan"),
        "mean_avg_side": float(np.mean(avg_sides)) if avg_sides else float("nan"),
        "mean_sqrt_area": float(np.mean(sqrt_areas)) if sqrt_areas else float("nan"),
        "mean_diagonal": float(np.mean(diagonals)) if diagonals else float("nan"),
        "mean_area": float(np.mean(areas)) if areas else float("nan"),
        "ts_amid_box": float(np.mean(frame_amids)) if frame_amids else float("nan"),
        "frame_min_box_dist_mean": float(np.mean(frame_min_dists)) if frame_min_dists else float("nan"),
        "zero_min_dist_frame_rate": zero_min_dist_frames / len(frame_amids) if frame_amids else float("nan"),
    }


def collect_box_values(gt_root: Path, scenes: list[str]) -> dict[str, list[float]]:
    values = {
        "width": [],
        "height": [],
        "avg_side": [],
        "sqrt_area": [],
        "diagonal": [],
        "area": [],
    }
    for scene in scenes:
        for _, _, _, _, w, h in load_scene(gt_root, scene):
            area = w * h
            values["width"].append(w)
            values["height"].append(h)
            values["avg_side"].append((w + h) / 2.0)
            values["sqrt_area"].append(math.sqrt(area))
            values["diagonal"].append(math.hypot(w, h))
            values["area"].append(area)
    return values


def format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def write_scene_lists(out_dir: Path, rows: list[dict], column: str, order: tuple[str, ...]) -> None:
    scene_dir = out_dir / "scene_lists"
    scene_dir.mkdir(parents=True, exist_ok=True)
    for level in order:
        scenes = [str(row["scene"]) for row in rows if row[column] == level]
        (scene_dir / f"test_{level}.txt").write_text("\n".join(scenes) + "\n", encoding="utf-8")


def plot_histograms(out_dir: Path, dataset_name: str, test_rows: list[dict], split_summary: list[dict]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ts = np.asarray([float(row["ts_amid_box"]) for row in test_rows], dtype=np.float64)
    ratio = np.asarray([float(row["ts_amid_over_test_mean_sqrt_area"]) for row in test_rows], dtype=np.float64)
    targets = np.asarray([float(row["mean_targets_per_frame"]) for row in test_rows], dtype=np.float64)

    plt.figure(figsize=(7, 4.5))
    plt.hist(ts, bins=18, color="#4C78A8", edgecolor="white")
    for x, label in [(np.percentile(ts, 25), "P25"), (np.percentile(ts, 50), "P50"), (np.percentile(ts, 75), "P75")]:
        plt.axvline(x, color="#E45756", linestyle="--", linewidth=1)
        plt.text(x, plt.ylim()[1] * 0.92, label, rotation=90, va="top", ha="right", fontsize=8)
    plt.xlabel("Box-based TS-AMID (px)")
    plt.ylabel("Scenes")
    plt.title(f"{dataset_name} Test TS-AMID Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "test_ts_amid_box_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(ratio, bins=18, color="#59A14F", edgecolor="white")
    for x in [0.5, 0.75, 1.0, 1.25, 1.5]:
        plt.axvline(x, color="#E15759", linestyle="--", linewidth=1)
        plt.text(x, plt.ylim()[1] * 0.92, f"{x:g}x", rotation=90, va="top", ha="right", fontsize=8)
    plt.xlabel("TS-AMID / mean sqrt(w*h)")
    plt.ylabel("Scenes")
    plt.title(f"{dataset_name} Test Normalized Scene Difficulty")
    plt.tight_layout()
    plt.savefig(fig_dir / "test_ts_amid_over_target_size_hist.png", dpi=200)
    plt.close()

    counts = Counter(str(row["difficulty_3"]) for row in test_rows)
    labels = ["Dense / hard", "Medium", "Sparse-relative"]
    values = [counts[level] for level in DIFFICULTY_3_ORDER]
    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, values, color=["#D95F02", "#7570B3", "#1B9E77"])
    plt.ylabel("Scenes")
    plt.title(f"{dataset_name} Test Difficulty Groups")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value + 0.5, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(fig_dir / "test_difficulty_3_counts.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    colors = {
        "dense_hard_lt_0p75x": "#D95F02",
        "medium_0p75_1p25x": "#7570B3",
        "sparse_relative_ge_1p25x": "#1B9E77",
    }
    for level, label in zip(DIFFICULTY_3_ORDER, labels):
        xs = [float(row["ts_amid_box"]) for row in test_rows if row["difficulty_3"] == level]
        ys = [float(row["mean_targets_per_frame"]) for row in test_rows if row["difficulty_3"] == level]
        plt.scatter(xs, ys, s=34, alpha=0.78, label=label, color=colors[level])
    plt.xlabel("Box-based TS-AMID (px)")
    plt.ylabel("Mean targets per frame")
    plt.title("Scene Density vs Target Count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "test_ts_amid_vs_targets_scatter.png", dpi=200)
    plt.close()

    split_names = [str(row["split"]) for row in split_summary]
    means = [float(row["mean_ts_amid_box"]) for row in split_summary]
    medians = [float(row["p50_ts_amid_box"]) for row in split_summary]
    x = np.arange(len(split_names))
    width = 0.35
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(x - width / 2, means, width, label="Mean", color="#4C78A8")
    plt.bar(x + width / 2, medians, width, label="Median", color="#F58518")
    plt.xticks(x, split_names)
    plt.ylabel("Box-based TS-AMID (px)")
    plt.title("TS-AMID by Split")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "split_ts_amid_box_summary.png", dpi=200)
    plt.close()


def plot_box_size_distributions(
    out_dir: Path,
    dataset_name: str,
    gt_root: Path,
    scenes_by_split: dict[str, list[str]],
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    box_values = {split: collect_box_values(gt_root, scenes) for split, scenes in scenes_by_split.items()}
    metrics = [
        ("width", "Box width (px)", "box_width_hist.png"),
        ("height", "Box height (px)", "box_height_hist.png"),
        ("sqrt_area", "Equivalent target size sqrt(w*h) (px)", "box_sqrt_area_hist.png"),
        ("area", "Box area (px^2)", "box_area_hist.png"),
        ("diagonal", "Box diagonal (px)", "box_diagonal_hist.png"),
    ]

    for key, xlabel, filename in metrics:
        plt.figure(figsize=(7.2, 4.6))
        for split, color in zip(SPLITS, ["#4C78A8", "#F58518", "#54A24B"]):
            values = np.asarray(box_values[split][key], dtype=np.float64)
            plt.hist(
                values,
                bins=30,
                histtype="step",
                linewidth=1.8,
                density=True,
                label=f"{split} mean={values.mean():.2f}",
                color=color,
            )
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.title(f"{dataset_name} GT {xlabel} Distribution")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=200)
        plt.close()

    test = box_values["test"]
    width = np.asarray(test["width"], dtype=np.float64)
    height = np.asarray(test["height"], dtype=np.float64)
    plt.figure(figsize=(5.6, 5.2))
    plt.hist2d(width, height, bins=[np.arange(1.5, 18.6, 1.0), np.arange(1.5, 18.6, 1.0)], cmap="viridis")
    plt.colorbar(label="GT boxes")
    plt.xlabel("Box width (px)")
    plt.ylabel("Box height (px)")
    plt.title(f"{dataset_name} Test Box Width-Height Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "test_box_width_height_heatmap.png", dpi=200)
    plt.close()

    labels = ["Width", "Height", "sqrt(area)", "Diagonal"]
    data = [test["width"], test["height"], test["sqrt_area"], test["diagonal"]]
    plt.figure(figsize=(7.4, 4.6))
    parts = plt.violinplot(data, showmeans=True, showmedians=True, widths=0.82)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_alpha(0.45)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.ylabel("Pixels")
    plt.title(f"{dataset_name} Test Target Size Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "test_target_size_violin.png", dpi=200)
    plt.close()


def make_summary(
    out_dir: Path,
    dataset_name: str,
    threshold: float,
    split_summary: list[dict],
    test_rows: list[dict],
) -> None:
    ranges = {
        "dense_hard_lt_0p75x": ("Dense / hard", "< 0.75 x target size", f"< {0.75 * threshold:.2f} px"),
        "medium_0p75_1p25x": ("Medium", "0.75-1.25 x target size", f"{0.75 * threshold:.2f}-{1.25 * threshold:.2f} px"),
        "sparse_relative_ge_1p25x": ("Sparse-relative / easy", ">= 1.25 x target size", f">= {1.25 * threshold:.2f} px"),
    }

    lines = [
        f"# {dataset_name} TS-AMID Difficulty Analysis",
        "",
        "## Conventions",
        "",
        "- TS-AMID is computed only from GT bounding-box region distances.",
        "- Center distance is not used.",
        "- Target size is represented by `sqrt(w*h)`.",
        "- Difficulty thresholds use the test split mean `sqrt(w*h)`.",
        "",
        "## Target Size Reference",
        "",
        f"- Test mean `sqrt(w*h)`: `{threshold:.4f}` px",
        f"- `0.50x`: `{0.5 * threshold:.4f}` px",
        f"- `0.75x`: `{0.75 * threshold:.4f}` px",
        f"- `1.00x`: `{threshold:.4f}` px",
        f"- `1.25x`: `{1.25 * threshold:.4f}` px",
        f"- `1.50x`: `{1.5 * threshold:.4f}` px",
        "",
        "## Split Summary",
        "",
        "| Split | Scenes | GT boxes | Mean width | Mean height | Mean sqrt(area) | Mean TS-AMID | P50 TS-AMID | Mean ratio | P50 ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_summary:
        lines.append(
            "| {split} | {scenes} | {gt_boxes} | {mean_width:.2f} | {mean_height:.2f} | "
            "{mean_sqrt_area:.2f} | {mean_ts_amid_box:.2f} | {p50_ts_amid_box:.2f} | "
            "{mean_ts_amid_ratio:.3f} | {p50_ts_amid_ratio:.3f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Recommended Three-Level Difficulty Split",
            "",
            "| Difficulty | Criterion | Pixel range | Test scenes | Mean TS-AMID | Mean ratio | Mean targets/frame |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for level in DIFFICULTY_3_ORDER:
        rows = [row for row in test_rows if row["difficulty_3"] == level]
        name, criterion, pixel_range = ranges[level]
        lines.append(
            f"| {name} | `{criterion}` | `{pixel_range}` | {len(rows)} | "
            f"{np.mean([float(row['ts_amid_box']) for row in rows]):.2f} | "
            f"{np.mean([float(row['ts_amid_over_test_mean_sqrt_area']) for row in rows]):.3f} | "
            f"{np.mean([float(row['mean_targets_per_frame']) for row in rows]):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `test_scene_ts_amid_box_difficulty.csv`: test split scene difficulty table.",
            "- `scene_ts_amid_box_difficulty_all_splits.csv`: scene difficulty table for train/val/test.",
            "- `ts_amid_box_size_split_summary.csv`: split-level box size and TS-AMID summary.",
            "- `scene_lists/*.txt`: ready-to-use test scene lists by difficulty.",
            "- `figures/*.png`: visual summaries, including TS-AMID distributions and GT box-size distributions.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze box-based TS-AMID difficulty groups.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    ds = load_dataset_config(args.dataset)
    out_dir = args.output_dir or (ds.data_root / "analysis" / "ts_amid_difficulty")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes_by_split = {split: read_split(ds.split_root, split) for split in SPLITS}
    test_box_values = collect_box_values(ds.gt_root, scenes_by_split["test"])
    test_mean_sqrt_area = qstats(test_box_values["sqrt_area"])["mean"]

    all_rows = []
    for split in SPLITS:
        for scene in scenes_by_split[split]:
            row = scene_stats(ds.gt_root, split, scene)
            ratio = float(row["ts_amid_box"]) / test_mean_sqrt_area
            row["ts_amid_over_test_mean_sqrt_area"] = ratio
            row["difficulty_3"] = difficulty_3(ratio)
            row["difficulty_6"] = difficulty_6(ratio)
            all_rows.append(row)

    scene_fields = [
        "split",
        "scene",
        "difficulty_3",
        "difficulty_6",
        "ts_amid_box",
        "ts_amid_over_test_mean_sqrt_area",
        "frames",
        "multi_target_frames",
        "gt_boxes",
        "mean_targets_per_frame",
        "mean_width",
        "mean_height",
        "mean_avg_side",
        "mean_sqrt_area",
        "mean_diagonal",
        "mean_area",
        "frame_min_box_dist_mean",
        "zero_min_dist_frame_rate",
    ]
    test_rows = sorted([row for row in all_rows if row["split"] == "test"], key=lambda row: float(row["ts_amid_box"]))
    write_csv(out_dir / "test_scene_ts_amid_box_difficulty.csv", test_rows, scene_fields)
    write_csv(
        out_dir / "scene_ts_amid_box_difficulty_all_splits.csv",
        sorted(all_rows, key=lambda row: (SPLITS.index(str(row["split"])), str(row["scene"]))),
        scene_fields,
    )

    split_summary = []
    for split in (*SPLITS, "all"):
        split_rows = all_rows if split == "all" else [row for row in all_rows if row["split"] == split]
        scenes = [str(row["scene"]) for row in split_rows]
        box_values = collect_box_values(ds.gt_root, scenes)
        width = qstats(box_values["width"])
        height = qstats(box_values["height"])
        sqrt_area = qstats(box_values["sqrt_area"])
        area = qstats(box_values["area"])
        ts = qstats([float(row["ts_amid_box"]) for row in split_rows])
        ratio = qstats([float(row["ts_amid_over_test_mean_sqrt_area"]) for row in split_rows])
        split_summary.append(
            {
                "split": split,
                "scenes": len(split_rows),
                "gt_boxes": int(width["n"]),
                "mean_width": width["mean"],
                "p50_width": width["p50"],
                "mean_height": height["mean"],
                "p50_height": height["p50"],
                "mean_sqrt_area": sqrt_area["mean"],
                "p50_sqrt_area": sqrt_area["p50"],
                "mean_area": area["mean"],
                "p50_area": area["p50"],
                "mean_ts_amid_box": ts["mean"],
                "p25_ts_amid_box": ts["p25"],
                "p50_ts_amid_box": ts["p50"],
                "p75_ts_amid_box": ts["p75"],
                "p90_ts_amid_box": ts["p90"],
                "mean_ts_amid_ratio": ratio["mean"],
                "p25_ts_amid_ratio": ratio["p25"],
                "p50_ts_amid_ratio": ratio["p50"],
                "p75_ts_amid_ratio": ratio["p75"],
                "p90_ts_amid_ratio": ratio["p90"],
            }
        )
    write_csv(out_dir / "ts_amid_box_size_split_summary.csv", split_summary, list(split_summary[0]))
    write_scene_lists(out_dir, test_rows, "difficulty_3", DIFFICULTY_3_ORDER)
    write_scene_lists(out_dir, test_rows, "difficulty_6", DIFFICULTY_6_ORDER)
    plot_histograms(out_dir, ds.name, test_rows, split_summary)
    plot_box_size_distributions(out_dir, ds.name, ds.gt_root, scenes_by_split)
    make_summary(out_dir, ds.name, test_mean_sqrt_area, split_summary, test_rows)
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
