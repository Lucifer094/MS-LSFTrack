from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneInfo:
    scene: str
    category: str
    ts_amid: float
    difficulty: str
    frames: int
    mean_targets: float


def read_scene_tsamid(path: Path) -> list[SceneInfo]:
    rows: list[SceneInfo] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                SceneInfo(
                    scene=row["sequence"],
                    category=row.get("category", ""),
                    ts_amid=float(row["ts_amid"]),
                    difficulty=row.get("difficulty", ""),
                    frames=int(float(row.get("frames", 0))),
                    mean_targets=float(row.get("mean_targets", 0.0)),
                )
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def split_score(groups: dict[str, list[SceneInfo]], target_mean: float) -> float:
    score = 0.0
    for name, items in groups.items():
        if not items:
            score += 1e6
            continue
        mean = sum(x.ts_amid for x in items) / len(items)
        score += abs(mean - target_mean)
    return score


def make_split(
    scenes: list[SceneInfo],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    restarts: int,
) -> dict[str, list[SceneInfo]]:
    if train_count + val_count + test_count != len(scenes):
        raise ValueError(
            f"split counts {train_count}+{val_count}+{test_count} != {len(scenes)}"
        )
    rng = random.Random(seed)
    target_mean = sum(x.ts_amid for x in scenes) / len(scenes)
    best_groups: dict[str, list[SceneInfo]] | None = None
    best_score = float("inf")

    # Greedy assignment over descending difficulty with random tie jitter.
    for restart in range(restarts):
        items = list(scenes)
        rng.shuffle(items)
        items.sort(key=lambda x: x.ts_amid, reverse=True)
        groups = {"train": [], "val": [], "test": []}
        limits = {"train": train_count, "val": val_count, "test": test_count}
        sums = {"train": 0.0, "val": 0.0, "test": 0.0}
        for item in items:
            candidates = []
            for name in ["train", "val", "test"]:
                if len(groups[name]) >= limits[name]:
                    continue
                new_mean = (sums[name] + item.ts_amid) / (len(groups[name]) + 1)
                capacity_pressure = len(groups[name]) / max(limits[name], 1)
                candidates.append((abs(new_mean - target_mean), -capacity_pressure, rng.random(), name))
            _, _, _, chosen = min(candidates)
            groups[chosen].append(item)
            sums[chosen] += item.ts_amid

        # Local random swaps to further balance means.
        improved = True
        while improved:
            improved = False
            current = split_score(groups, target_mean)
            names = ["train", "val", "test"]
            for a in names:
                for b in names:
                    if a >= b:
                        continue
                    for ia, va in enumerate(groups[a]):
                        for ib, vb in enumerate(groups[b]):
                            trial = {k: list(v) for k, v in groups.items()}
                            trial[a][ia], trial[b][ib] = trial[b][ib], trial[a][ia]
                            new_score = split_score(trial, target_mean)
                            if new_score + 1e-12 < current:
                                groups = trial
                                current = new_score
                                improved = True
        score = split_score(groups, target_mean)
        if score < best_score:
            best_score = score
            best_groups = groups
    assert best_groups is not None
    for values in best_groups.values():
        values.sort(key=lambda x: x.scene)
    return best_groups


def summarize(groups: dict[str, list[SceneInfo]]) -> list[dict]:
    rows = []
    for split, items in groups.items():
        ts = [x.ts_amid for x in items]
        rows.append(
            {
                "split": split,
                "num_scenes": len(items),
                "mean_tsamid": sum(ts) / len(ts) if ts else 0.0,
                "min_tsamid": min(ts) if ts else 0.0,
                "max_tsamid": max(ts) if ts else 0.0,
                "mean_targets": sum(x.mean_targets for x in items) / len(items) if items else 0.0,
                "scenes": " ".join(x.scene for x in items),
            }
        )
    return rows


def write_split_files(output: Path, groups: dict[str, list[SceneInfo]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for split, items in groups.items():
        (output / f"{split}.txt").write_text(
            "\n".join(x.scene for x in items) + "\n",
            encoding="utf-8",
        )
    all_scenes = sorted(x.scene for items in groups.values() for x in items)
    (output / "all.txt").write_text("\n".join(all_scenes) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data_root = Path(os.environ.get("MS_LSF_DATA_ROOT", "./data"))
    parser = argparse.ArgumentParser(
        description="Create GMOT-40-small-target train/val/test splits balanced by scene TS-AMID."
    )
    parser.add_argument(
        "--scene-tsamid",
        type=Path,
        default=data_root / "GMOT-40-small-target" / "splits" / "scene_difficulty_tsamid.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_root / "GMOT-40-small-target" / "splits",
    )
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=200)
    args = parser.parse_args()

    scenes = read_scene_tsamid(args.scene_tsamid)
    groups = make_split(
        scenes,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
        restarts=args.restarts,
    )
    write_split_files(args.output, groups)

    assignment_rows = []
    for split, items in groups.items():
        for item in items:
            assignment_rows.append(
                {
                    "split": split,
                    "scene": item.scene,
                    "category": item.category,
                    "ts_amid": item.ts_amid,
                    "difficulty": item.difficulty,
                    "frames": item.frames,
                    "mean_targets": item.mean_targets,
                }
            )
    assignment_rows.sort(key=lambda r: (r["split"], r["scene"]))
    summary_rows = summarize(groups)
    write_csv(args.output / "split_assignment.csv", assignment_rows)
    write_csv(args.output / "split_summary.csv", summary_rows)
    metadata = {
        "source": str(args.scene_tsamid),
        "output": str(args.output),
        "seed": args.seed,
        "restarts": args.restarts,
        "counts": {
            "train": args.train_count,
            "val": args.val_count,
            "test": args.test_count,
        },
        "summary": summary_rows,
    }
    (args.output / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
