from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_ROOT = ROOT / "third_party"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from mslsftrack.config import load_dataset_config
from unified_eval_suite.io import ensure_dir, read_csv_dicts, write_csv_dicts


DATASETS = [
    ("configs/datasets/irdmstrack_v3_det_label.yaml", "MS-LSFTrack"),
    ("configs/datasets/gmot40_small_det_label.yaml", "MS-LSFTrack"),
    ("configs/datasets/irsatvideo_leo_resunet_rfr.yaml", "MS-LSFTrack"),
]

METHOD_NAMES = {
    "bytetrack": "ByteTrack",
    "ocsort": "OC-SORT",
    "sfsort": "SF-SORT",
    "strongsort": "StrongSORT",
    "botsort": "BoT-SORT",
    "deepocsort": "DeepOCSORT",
    "hybridsort": "HybridSORT",
    "boosttrack": "BoostTrack",
    "MS-LSFTrack": "MS-LSFTrack",
    "putr": "PuTR",
}

MACRO_METRICS = [
    "Pt-HOTA",
    "Pt-MOTA",
    "Pt-IDF1",
    "Pt-AssA",
    "Pt-AssR",
    "Pt-IDs",
    "Pt-Frag",
    "IoU-HOTA",
    "IoU-MOTA",
    "IoU-IDF1",
    "IoU-AssA",
    "IoU-IDs",
    "IoU-Frag",
    "fps_update",
]

DATASET_METRICS = ["Pt-HOTA", "Pt-IDF1", "IoU-HOTA"]
RANKING_METRICS = ["Pt-HOTA", "Pt-MOTA", "Pt-IDF1", "Pt-AssA", "IoU-HOTA", "IoU-IDF1", "fps_update"]


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt4(value: Any) -> str:
    number = as_float(value)
    return "" if number is None else f"{number:.4f}"


def fmt2(value: Any) -> str:
    number = as_float(value)
    return "" if number is None else f"{number:.2f}"


def load_rows(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_config, ms_tracker in DATASETS:
        ds = load_dataset_config(ROOT / dataset_config)
        table_path = ds.unified_root / "comparison" / "paper_tables" / f"{ds.det_version}_{split}_paper_table.csv"
        for source_row in read_csv_dicts(table_path):
            tracker = source_row.get("tracker", "")
            method = METHOD_NAMES.get(tracker, tracker)
            row = dict(source_row)
            row["method"] = method
            row["source_table"] = str(table_path)
            row["tracker_group"] = "MS-LSFTrack" if tracker == ms_tracker else ("comparison" if tracker == "putr" else "baseline")
            rows.append(row)
    return rows


def build_macro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method", ""))].append(row)

    macro_rows: list[dict[str, Any]] = []
    for method, method_rows in by_method.items():
        out: dict[str, Any] = {"method": method, "num_datasets": len({row.get("dataset", "") for row in method_rows})}
        for metric in MACRO_METRICS:
            values = [as_float(row.get(metric)) for row in method_rows]
            values = [value for value in values if value is not None]
            out[f"macro_{metric}"] = sum(values) / len(values) if values else ""
        for row in sorted(method_rows, key=lambda item: str(item.get("dataset", ""))):
            dataset = str(row.get("dataset", ""))
            for metric in DATASET_METRICS:
                out[f"{dataset}_{metric}"] = row.get(metric, "")
        macro_rows.append(out)

    macro_rows.sort(key=lambda row: as_float(row.get("macro_Pt-HOTA")) or -1.0, reverse=True)
    return macro_rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    aligns = ["---:" if header in {"Rank"} or header.endswith(")") or header.startswith("Pt-") or header.startswith("IoU-") else "---" for header in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(rows: list[dict[str, Any]], macro_rows: list[dict[str, Any]], split: str) -> str:
    lines = [
        "# All Datasets Test Comparison",
        "",
        f"Generated from formal `{split}` point/IoU summaries.",
        "Temporary `smoke`/`cpu_probe` outputs are excluded from this table.",
        "",
        "## Macro Average Ranking",
        "",
    ]
    macro_table_rows: list[list[str]] = []
    for rank, row in enumerate(macro_rows, start=1):
        macro_table_rows.append(
            [
                str(rank),
                str(row.get("method", "")),
                str(row.get("num_datasets", "")),
                *[fmt4(row.get(f"macro_{metric}")) for metric in RANKING_METRICS[:-1]],
                fmt2(row.get("macro_fps_update")),
            ]
        )
    lines.append(markdown_table(["Rank", "Method", "Datasets", "Pt-HOTA", "Pt-MOTA", "Pt-IDF1", "Pt-AssA", "IoU-HOTA", "IoU-IDF1", "FPS(update)"], macro_table_rows))
    lines.extend(["", "## Per-Dataset Pt-HOTA / Pt-IDF1", ""])

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset", ""))].append(row)
    for dataset in sorted(by_dataset):
        dataset_rows = sorted(by_dataset[dataset], key=lambda row: as_float(row.get("Pt-HOTA")) or -1.0, reverse=True)
        lines.extend([f"### {dataset}", ""])
        table_rows: list[list[str]] = []
        for rank, row in enumerate(dataset_rows, start=1):
            table_rows.append(
                [
                    str(rank),
                    str(row.get("method", "")),
                    fmt4(row.get("Pt-HOTA")),
                    fmt4(row.get("Pt-MOTA")),
                    fmt4(row.get("Pt-IDF1")),
                    fmt4(row.get("Pt-AssA")),
                    fmt4(row.get("IoU-HOTA")),
                    fmt4(row.get("IoU-IDF1")),
                    fmt2(row.get("fps_update")),
                ]
            )
        lines.append(markdown_table(["Rank", "Method", "Pt-HOTA", "Pt-MOTA", "Pt-IDF1", "Pt-AssA", "IoU-HOTA", "IoU-IDF1", "FPS(update)"], table_rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the all-datasets comparison tables from formal paper tables.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("MS_LSF_SUMMARY_ROOT", "./outputs/_summary")))
    args = parser.parse_args()

    rows = load_rows(args.split)
    out_dir = ensure_dir(args.output_dir)
    paper_path = out_dir / f"all_datasets_{args.split}_paper_table.csv"
    macro_path = out_dir / f"all_datasets_{args.split}_macro_average.csv"
    markdown_path = out_dir / f"all_datasets_{args.split}_comparison.md"

    paper_fields = [
        "dataset",
        "method",
        "Pt-HOTA",
        "Pt-IDF1",
        "Pt-MOTA",
        "IoU-HOTA",
        "IoU-IDF1",
        "IoU-MOTA",
        "fps_update",
    ]
    write_csv_dicts(paper_path, rows, paper_fields)

    macro_rows = build_macro(rows)
    macro_fields = [key for row in macro_rows for key in row.keys()]
    macro_fields = list(dict.fromkeys(macro_fields))
    write_csv_dicts(macro_path, macro_rows, macro_fields)

    markdown_path.write_text(build_markdown(rows, macro_rows, args.split), encoding="utf-8")
    print(f"[DONE] paper={paper_path}")
    print(f"[DONE] macro={macro_path}")
    print(f"[DONE] markdown={markdown_path}")


if __name__ == "__main__":
    main()
