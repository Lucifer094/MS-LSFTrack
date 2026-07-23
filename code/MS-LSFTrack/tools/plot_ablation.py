from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bar(labels: list[str], values: list[float], title: str, ylabel: str, output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(labels)), 4.2))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, max(values) * 1.08 if values else 1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clear ablation figures from CSV files.")
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--tracking-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.candidate_csv:
        rows = read_csv(args.candidate_csv)
        if rows:
            labels = [r.get("gate_mode", r.get("variant", f"v{i}")) for i, r in enumerate(rows)]
            bar(
                labels,
                [float(r.get("hard_full_top1", 0)) for r in rows],
                "Candidate-level hard Top-1 ablation",
                "Hard Top-1",
                args.output_dir / "candidate_hard_top1.png",
            )
            bar(
                labels,
                [float(r.get("full_top1", 0)) for r in rows],
                "Candidate-level Top-1 ablation",
                "Top-1",
                args.output_dir / "candidate_top1.png",
            )
    if args.tracking_csv:
        rows = read_csv(args.tracking_csv)
        if rows:
            labels = [r.get("tracker", f"t{i}") for i, r in enumerate(rows)]
            bar(
                labels,
                [float(r.get("Pt-HOTA", 0)) for r in rows],
                "Tracking-level HOTA ablation",
                "Pt-HOTA",
                args.output_dir / "tracking_hota.png",
            )
            bar(
                labels,
                [float(r.get("Pt-IDF1", 0)) for r in rows],
                "Tracking-level IDF1 ablation",
                "Pt-IDF1",
                args.output_dir / "tracking_idf1.png",
            )
    print(f"[DONE] figures={args.output_dir}")


if __name__ == "__main__":
    main()

