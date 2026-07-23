from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact Chinese ablation summary.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--tracking-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_rows = read_csv(args.candidate_csv)
    tracking_rows = read_csv(args.tracking_csv) if args.tracking_csv else []
    lines = [
        "# 消融实验结果说明",
        "",
        "本文件由 `tools/make_ablation_readme.py` 生成，用于把候选级和跟踪级消融结果放在同一个说明文档中。",
        "",
        "## 候选级关联消融",
        "",
        "| 设置 | Top-1 | Hard Top-1 | Structure Top-1 | Hard Structure Top-1 | 说明 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in candidate_rows:
        mode = row.get("gate_mode", row.get("variant", ""))
        lines.append(
            f"| {mode} | {float(row.get('full_top1', 0)):.6f} | "
            f"{float(row.get('hard_full_top1', 0)):.6f} | "
            f"{float(row.get('structure_top1', 0)):.6f} | "
            f"{float(row.get('hard_structure_top1', 0)):.6f} | 候选组 Top-1，Hard 表示中心距离差很小的困难样本 |"
        )
    if tracking_rows:
        lines.extend([
            "",
            "## 跟踪级消融",
            "",
            "| Tracker | Pt-HOTA | Pt-IDF1 | Pt-AssA | Pt-MOTA | IDs | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in tracking_rows:
            lines.append(
                f"| {row.get('tracker', '')} | {float(row.get('Pt-HOTA', 0)):.6f} | "
                f"{float(row.get('Pt-IDF1', 0)):.6f} | {float(row.get('Pt-AssA', 0)):.6f} | "
                f"{float(row.get('Pt-MOTA', 0)):.6f} | {float(row.get('Pt-IDs', 0)):.0f} | "
                f"{float(row.get('Pt-FP', 0)):.0f} | {float(row.get('Pt-FN', 0)):.0f} |"
            )
    lines.extend([
        "",
        "## 推荐论文解读",
        "",
        "- `geometry/distance` 用来证明仅靠位置或几何头不足以解决密集小目标的困难关联。",
        "- `structure/full` 用来证明局部结构场对候选消歧有贡献。",
        "- `learned/uniform/small/middle/large` 用来证明多尺度门控不是简单平均，而是有自适应融合收益。",
        "- `radius/knn` 是邻域构建方式消融。若 KNN 更好，不应声称 radius all-neighbor 性能更优；应把创新收束到可学习局部结构场和多尺度结构融合。",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()

