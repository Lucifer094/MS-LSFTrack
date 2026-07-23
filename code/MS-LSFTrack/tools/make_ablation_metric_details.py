from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCHEMAS = {
    "modules": {
        "meta": {
            "motion_prior": (
                "Motion prior only",
                "Yes",
                "No",
                "None",
                "small",
                "ms_lsf_ird/best.pt",
                "No",
                "motion_prior_logits only",
            ),
            "structure": (
                "Structure only",
                "No",
                "Yes",
                "None",
                "small",
                "ms_lsf_ird/best.pt",
                "No",
                "normalized small-scale structure logits only",
            ),
            "fixed_m0p5": (
                "Fixed fusion 0.50/0.50",
                "Yes",
                "Yes",
                "fixed weighted sum",
                "small",
                "ms_lsf_ird/best.pt",
                "No",
                "motion=0.50, structure=0.50",
            ),
            "residual_full": (
                "Full residual model",
                "Yes",
                "Yes",
                "Residual gate",
                "small",
                "ms_lsf_ird/best.pt",
                "No",
                "motion prior plus ambiguity-gated structure residual",
            ),
        },
        "columns": [
            "Variant",
            "Motion prior",
            "Structure branch",
            "Fusion",
            "Scale",
            "Checkpoint",
            "Retrained",
            "Note",
        ],
    },
    "scales": {
        "meta": {
            "scale_small": (
                "Single-scale small",
                "radius 10",
                "Residual gate",
                "ms_lsf_ird/best.pt",
                "No",
                "fixed small local structure",
            ),
            "scale_middle": (
                "Single-scale middle",
                "radius 20",
                "Residual gate",
                "ms_lsf_ird/best.pt",
                "No",
                "fixed middle local structure",
            ),
            "scale_large": (
                "Single-scale large",
                "radius 30",
                "Residual gate",
                "ms_lsf_ird/best.pt",
                "No",
                "fixed large local structure",
            ),
            "scale_uniform": (
                "Uniform multi-scale",
                "1/3, 1/3, 1/3",
                "Residual gate",
                "ms_lsf_ird/best.pt",
                "No",
                "non-learned multi-scale average",
            ),
            "scale_learned": (
                "Learned multi-scale",
                "learned gate",
                "Residual gate",
                "ms_lsf_ird/best.pt",
                "No",
                "current learned scale weights",
            ),
        },
        "columns": ["Variant", "Radius / Weights", "Fusion", "Checkpoint", "Retrained", "Note"],
    },
    "fusion": {
        "meta": {
            "fixed_m0p75": (
                "Fixed 0.75/0.25",
                "fixed weighted sum",
                "ms_lsf_ird/best.pt",
                "No",
                "motion=0.75, structure=0.25",
            ),
            "fixed_m0p5": (
                "Fixed 0.50/0.50",
                "fixed weighted sum",
                "ms_lsf_ird/best.pt",
                "No",
                "motion=0.50, structure=0.50",
            ),
            "fixed_m0p25": (
                "Fixed 0.25/0.75",
                "fixed weighted sum",
                "ms_lsf_ird/best.pt",
                "No",
                "motion=0.25, structure=0.75",
            ),
            "residual": (
                "Residual gate",
                "ambiguity-gated structure residual",
                "ms_lsf_ird/best.pt",
                "No",
                "motion prior plus gated centered structure residual",
            ),
            "competitive": (
                "Competitive reliability gate",
                "dynamic motion/structure weights",
                "ms_lsf_ird_competitive/best.pt",
                "Yes",
                "trained softmax gate over motion and structure experts",
            ),
        },
        "columns": ["Variant", "Fusion", "Checkpoint", "Retrained", "Note"],
    },
}

METRIC_COLUMNS = [
    "GT",
    "Pred",
    "Pt-TP",
    "Pt-FP",
    "Pt-FN",
    "Pt-IDs",
    "Pt-DetA",
    "Pt-AssA",
    "Pt-MOTA",
    "Pt-IDTP",
    "Pt-IDFP",
    "Pt-IDFN",
    "Pt-IDF1",
    "Pt-HOTA",
    "IoU-TP",
    "IoU-FP",
    "IoU-FN",
    "IoU-IDs",
    "IoU-DetA",
    "IoU-AssA",
    "IoU-MOTA",
    "IoU-IDTP",
    "IoU-IDFP",
    "IoU-IDFN",
    "IoU-IDF1",
    "IoU-HOTA",
    "FPS",
]

INT_COLUMNS = {
    "GT",
    "Pred",
    "Pt-TP",
    "Pt-FP",
    "Pt-FN",
    "Pt-IDs",
    "Pt-IDTP",
    "Pt-IDFP",
    "Pt-IDFN",
    "IoU-TP",
    "IoU-FP",
    "IoU-FN",
    "IoU-IDs",
    "IoU-IDTP",
    "IoU-IDFP",
    "IoU-IDFN",
}


def variant_suffix(tracker: str, known_suffixes: list[str]) -> str:
    for suffix in sorted(known_suffixes, key=len, reverse=True):
        if tracker.endswith(f"_{suffix}") or tracker == suffix:
            return suffix
    return tracker.rsplit("_", 1)[-1]


def format_value(row: dict[str, str], key: str) -> str:
    source_key = {"GT": "num_gt_dets", "Pred": "num_pred_dets", "FPS": "fps_update"}.get(key, key)
    value = row.get(source_key, "")
    if value == "":
        return ""
    number = float(value)
    if key in INT_COLUMNS:
        return str(int(round(number)))
    if key == "FPS":
        return f"{number:.2f}"
    return f"{number:.4f}"


def make_rows(
    rows: list[dict[str, str]],
    schema: str,
    checkpoint_label: str,
    competitive_checkpoint_label: str,
) -> tuple[list[str], list[dict[str, str]]]:
    spec = SCHEMAS[schema]
    meta = spec["meta"]
    out_fields = [*spec["columns"], *METRIC_COLUMNS]
    pretty_rows = []
    for row in rows:
        suffix = variant_suffix(row["tracker"], list(meta))
        values = meta.get(suffix, (row["tracker"], suffix, ""))
        pretty = dict(zip(spec["columns"], values))
        if "Checkpoint" in pretty and pretty["Checkpoint"] == "ms_lsf_ird/best.pt":
            pretty["Checkpoint"] = checkpoint_label
        if "Checkpoint" in pretty and pretty["Checkpoint"] == "ms_lsf_ird_competitive/best.pt":
            pretty["Checkpoint"] = competitive_checkpoint_label
        for key in METRIC_COLUMNS:
            pretty[key] = format_value(row, key)
        pretty_rows.append(pretty)
    return out_fields, pretty_rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, fields: list[str], rows: list[dict[str, str]], title: str) -> None:
    text = [f"# {title}", ""]
    text.append("| " + " | ".join(fields) + " |")
    text.append(
        "| "
        + " | ".join("---" if field not in METRIC_COLUMNS else "---:" for field in fields)
        + " |"
    )
    for row in rows:
        text.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create detailed ablation metric tables.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--schema", choices=sorted(SCHEMAS), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", default="Ablation Metric Details")
    parser.add_argument("--checkpoint-label", default="ms_lsf_ird/best.pt")
    parser.add_argument("--competitive-checkpoint-label", default="ms_lsf_ird_competitive/best.pt")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields, pretty_rows = make_rows(
        rows,
        args.schema,
        checkpoint_label=args.checkpoint_label,
        competitive_checkpoint_label=args.competitive_checkpoint_label,
    )
    write_csv(args.output_csv, fields, pretty_rows)
    write_markdown(args.output_md, fields, pretty_rows, args.title)
    print(f"[DONE] csv={args.output_csv}")
    print(f"[DONE] md={args.output_md}")


if __name__ == "__main__":
    main()
