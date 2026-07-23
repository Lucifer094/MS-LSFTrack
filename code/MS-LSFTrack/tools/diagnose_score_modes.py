from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.datasets import MultiScaleRealCandidateGroupDataset, RealCandidateGroupDataset
from mslsftrack.models import (
    CandidateGroupStructureNet,
    MultiScaleCandidateGroupStructureNet,
    fuse_motion_structure,
    fuse_motion_structure_with_mode,
)


SCORE_MODES = ("full", "residual", "competitive", "geometry", "structure", "distance", "field_l1")


def parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def top1(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return scores.argmax(dim=1) == target


def masked_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(~mask, -1e4)


def make_dataset(cache_path: Path, config: dict):
    if "num_scales" in config:
        return MultiScaleRealCandidateGroupDataset(
            cache_path,
            radii=tuple(float(v) for v in config.get("radii", (10.0, 15.0, 20.0))),
            neighbor_mode=str(config.get("neighborhood", "radius")),
            knn=tuple(int(v) for v in config["ks"]) if "ks" in config else None,
        )
    return RealCandidateGroupDataset(
        cache_path,
        radius=float(config.get("radius", 20.0)),
        neighbor_mode=str(config.get("neighborhood", "radius")),
    )


def make_model(config: dict, device: str):
    if "num_scales" in config:
        return MultiScaleCandidateGroupStructureNet(
            grid_size=int(config.get("grid_size", 24)),
            width=int(config.get("width", 24)),
            num_scales=int(config["num_scales"]),
            fusion_mode=str(config.get("fusion_mode", "residual")),
        ).to(device)
    return CandidateGroupStructureNet(
        grid_size=int(config.get("grid_size", 24)),
        width=int(config.get("width", 24)),
        fusion_mode=str(config.get("fusion_mode", "residual")),
    ).to(device)


def score_tensors(out: dict[str, torch.Tensor], geometry: torch.Tensor, mask: torch.Tensor, model) -> dict[str, torch.Tensor]:
    geometry_logits = masked_scores(out["geometry_logits"], mask)
    structure_logits = masked_scores(out["structure_logits"], mask)
    residual_logits = masked_scores(
        fuse_motion_structure(geometry, structure_logits, F.softplus(model.fusion_scale), mask)["logits"],
        mask,
    )
    if hasattr(model, "reliability_gate"):
        competitive_logits = masked_scores(
            fuse_motion_structure_with_mode(
                geometry,
                structure_logits,
                F.softplus(model.fusion_scale),
                mask,
                fusion_mode="competitive",
                reliability_gate=model.reliability_gate,
            )["logits"],
            mask,
        )
    else:
        competitive_logits = residual_logits
    full_logits = competitive_logits if getattr(model, "fusion_mode", "residual") == "competitive" else residual_logits
    distance_scores = -geometry[..., 2].masked_fill(~mask, 1e4)

    diff = torch.abs(out["candidate_field"] - out["history_field"][:, None])
    if diff.ndim == 6:
        scale_l1 = diff.mean(dim=(3, 4, 5))
        if "scale_weights" in out:
            field_l1 = -(scale_l1 * out["scale_weights"]).sum(dim=-1)
        else:
            field_l1 = -scale_l1.mean(dim=-1)
    elif diff.ndim == 5:
        field_l1 = -diff.mean(dim=(2, 3, 4))
    else:
        raise ValueError(f"unexpected field diff shape: {tuple(diff.shape)}")
    field_l1 = masked_scores(field_l1, mask)

    return {
        "full": full_logits,
        "residual": residual_logits,
        "competitive": competitive_logits,
        "geometry": geometry_logits,
        "structure": structure_logits,
        "distance": distance_scores,
        "field_l1": field_l1,
    }


@torch.no_grad()
def diagnose(model, loader, device: str, thresholds: list[float]) -> tuple[list[dict], list[dict]]:
    model.eval()
    totals = {
        mode: {
            "count": 0,
            "correct": 0,
            "hard_count": 0,
            "hard_correct": 0,
            "valid_scores": [],
            "positive_scores": [],
            "negative_scores": [],
            "margins": [],
            "ranks": [],
        }
        for mode in SCORE_MODES
    }
    scale_weight_sum = None
    scale_weight_count = 0

    for batch in loader:
        moved = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key != "query_index"
        }
        out = model(
            moved["history_points"],
            moved["history_mask"],
            moved["candidate_points"],
            moved["candidate_neighbor_mask"],
            moved["geometry"],
            moved["candidate_mask"],
        )
        mask = moved["candidate_mask"]
        target = moved["target"]
        scores_by_mode = score_tensors(out, moved["geometry"], mask, model)
        sorted_distance = moved["geometry"][..., 2].masked_fill(~mask, 1e4).sort(dim=1).values
        hard_mask = (sorted_distance[:, 1] - sorted_distance[:, 0]) <= 0.2

        if "scale_weights" in out:
            valid_weights = out["scale_weights"][mask]
            if valid_weights.numel():
                summed = valid_weights.sum(dim=0).detach()
                scale_weight_sum = summed if scale_weight_sum is None else scale_weight_sum + summed
                scale_weight_count += int(valid_weights.shape[0])

        for mode, scores in scores_by_mode.items():
            valid_scores = scores[mask].detach().cpu().numpy().astype(np.float64)
            positive_scores = scores.gather(1, target[:, None]).squeeze(1)
            negatives = scores.clone()
            negatives.scatter_(1, target[:, None], -1e4)
            best_negative = negatives.max(dim=1).values
            margin = positive_scores - best_negative
            prediction = scores.argmax(dim=1)
            correct = prediction == target
            rank = 1 + (scores > positive_scores[:, None]).sum(dim=1)

            mode_totals = totals[mode]
            mode_totals["count"] += int(target.numel())
            mode_totals["correct"] += int(correct.sum())
            mode_totals["hard_count"] += int(hard_mask.sum())
            mode_totals["hard_correct"] += int((correct & hard_mask).sum())
            mode_totals["valid_scores"].append(valid_scores)
            mode_totals["positive_scores"].append(positive_scores.detach().cpu().numpy().astype(np.float64))
            mode_totals["negative_scores"].append(negatives[negatives > -1e3].detach().cpu().numpy().astype(np.float64))
            mode_totals["margins"].append(margin.detach().cpu().numpy().astype(np.float64))
            mode_totals["ranks"].append(rank.detach().cpu().numpy().astype(np.float64))

    summary_rows: list[dict] = []
    threshold_rows: list[dict] = []
    avg_scale_weights = (
        (scale_weight_sum / max(scale_weight_count, 1)).detach().cpu().tolist()
        if scale_weight_sum is not None
        else []
    )
    for mode in SCORE_MODES:
        mode_totals = totals[mode]
        valid = np.concatenate(mode_totals["valid_scores"]) if mode_totals["valid_scores"] else np.asarray([])
        positive = np.concatenate(mode_totals["positive_scores"]) if mode_totals["positive_scores"] else np.asarray([])
        negative = np.concatenate(mode_totals["negative_scores"]) if mode_totals["negative_scores"] else np.asarray([])
        margins = np.concatenate(mode_totals["margins"]) if mode_totals["margins"] else np.asarray([])
        ranks = np.concatenate(mode_totals["ranks"]) if mode_totals["ranks"] else np.asarray([])
        valid_stats = quantiles(valid)
        positive_stats = quantiles(positive)
        negative_stats = quantiles(negative)
        margin_stats = quantiles(margins)
        summary_rows.append(
            {
                "score_mode": mode,
                "num_queries": mode_totals["count"],
                "num_valid_scores": int(valid.size),
                "top1": mode_totals["correct"] / max(mode_totals["count"], 1),
                "hard_num_queries": mode_totals["hard_count"],
                "hard_top1": mode_totals["hard_correct"] / max(mode_totals["hard_count"], 1),
                "mean_positive_rank": float(np.mean(ranks)) if ranks.size else float("nan"),
                "valid_score_mean": valid_stats["mean"],
                "valid_score_std": valid_stats["std"],
                "valid_score_min": valid_stats["min"],
                "valid_score_p01": valid_stats["p01"],
                "valid_score_p05": valid_stats["p05"],
                "valid_score_p50": valid_stats["p50"],
                "valid_score_p95": valid_stats["p95"],
                "valid_score_p99": valid_stats["p99"],
                "valid_score_max": valid_stats["max"],
                "positive_score_mean": positive_stats["mean"],
                "negative_score_mean": negative_stats["mean"],
                "margin_mean": margin_stats["mean"],
                "margin_p05": margin_stats["p05"],
                "margin_p50": margin_stats["p50"],
                "margin_p95": margin_stats["p95"],
                "avg_scale_weights": json.dumps(avg_scale_weights),
            }
        )
        for threshold in thresholds:
            threshold_rows.append(
                {
                    "score_mode": mode,
                    "threshold": threshold,
                    "valid_pass_rate": float(np.mean(valid >= threshold)) if valid.size else float("nan"),
                    "positive_pass_rate": float(np.mean(positive >= threshold)) if positive.size else float("nan"),
                    "negative_pass_rate": float(np.mean(negative >= threshold)) if negative.size else float("nan"),
                }
            )
    return summary_rows, threshold_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose raw candidate scores for score-mode ablations.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--thresholds", default="-17,-10,-5,-2,-1,0,1,2,5,10")
    args = parser.parse_args()

    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    dataset = make_dataset(args.cache, config)
    if args.max_queries:
        dataset.cache.queries = dataset.cache.queries[: args.max_queries]
    model = make_model(config, device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    thresholds = parse_floats(args.thresholds)
    summary_rows, threshold_rows = diagnose(model, loader, device, thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "score_mode_diagnostics.csv", summary_rows)
    write_csv(args.output_dir / "score_mode_thresholds.csv", threshold_rows)
    (args.output_dir / "score_mode_diagnostics.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "cache": str(args.cache),
                "device": device,
                "num_queries": len(dataset),
                "summary": summary_rows,
                "thresholds": threshold_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"[DONE] {args.output_dir}")


if __name__ == "__main__":
    main()
