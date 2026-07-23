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


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_ints(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def top1(scores: torch.Tensor, target: torch.Tensor) -> int:
    return int((scores.argmax(dim=1) == target).sum())


@torch.no_grad()
def evaluate_loader(model, loader, device: str, gate_mode: str) -> dict:
    model.eval()
    totals = {
        "count": 0,
        "full": 0,
        "residual": 0,
        "competitive": 0,
        "structure": 0,
        "geometry": 0,
        "distance": 0,
    }
    hard = dict.fromkeys(totals.keys(), 0)
    margins = []
    weight_sum = None
    weight_count = 0
    for batch in loader:
        moved = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "query_index"}
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
        geometry_logits = out["geometry_logits"].masked_fill(~mask, -1e4)
        if "scale_logits" in out:
            if gate_mode == "learned":
                weights = out["scale_weights"]
            elif gate_mode == "uniform":
                weights = torch.full_like(out["scale_weights"], 1.0 / out["scale_weights"].shape[-1])
            elif gate_mode == "small":
                weights = torch.zeros_like(out["scale_weights"])
                weights[..., 0] = 1.0
            elif gate_mode == "middle":
                weights = torch.zeros_like(out["scale_weights"])
                weights[..., min(1, weights.shape[-1] - 1)] = 1.0
            elif gate_mode == "large":
                weights = torch.zeros_like(out["scale_weights"])
                weights[..., -1] = 1.0
            else:
                raise ValueError(f"unknown gate_mode: {gate_mode}")
            structure_logits = (out["scale_logits"] * weights).sum(dim=-1).masked_fill(~mask, -1e4)
            valid_weights = weights[mask]
            if valid_weights.numel():
                summed = valid_weights.sum(dim=0).detach()
                weight_sum = summed if weight_sum is None else weight_sum + summed
                weight_count += int(valid_weights.shape[0])
        else:
            structure_logits = out["structure_logits"].masked_fill(~mask, -1e4)
        residual_logits = fuse_motion_structure(
            moved["geometry"],
            structure_logits,
            F.softplus(model.fusion_scale),
            mask,
        )["logits"].masked_fill(~mask, -1e4)
        competitive_logits = None
        if hasattr(model, "reliability_gate"):
            competitive_logits = fuse_motion_structure_with_mode(
                moved["geometry"],
                structure_logits,
                F.softplus(model.fusion_scale),
                mask,
                fusion_mode="competitive",
                reliability_gate=model.reliability_gate,
            )["logits"].masked_fill(~mask, -1e4)
        full_logits = competitive_logits if competitive_logits is not None else residual_logits
        distance_scores = -moved["geometry"][..., 2].masked_fill(~mask, 1e4)
        sorted_distance = moved["geometry"][..., 2].masked_fill(~mask, 1e4).sort(dim=1).values
        hard_mask = (sorted_distance[:, 1] - sorted_distance[:, 0]) <= 0.2

        totals["count"] += len(target)
        totals["full"] += top1(full_logits, target)
        totals["residual"] += top1(residual_logits, target)
        if competitive_logits is not None:
            totals["competitive"] += top1(competitive_logits, target)
        totals["structure"] += top1(structure_logits, target)
        totals["geometry"] += top1(geometry_logits, target)
        totals["distance"] += top1(distance_scores, target)
        hard["count"] += int(hard_mask.sum())
        hard["full"] += int(((full_logits.argmax(1) == target) & hard_mask).sum())
        hard["residual"] += int(((residual_logits.argmax(1) == target) & hard_mask).sum())
        if competitive_logits is not None:
            hard["competitive"] += int(((competitive_logits.argmax(1) == target) & hard_mask).sum())
        hard["structure"] += int(((structure_logits.argmax(1) == target) & hard_mask).sum())
        hard["geometry"] += int(((geometry_logits.argmax(1) == target) & hard_mask).sum())
        hard["distance"] += int(((distance_scores.argmax(1) == target) & hard_mask).sum())
        pos = full_logits.gather(1, target[:, None]).squeeze(1)
        neg = full_logits.clone()
        neg.scatter_(1, target[:, None], -1e4)
        margins.extend((pos - neg.max(dim=1).values).cpu().tolist())

    count = max(totals["count"], 1)
    hard_count = max(hard["count"], 1)
    return {
        "gate_mode": gate_mode,
        "num_queries": totals["count"],
        "full_top1": totals["full"] / count,
        "residual_top1": totals["residual"] / count,
        "competitive_top1": totals["competitive"] / count if totals["competitive"] else 0.0,
        "structure_top1": totals["structure"] / count,
        "geometry_head_top1": totals["geometry"] / count,
        "nearest_distance_top1": totals["distance"] / count,
        "hard_num_queries": hard["count"],
        "hard_full_top1": hard["full"] / hard_count,
        "hard_residual_top1": hard["residual"] / hard_count,
        "hard_competitive_top1": hard["competitive"] / hard_count if totals["competitive"] else 0.0,
        "hard_structure_top1": hard["structure"] / hard_count,
        "hard_geometry_head_top1": hard["geometry"] / hard_count,
        "hard_nearest_distance_top1": hard["distance"] / hard_count,
        "mean_margin": float(np.mean(margins)) if margins else float("nan"),
        "avg_scale_weights": (weight_sum / max(weight_count, 1)).cpu().tolist() if weight_sum is not None else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate-level ablation for local structure field.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-test-queries", type=int)
    parser.add_argument("--gate-modes", default="learned,uniform,small,middle,large")
    args = parser.parse_args()

    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    if "num_scales" in config:
        radii = tuple(float(v) for v in config.get("radii", (10.0, 15.0, 20.0)))
        ds = MultiScaleRealCandidateGroupDataset(
            args.test_cache,
            radii=radii,
            neighbor_mode=str(config.get("neighborhood", "radius")),
            knn=tuple(int(v) for v in config["ks"]) if "ks" in config else None,
        )
        model = MultiScaleCandidateGroupStructureNet(
            grid_size=24,
            width=24,
            num_scales=len(radii),
            fusion_mode=str(config.get("fusion_mode", "residual")),
        ).to(device)
        gate_modes = [mode.strip() for mode in args.gate_modes.split(",") if mode.strip()]
    else:
        ds = RealCandidateGroupDataset(
            args.test_cache,
            radius=float(config.get("radius", 20.0)),
            neighbor_mode=str(config.get("neighborhood", "radius")),
        )
        model = CandidateGroupStructureNet(
            grid_size=24,
            width=24,
            fusion_mode=str(config.get("fusion_mode", "residual")),
        ).to(device)
        gate_modes = ["single"]
    if args.max_test_queries:
        ds.cache.queries = ds.cache.queries[: args.max_test_queries]
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    rows = [evaluate_loader(model, loader, device, mode) for mode in gate_modes]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidate_ablation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "candidate_ablation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
