from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mslsftrack.datasets import RealCandidateGroupDataset, MultiScaleRealCandidateGroupDataset
from mslsftrack.models import CandidateGroupStructureNet, MultiScaleCandidateGroupStructureNet


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_ints(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    model.eval()
    totals = {"count": 0, "full": 0, "structure": 0, "geometry": 0, "distance": 0}
    hard = dict.fromkeys(totals.keys(), 0)
    margins: list[float] = []
    rows = []
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
        target = moved["target"]
        full_pred = out["logits"].argmax(dim=1)
        structure_pred = out["structure_logits"].argmax(dim=1)
        geometry_pred = out["geometry_logits"].argmax(dim=1)
        distance_pred = moved["geometry"][..., 2].masked_fill(~moved["candidate_mask"], 1e4).argmin(dim=1)
        sorted_distance = moved["geometry"][..., 2].masked_fill(~moved["candidate_mask"], 1e4).sort(dim=1).values
        hard_mask = (sorted_distance[:, 1] - sorted_distance[:, 0]) <= 0.2

        totals["count"] += len(target)
        totals["full"] += int((full_pred == target).sum())
        totals["structure"] += int((structure_pred == target).sum())
        totals["geometry"] += int((geometry_pred == target).sum())
        totals["distance"] += int((distance_pred == target).sum())
        hard["count"] += int(hard_mask.sum())
        hard["full"] += int(((full_pred == target) & hard_mask).sum())
        hard["structure"] += int(((structure_pred == target) & hard_mask).sum())
        hard["geometry"] += int(((geometry_pred == target) & hard_mask).sum())
        hard["distance"] += int(((distance_pred == target) & hard_mask).sum())

        positive = out["logits"].gather(1, target[:, None]).squeeze(1)
        negatives = out["logits"].clone()
        negatives.scatter_(1, target[:, None], -1e4)
        margin = positive - negatives.max(dim=1).values
        margins.extend(margin.cpu().tolist())
        if "scale_weights" in out:
            valid_weights = out["scale_weights"][moved["candidate_mask"]]
            if valid_weights.numel():
                summed = valid_weights.sum(dim=0).detach()
                scale_weight_sum = summed if scale_weight_sum is None else scale_weight_sum + summed
                scale_weight_count += int(valid_weights.shape[0])
        for q, t, p, m in zip(
            batch["query_index"].tolist(),
            target.cpu().tolist(),
            full_pred.cpu().tolist(),
            margin.cpu().tolist(),
        ):
            rows.append({"query_index": q, "target": t, "prediction": p, "correct": int(t == p), "margin": m})

    count = max(totals["count"], 1)
    hard_count = max(hard["count"], 1)
    return {
        "num_queries": totals["count"],
        "full_top1": totals["full"] / count,
        "structure_top1": totals["structure"] / count,
        "geometry_head_top1": totals["geometry"] / count,
        "nearest_distance_top1": totals["distance"] / count,
        "hard_num_queries": hard["count"],
        "hard_full_top1": hard["full"] / hard_count,
        "hard_structure_top1": hard["structure"] / hard_count,
        "hard_geometry_head_top1": hard["geometry"] / hard_count,
        "hard_nearest_distance_top1": hard["distance"] / hard_count,
        "mean_margin": float(np.mean(margins)) if margins else float("nan"),
        "avg_scale_weights": (
            (scale_weight_sum / max(scale_weight_count, 1)).detach().cpu().tolist()
            if scale_weight_sum is not None
            else []
        ),
        "rows": rows,
    }


def make_dataset(args, cache: Path):
    if args.model == "single_lsf":
        return RealCandidateGroupDataset(
            cache,
            radius=args.radius,
            max_neighbors=args.max_neighbors,
            max_candidates=args.max_candidates,
            neighbor_mode=args.neighbor_mode,
        )
    return MultiScaleRealCandidateGroupDataset(
        cache,
        radii=parse_floats(args.radii),
        max_neighbors=args.max_neighbors,
        max_candidates=args.max_candidates,
        neighbor_mode=args.neighbor_mode,
        knn=parse_ints(args.knn),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local-structure association model.")
    parser.add_argument("--model", choices=["single_lsf", "multiscale_lsf"], default="multiscale_lsf")
    parser.add_argument("--fusion-mode", choices=["residual", "competitive"], default="residual")
    parser.add_argument("--neighbor-mode", choices=["radius", "knn"], default="radius")
    parser.add_argument("--radii", default="10,15,20")
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--knn", default=None, help="Comma-separated K per scale, e.g. 4,8,16.")
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-queries", type=int)
    parser.add_argument("--max-val-queries", type=int)
    parser.add_argument("--max-test-queries", type=int)
    parser.add_argument("--init-checkpoint", type=Path, help="Optional checkpoint used to initialize shared weights.")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train_ds = make_dataset(args, args.train_cache)
    val_ds = make_dataset(args, args.val_cache)
    test_ds = make_dataset(args, args.test_cache)
    if args.max_train_queries:
        train_ds.cache.queries = train_ds.cache.queries[: args.max_train_queries]
    if args.max_val_queries:
        val_ds.cache.queries = val_ds.cache.queries[: args.max_val_queries]
    if args.max_test_queries:
        test_ds.cache.queries = test_ds.cache.queries[: args.max_test_queries]

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = args.device if use_cuda else "cpu"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": use_cuda,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    if args.model == "single_lsf":
        model = CandidateGroupStructureNet(grid_size=24, width=24, fusion_mode=args.fusion_mode).to(device)
        model_config = {
            "grid_size": 24,
            "width": 24,
            "radius": args.radius,
            "neighborhood": args.neighbor_mode,
            "fusion": (
                "motion_structure_competitive_gate"
                if args.fusion_mode == "competitive"
                else "motion_prior_structure_residual"
            ),
            "fusion_mode": args.fusion_mode,
        }
    else:
        radii = parse_floats(args.radii)
        model = MultiScaleCandidateGroupStructureNet(
            grid_size=24,
            width=24,
            num_scales=len(radii),
            fusion_mode=args.fusion_mode,
        ).to(device)
        model_config = {
            "grid_size": 24,
            "width": 24,
            "num_scales": len(radii),
            "radii": radii,
            "neighborhood": args.neighbor_mode,
            "fusion": (
                "motion_structure_competitive_gate"
                if args.fusion_mode == "competitive"
                else "motion_prior_structure_residual"
            ),
            "fusion_mode": args.fusion_mode,
        }
        if args.neighbor_mode == "knn":
            model_config["ks"] = parse_ints(args.knn) or tuple([args.max_neighbors] * len(radii))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.init_checkpoint is not None:
        init = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(init["model"], strict=False)
        print(
            f"[INIT] checkpoint={args.init_checkpoint} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        correct = 0
        count = 0
        for batch in train_loader:
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
            full_loss = F.cross_entropy(out["logits"], moved["target"], reduction="none")
            structure_loss = F.cross_entropy(out["structure_logits"], moved["target"], reduction="none")
            sorted_distance = moved["geometry"][..., 2].masked_fill(~moved["candidate_mask"], 1e4).sort(dim=1).values
            hard_weight = torch.where(
                (sorted_distance[:, 1] - sorted_distance[:, 0]) <= 0.2,
                torch.full_like(full_loss, 2.0),
                torch.ones_like(full_loss),
            )
            loss = ((full_loss + 0.5 * structure_loss) * hard_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            correct += int((out["logits"].argmax(1) == moved["target"]).sum())
            count += len(moved["target"])
        scheduler.step()
        val = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_top1": correct / max(count, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in val.items() if key != "rows"},
            "sigma": float(model.rasterizer.log_sigma.exp().detach().cpu()),
            "structure_weight": float(F.softplus(model.fusion_scale).detach().cpu()),
        }
        history.append(record)
        print(
            f"[Epoch {epoch:02d}] loss={record['train_loss']:.4f} "
            f"train={record['train_top1']:.4f} val={record['val_full_top1']:.4f} "
            f"hard={record['val_hard_full_top1']:.4f}",
            flush=True,
        )
        if val["full_top1"] > best:
            best = val["full_top1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": model_config,
                    "epoch": epoch,
                    "validation": {k: v for k, v in val.items() if k != "rows"},
                },
                args.output / "best.pt",
            )

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader, device)
    with (args.output / "test_query_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_index", "target", "prediction", "correct", "margin"])
        writer.writeheader()
        writer.writerows(test.pop("rows"))
    report = {
        "model": args.model,
        "neighbor_mode": args.neighbor_mode,
        "device": device,
        "model_config": model_config,
        "train_cache_stats": train_ds.cache.stats,
        "val_cache_stats": val_ds.cache.stats,
        "test_cache_stats": test_ds.cache.stats,
        "effective_train_queries": len(train_ds),
        "effective_val_queries": len(val_ds),
        "effective_test_queries": len(test_ds),
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation"],
        "test": test,
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(test, ensure_ascii=False, indent=2))
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
