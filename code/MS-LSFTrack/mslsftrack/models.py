from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def motion_prior_logits(geometry: torch.Tensor) -> torch.Tensor:
    """Monotonic motion prior used as the association-score backbone."""
    return -geometry[..., 2].clamp_min(0.0)


def _candidate_mask(geometry: torch.Tensor, candidate_mask: torch.Tensor | None) -> torch.Tensor:
    if candidate_mask is not None:
        return candidate_mask
    return torch.ones(geometry.shape[:-1], dtype=torch.bool, device=geometry.device)


def masked_group_center(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask_f).sum(dim=1, keepdim=True) / denom
    return values - mean


def masked_group_zscore(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    centered = masked_group_center(values, mask)
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    var = (centered.square() * mask_f).sum(dim=1, keepdim=True) / denom
    return centered / torch.sqrt(var + eps)


def ambiguity_gate_from_geometry(
    geometry: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    margin_low: float = 0.02,
    margin_high: float = 0.20,
) -> torch.Tensor:
    """Return a group-level gate that opens only for ambiguous candidate sets.

    `geometry[..., 2]` is the distance normalized by the association gate. A
    small nearest-vs-second-nearest margin means motion is unreliable, so the
    structure residual should be allowed to influence the final score.
    """
    mask = _candidate_mask(geometry, candidate_mask)
    distance = geometry[..., 2].clamp_min(0.0).masked_fill(~mask, 1e4)
    sorted_distance = distance.sort(dim=1).values
    nearest = sorted_distance[:, :1]
    if sorted_distance.shape[1] > 1:
        second = sorted_distance[:, 1:2]
    else:
        second = nearest + float(margin_high)
    margin = (second - nearest).clamp_min(0.0)
    if margin_high <= margin_low:
        ambiguity = (margin <= margin_low).to(geometry.dtype)
    else:
        ambiguity = 1.0 - ((margin - margin_low) / (margin_high - margin_low)).clamp(0.0, 1.0)

    candidate_count = mask.sum(dim=1, keepdim=True).to(geometry.dtype)
    density_gate = ((candidate_count - 2.0) / 3.0).clamp(0.0, 1.0) * 0.35
    gate = torch.maximum(ambiguity, density_gate)
    gate = torch.where(candidate_count > 1.0, gate, torch.zeros_like(gate))
    return gate.expand_as(distance).masked_fill(~mask, 0.0)


def group_reliability_features(
    geometry: torch.Tensor,
    structure_logits: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build group-level cues for motion-vs-structure reliability gating."""
    mask = _candidate_mask(geometry, candidate_mask)
    mask_f = mask.to(geometry.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

    distance = geometry[..., 2].clamp_min(0.0).masked_fill(~mask, 1e4)
    sorted_distance = distance.sort(dim=1).values
    nearest = sorted_distance[:, :1].clamp(0.0, 1.0)
    if sorted_distance.shape[1] > 1:
        second = sorted_distance[:, 1:2]
    else:
        second = sorted_distance[:, :1] + 0.20
    distance_margin = (second - sorted_distance[:, :1]).clamp(0.0, 1.0)
    motion_ambiguity = ambiguity_gate_from_geometry(geometry, mask)[:, :1]

    candidate_count = mask_f.sum(dim=1, keepdim=True)
    candidate_density = ((candidate_count - 1.0) / 7.0).clamp(0.0, 1.0)
    mean_score = (geometry[..., 3:4] * mask_f[..., None]).sum(dim=1) / denom
    mean_gap = (geometry[..., 4:5] * mask_f[..., None]).sum(dim=1) / denom

    structure_valid = structure_logits.masked_fill(~mask, -1e4)
    if structure_valid.shape[1] > 1:
        top2 = structure_valid.topk(k=2, dim=1).values
        structure_margin = (top2[:, :1] - top2[:, 1:2]).clamp_min(0.0)
    else:
        structure_margin = torch.zeros_like(nearest)
    structure_centered = masked_group_center(structure_logits, mask)
    structure_std = torch.sqrt(
        (structure_centered.square() * mask_f).sum(dim=1, keepdim=True) / denom + 1e-4
    )

    return torch.cat(
        [
            nearest,
            distance_margin,
            motion_ambiguity,
            candidate_density,
            mean_score.clamp(0.0, 1.0),
            mean_gap.clamp(0.0, 1.0),
            torch.tanh(structure_margin),
            torch.tanh(structure_std),
        ],
        dim=1,
    )


def fuse_motion_structure(
    geometry: torch.Tensor,
    structure_logits: torch.Tensor,
    structure_weight: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Fuse monotonic motion prior with a centered, ambiguity-gated LSF residual."""
    mask = _candidate_mask(geometry, candidate_mask)
    motion_logits = motion_prior_logits(geometry)
    structure_residual = masked_group_center(structure_logits, mask)
    structure_gate = ambiguity_gate_from_geometry(geometry, mask)
    logits = motion_logits + structure_weight * structure_gate * structure_residual
    return {
        "logits": logits,
        "motion_logits": motion_logits,
        "structure_residual": structure_residual,
        "structure_gate": structure_gate,
    }


def fuse_motion_structure_competitive(
    geometry: torch.Tensor,
    structure_logits: torch.Tensor,
    fusion_weights: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Fuse motion and structure as two competing, reliability-weighted experts."""
    mask = _candidate_mask(geometry, candidate_mask)
    motion_logits = motion_prior_logits(geometry)
    motion_score = masked_group_zscore(motion_logits, mask)
    structure_score = masked_group_zscore(structure_logits, mask)
    motion_weight = fusion_weights[:, :1]
    structure_weight = fusion_weights[:, 1:2]
    logits = motion_weight * motion_score + structure_weight * structure_score
    return {
        "logits": logits,
        "motion_logits": motion_logits,
        "motion_score": motion_score,
        "structure_score": structure_score,
        "fusion_weights": fusion_weights,
    }


def fuse_motion_structure_fixed(
    geometry: torch.Tensor,
    structure_logits: torch.Tensor,
    motion_weight: float,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Fuse normalized motion and structure scores with a fixed motion weight."""
    mask = _candidate_mask(geometry, candidate_mask)
    motion_logits = motion_prior_logits(geometry)
    motion_score = masked_group_zscore(motion_logits, mask)
    structure_score = masked_group_zscore(structure_logits, mask)
    weight = float(max(0.0, min(1.0, motion_weight)))
    logits = weight * motion_score + (1.0 - weight) * structure_score
    return {
        "logits": logits,
        "motion_logits": motion_logits,
        "motion_score": motion_score,
        "structure_score": structure_score,
        "fixed_fusion_weights": torch.tensor(
            [weight, 1.0 - weight],
            dtype=geometry.dtype,
            device=geometry.device,
        ).view(1, 2),
    }


def make_reliability_gate(feature_dim: int = 8) -> nn.Sequential:
    gate = nn.Sequential(
        nn.Linear(feature_dim, 32),
        nn.LayerNorm(32),
        nn.GELU(),
        nn.Linear(32, 2),
    )
    final = gate[-1]
    if isinstance(final, nn.Linear):
        nn.init.zeros_(final.weight)
        final.bias.data.copy_(torch.tensor([1.0, 0.0]))
    return gate


def fuse_motion_structure_with_mode(
    geometry: torch.Tensor,
    structure_logits: torch.Tensor,
    structure_weight: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    fusion_mode: str = "residual",
    reliability_gate: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    residual = fuse_motion_structure(geometry, structure_logits, structure_weight, candidate_mask)
    result = {
        **residual,
        "residual_logits": residual["logits"],
        "residual_structure_weight": structure_weight,
    }
    if fusion_mode == "residual":
        return result
    if fusion_mode != "competitive":
        raise ValueError(f"unknown fusion_mode: {fusion_mode}")
    if reliability_gate is None:
        raise ValueError("competitive fusion requires a reliability_gate module")
    features = group_reliability_features(geometry, structure_logits, candidate_mask)
    fusion_weights = torch.softmax(reliability_gate(features), dim=-1)
    competitive = fuse_motion_structure_competitive(
        geometry,
        structure_logits,
        fusion_weights,
        candidate_mask,
    )
    result.update(
        {
            "logits": competitive["logits"],
            "competitive_logits": competitive["logits"],
            "motion_score": competitive["motion_score"],
            "structure_score": competitive["structure_score"],
            "fusion_weights": fusion_weights,
            "reliability_features": features,
        }
    )
    return result


class StructureEncoder(nn.Module):
    """Lightweight shared CNN encoder for 2-D local structure fields."""

    def __init__(self, in_channels: int = 2, width: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiameseStructureMatcher(nn.Module):
    """Siamese matcher for historical/current local structure consistency.

    Two structure fields are encoded by a shared CNN. The comparison head uses
    original features, absolute differences and element-wise products, so the
    model does not collapse the local structure map into a single handcrafted
    vector too early.
    """

    def __init__(self, in_channels: int = 2, width: int = 24):
        super().__init__()
        self.encoder = StructureEncoder(in_channels=in_channels, width=width)
        channels = width * 2
        self.compare = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels, 1),
        )

    def forward(self, history: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        hist_feat = self.encoder(history)
        cand_feat = self.encoder(candidate)
        pair = torch.cat(
            [hist_feat, cand_feat, torch.abs(hist_feat - cand_feat), hist_feat * cand_feat],
            dim=1,
        )
        return self.head(self.compare(pair)).squeeze(1)


class GaussianStructureRasterizer(nn.Module):
    """Differentiable local structure field rasterizer.

    The input support points are normalized by their neighborhood radius into
    [-1, 1]. Points outside the neighborhood are removed by the dataset layer;
    the network learns only the soft spatial diffusion scale inside the local
    support region.
    """

    def __init__(self, grid_size: int = 24, initial_sigma: float = 0.09):
        super().__init__()
        coordinates = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy], dim=-1), persistent=False)
        self.log_sigma = nn.Parameter(torch.tensor(float(initial_sigma)).log())

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # points: [..., N, 3], mask: [..., N]
        xy = points[..., :2]
        confidence = points[..., 2]
        prefix = [1] * (xy.ndim - 2)
        grid = self.grid.view(*prefix, 1, *self.grid.shape)
        delta = grid - xy.unsqueeze(-2).unsqueeze(-2)
        dist2 = (delta * delta).sum(dim=-1)
        sigma = self.log_sigma.exp().clamp(0.025, 0.30)
        kernel = torch.exp(-0.5 * dist2 / (sigma * sigma))
        kernel = kernel * mask[..., :, None, None].to(kernel.dtype)
        density = kernel.sum(dim=-3)
        weighted = (kernel * confidence[..., :, None, None]).sum(dim=-3)
        occupancy = 1.0 - torch.exp(-density)
        confidence_field = 1.0 - torch.exp(-weighted)
        return torch.stack([occupancy, confidence_field], dim=-3)


class CandidateGroupStructureNet(nn.Module):
    """Single-scale candidate association network with local structure fields."""

    def __init__(
        self,
        grid_size: int = 24,
        width: int = 24,
        geometry_dim: int = 5,
        fusion_mode: str = "residual",
    ):
        super().__init__()
        if fusion_mode not in {"residual", "competitive"}:
            raise ValueError(f"unknown fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.rasterizer = GaussianStructureRasterizer(grid_size=grid_size)
        self.encoder = StructureEncoder(in_channels=2, width=width)
        channels = width * 2
        self.structure_compare = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )
        self.geometry_head = nn.Sequential(
            nn.Linear(geometry_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.fusion_scale = nn.Parameter(torch.tensor(-2.0))
        if self.fusion_mode == "competitive":
            self.reliability_gate = make_reliability_gate()

    def fuse_scores(
        self,
        geometry: torch.Tensor,
        structure_logits: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        fusion_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = fusion_mode or self.fusion_mode
        return fuse_motion_structure_with_mode(
            geometry,
            structure_logits,
            F.softplus(self.fusion_scale),
            candidate_mask,
            fusion_mode=mode,
            reliability_gate=getattr(self, "reliability_gate", None),
        )

    def forward(
        self,
        history_points: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_points: torch.Tensor,
        candidate_neighbor_mask: torch.Tensor,
        geometry: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, candidates = candidate_points.shape[:2]
        history_field = self.rasterizer(history_points, history_mask)
        candidate_field = self.rasterizer(candidate_points, candidate_neighbor_mask)
        history_feature = self.encoder(history_field)
        candidate_feature = self.encoder(candidate_field.flatten(0, 1)).view(
            batch,
            candidates,
            -1,
            history_feature.shape[-2],
            history_feature.shape[-1],
        )
        expanded_history = history_feature[:, None].expand_as(candidate_feature)
        pair = torch.cat(
            [
                expanded_history,
                candidate_feature,
                torch.abs(expanded_history - candidate_feature),
                expanded_history * candidate_feature,
            ],
            dim=2,
        )
        structure_logits = self.structure_compare(pair.flatten(0, 1)).view(batch, candidates)
        geometry_mlp_logits = self.geometry_head(geometry).squeeze(-1)
        fused = self.fuse_scores(geometry, structure_logits, candidate_mask)
        logits = fused["logits"]
        motion_logits = fused["motion_logits"]
        structure_residual = fused["structure_residual"]
        structure_gate = fused["structure_gate"]
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -1e4)
            structure_logits = structure_logits.masked_fill(~candidate_mask, -1e4)
            geometry_mlp_logits = geometry_mlp_logits.masked_fill(~candidate_mask, -1e4)
            motion_logits = motion_logits.masked_fill(~candidate_mask, -1e4)
            structure_residual = structure_residual.masked_fill(~candidate_mask, 0.0)
            structure_gate = structure_gate.masked_fill(~candidate_mask, 0.0)
        return {
            "logits": logits,
            "structure_logits": structure_logits,
            "geometry_logits": motion_logits,
            "motion_logits": motion_logits,
            "geometry_mlp_logits": geometry_mlp_logits,
            "structure_residual": structure_residual,
            "structure_gate": structure_gate,
            "history_field": history_field,
            "candidate_field": candidate_field,
            "structure_weight": fused["residual_structure_weight"],
            **{
                key: value
                for key, value in fused.items()
                if key
                in {
                    "residual_logits",
                    "competitive_logits",
                    "motion_score",
                    "structure_score",
                    "fusion_weights",
                    "reliability_features",
                }
            },
        }


class MultiScaleCandidateGroupStructureNet(nn.Module):
    """Multi-scale local structure field association network.

    For each history/candidate pair, local support detections are rasterized
    under several neighborhood radii and encoded by a shared CNN. Each scale
    produces a structure-consistency score. A lightweight gate predicts the
    scale fusion weights from candidate geometry, reducing reliance on a single
    manually chosen local radius.
    """

    def __init__(
        self,
        grid_size: int = 24,
        width: int = 24,
        geometry_dim: int = 5,
        num_scales: int = 3,
        fusion_mode: str = "residual",
    ):
        super().__init__()
        if fusion_mode not in {"residual", "competitive"}:
            raise ValueError(f"unknown fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.num_scales = int(num_scales)
        self.rasterizer = GaussianStructureRasterizer(grid_size=grid_size)
        self.encoder = StructureEncoder(in_channels=2, width=width)
        channels = width * 2
        self.scale_compare = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(geometry_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, self.num_scales),
        )
        self.scale_prior = nn.Parameter(torch.zeros(self.num_scales))
        self.geometry_head = nn.Sequential(
            nn.Linear(geometry_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.fusion_scale = nn.Parameter(torch.tensor(-2.0))
        if self.fusion_mode == "competitive":
            self.reliability_gate = make_reliability_gate()

    def fuse_scores(
        self,
        geometry: torch.Tensor,
        structure_logits: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        fusion_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = fusion_mode or self.fusion_mode
        return fuse_motion_structure_with_mode(
            geometry,
            structure_logits,
            F.softplus(self.fusion_scale),
            candidate_mask,
            fusion_mode=mode,
            reliability_gate=getattr(self, "reliability_gate", None),
        )

    def forward(
        self,
        history_points: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_points: torch.Tensor,
        candidate_neighbor_mask: torch.Tensor,
        geometry: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # history_points: [B, S, N, 3]
        # candidate_points: [B, C, S, N, 3]
        batch, scales = history_points.shape[:2]
        candidates = candidate_points.shape[1]
        if scales != self.num_scales:
            raise ValueError(f"expected {self.num_scales} scales, got {scales}")

        history_field = self.rasterizer(history_points, history_mask)
        candidate_field = self.rasterizer(candidate_points, candidate_neighbor_mask)
        history_feature = self.encoder(history_field.flatten(0, 1)).view(
            batch,
            scales,
            -1,
            history_field.shape[-2] // 4,
            history_field.shape[-1] // 4,
        )
        candidate_feature = self.encoder(candidate_field.flatten(0, 2)).view(
            batch,
            candidates,
            scales,
            -1,
            history_feature.shape[-2],
            history_feature.shape[-1],
        )
        expanded_history = history_feature[:, None].expand_as(candidate_feature)
        pair = torch.cat(
            [
                expanded_history,
                candidate_feature,
                torch.abs(expanded_history - candidate_feature),
                expanded_history * candidate_feature,
            ],
            dim=3,
        )
        scale_logits = self.scale_compare(pair.flatten(0, 2)).view(batch, candidates, scales)
        scale_weights = torch.softmax(self.scale_gate(geometry) + self.scale_prior, dim=-1)
        structure_logits = (scale_logits * scale_weights).sum(dim=-1)
        geometry_mlp_logits = self.geometry_head(geometry).squeeze(-1)
        fused = self.fuse_scores(geometry, structure_logits, candidate_mask)
        logits = fused["logits"]
        motion_logits = fused["motion_logits"]
        structure_residual = fused["structure_residual"]
        structure_gate = fused["structure_gate"]
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -1e4)
            structure_logits = structure_logits.masked_fill(~candidate_mask, -1e4)
            geometry_mlp_logits = geometry_mlp_logits.masked_fill(~candidate_mask, -1e4)
            motion_logits = motion_logits.masked_fill(~candidate_mask, -1e4)
            structure_residual = structure_residual.masked_fill(~candidate_mask, 0.0)
            structure_gate = structure_gate.masked_fill(~candidate_mask, 0.0)
            scale_logits = scale_logits.masked_fill(~candidate_mask[..., None], -1e4)
        return {
            "logits": logits,
            "structure_logits": structure_logits,
            "geometry_logits": motion_logits,
            "motion_logits": motion_logits,
            "geometry_mlp_logits": geometry_mlp_logits,
            "structure_residual": structure_residual,
            "structure_gate": structure_gate,
            "scale_logits": scale_logits,
            "scale_weights": scale_weights,
            "history_field": history_field,
            "candidate_field": candidate_field,
            "structure_weight": fused["residual_structure_weight"],
            **{
                key: value
                for key, value in fused.items()
                if key
                in {
                    "residual_logits",
                    "competitive_logits",
                    "motion_score",
                    "structure_score",
                    "fusion_weights",
                    "reliability_features",
                }
            },
        }
