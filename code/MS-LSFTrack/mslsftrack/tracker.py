from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .models import (
    CandidateGroupStructureNet,
    MultiScaleCandidateGroupStructureNet,
    fuse_motion_structure_fixed,
    masked_group_zscore,
)
from .mot_io import SceneDetections
from .datasets import extract_neighbors


def frame_box_density_rho(scene: SceneDetections, detections: np.ndarray) -> float | None:
    if len(detections) < 2:
        return None
    xy = scene.xy[detections].astype(np.float32)
    wh = np.maximum(scene.wh[detections].astype(np.float32), 1e-6)
    center_delta = np.abs(xy[:, None, :] - xy[None, :, :])
    size_sum = (wh[:, None, :] + wh[None, :, :]) * 0.5
    edge_delta = np.maximum(center_delta - size_sum, 0.0)
    distance = np.linalg.norm(edge_delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.min(distance, axis=1)
    finite = np.isfinite(nearest)
    if not np.any(finite):
        return None
    amid = float(np.mean(nearest[finite]))
    target_size = float(np.mean(np.sqrt(np.maximum(wh[:, 0] * wh[:, 1], 1e-6))))
    return target_size / max(amid, 1e-6)


@dataclass
class Track:
    track_id: int
    detection_indices: list[int] = field(default_factory=list)
    missing: int = 0
    confidence: float = 1.0

    @property
    def last_index(self) -> int:
        return self.detection_indices[-1]

    @property
    def length(self) -> int:
        return len(self.detection_indices)

    def predict(
        self,
        scene: SceneDetections,
        frame: int,
        history_size: int = 20,
        quadratic_min_points: int = 4,
    ) -> np.ndarray:
        last = self.last_index
        if len(self.detection_indices) < 2:
            return scene.xy[last].copy()
        indices = self.detection_indices[-history_size:]
        frames = scene.frames[indices].astype(np.float32)
        xy = scene.xy[indices].astype(np.float32)
        if len(indices) >= quadratic_min_points and np.unique(frames).size >= quadratic_min_points:
            try:
                degree = 2 if len(indices) >= quadratic_min_points else 1
                px = np.polyfit(frames, xy[:, 0], deg=degree)
                py = np.polyfit(frames, xy[:, 1], deg=degree)
                return np.asarray([np.polyval(px, frame), np.polyval(py, frame)], dtype=np.float32)
            except np.linalg.LinAlgError:
                pass
        previous = self.detection_indices[-2]
        dt = max(1, int(scene.frames[last] - scene.frames[previous]))
        gap = max(1, frame - int(scene.frames[last]))
        velocity = (scene.xy[last] - scene.xy[previous]) / float(dt)
        return scene.xy[last] + velocity * float(gap)

    def dynamic_gate(
        self,
        base_radius: float,
        frame: int,
        scene: SceneDetections,
        max_radius: float,
        missing_growth: float,
    ) -> float:
        gap = max(1, frame - int(scene.frames[self.last_index]))
        # Keep mature tracks slightly more tolerant, while keeping new tracks conservative.
        mature_bonus = min(2.0, 0.25 * max(0, self.length - 2))
        missing_bonus = missing_growth * max(0, gap - 1)
        return float(min(max_radius, base_radius + mature_bonus + missing_bonus))

    def update(self, det_index: int) -> None:
        self.detection_indices.append(det_index)
        self.missing = 0
        self.confidence = min(1.0, self.confidence * 0.98 + 0.04)

    def miss(self) -> None:
        self.missing += 1
        self.confidence = max(0.0, self.confidence * 0.95 - 0.01 * self.missing)


class LocalStructureFieldTracker:
    def __init__(
        self,
        model: CandidateGroupStructureNet,
        device: str,
        gate_radius: float = 10.0,
        structure_radius: float = 20.0,
        structure_radii: tuple[float, ...] | None = None,
        neighbor_mode: str = "radius",
        knn: tuple[int, ...] | None = None,
        max_neighbors: int = 32,
        max_missing: int = 20,
        min_track_length: int = 25,
        score_threshold: float = -4.0,
        inference_batch_size: int = 256,
        score_mode: str = "full",
        scale_mode: str = "learned",
        fixed_motion_weight: float = 0.5,
        max_gate_radius: float = 16.0,
        missing_gate_growth: float = 0.75,
        missing_score_penalty: float = 0.05,
        distance_score_weight: float = 0.0,
        distance_prior_mode: str = "fixed",
        adaptive_distance_margin_low: float = 0.02,
        adaptive_distance_margin_high: float = 0.20,
        adaptive_distance_min_factor: float = 0.0,
        density_tau_low: float = 0.05,
        density_tau_high: float = 0.20,
        density_init: float | None = None,
        density_alpha: float = 0.8,
        density_beta: float = 2.0,
        density_delta: float = 0.05,
        density_prior_lambda: float = 1.0,
        density_prior_direction: str = "normal",
        asr_density_gain: float = 1.0,
        mature_birth_length: int = 2,
        assignment: str = "hungarian",
    ):
        self.model = model.eval()
        self.device = device
        self.gate_radius = gate_radius
        self.structure_radius = structure_radius
        self.structure_radii = tuple(structure_radii) if structure_radii is not None else None
        self.neighbor_mode = neighbor_mode
        self.knn = tuple(knn) if knn is not None else None
        self.max_neighbors = max_neighbors
        self.max_missing = max_missing
        self.min_track_length = min_track_length
        self.score_threshold = score_threshold
        self.inference_batch_size = inference_batch_size
        self.score_mode = score_mode
        self.scale_mode = scale_mode
        self.fixed_motion_weight = float(fixed_motion_weight)
        self.max_gate_radius = max_gate_radius
        self.missing_gate_growth = missing_gate_growth
        self.missing_score_penalty = missing_score_penalty
        self.distance_score_weight = distance_score_weight
        self.distance_prior_mode = distance_prior_mode
        self.adaptive_distance_margin_low = adaptive_distance_margin_low
        self.adaptive_distance_margin_high = adaptive_distance_margin_high
        self.adaptive_distance_min_factor = adaptive_distance_min_factor
        self.density_tau_low = float(density_tau_low)
        self.density_tau_high = float(density_tau_high)
        self.density_state = float(density_init) if density_init is not None else None
        self.density_alpha = float(density_alpha)
        self.density_beta = float(density_beta)
        self.density_delta = float(density_delta)
        self.density_prior_lambda = float(density_prior_lambda)
        if density_prior_direction not in {"normal", "reverse"}:
            raise ValueError(f"unknown density_prior_direction: {density_prior_direction}")
        self.density_prior_direction = density_prior_direction
        self.asr_density_gain = float(asr_density_gain)
        self._density_scale_prior: np.ndarray | None = None
        self._density_q: float = 0.5
        self.mature_birth_length = mature_birth_length
        self.assignment = assignment

    def _neighbors(self, scene: SceneDetections, center_idx: int):
        if self.structure_radii is None:
            return extract_neighbors(
                scene,
                center_idx,
                self.structure_radius,
                self.max_neighbors,
                neighbor_mode=self.neighbor_mode,
            )
        points = []
        masks = []
        for scale_idx, radius in enumerate(self.structure_radii):
            selection_limit = None
            if self.neighbor_mode == "knn" and self.knn is not None:
                selection_limit = min(self.max_neighbors, self.knn[scale_idx])
            scale_points, scale_mask = extract_neighbors(
                scene,
                center_idx,
                radius,
                self.max_neighbors,
                neighbor_mode=self.neighbor_mode,
                selection_limit=selection_limit,
            )
            points.append(scale_points)
            masks.append(scale_mask)
        return np.stack(points, axis=0), np.stack(masks, axis=0)

    def _pair_features(
        self,
        scene: SceneDetections,
        track: Track,
        candidate_idx: int,
        frame: int,
    ):
        history_points, history_mask = self._neighbors(scene, track.last_index)
        candidate_points, candidate_mask = self._neighbors(scene, candidate_idx)
        predicted = track.predict(scene, frame)
        delta = scene.xy[candidate_idx] - predicted
        distance = float(np.linalg.norm(delta))
        gap = max(1, frame - int(scene.frames[track.last_index]))
        geometry = np.asarray(
            [
                delta[0] / self.gate_radius,
                delta[1] / self.gate_radius,
                distance / self.gate_radius,
                scene.scores[candidate_idx],
                min(gap / 3.0, 1.0),
            ],
            dtype=np.float32,
        )
        return history_points, history_mask, candidate_points, candidate_mask, geometry

    @staticmethod
    def _pad_group(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        max_candidates = max(len(group) for group in values)
        sample = values[0][0]
        padded = np.zeros((len(values), max_candidates, *sample.shape), dtype=sample.dtype)
        mask = np.zeros((len(values), max_candidates), dtype=np.bool_)
        for group_idx, group in enumerate(values):
            for candidate_idx, value in enumerate(group):
                padded[group_idx, candidate_idx] = value
                mask[group_idx, candidate_idx] = True
        return padded, mask

    def _density_prior_tensor(self, scale_logits: torch.Tensor) -> torch.Tensor:
        num_scales = scale_logits.shape[-1]
        prior = self._density_scale_prior
        if prior is None or len(prior) != num_scales:
            prior = np.full(num_scales, 1.0 / float(num_scales), dtype=np.float32)
        return torch.as_tensor(prior, dtype=scale_logits.dtype, device=scale_logits.device)

    def _scale_weights_for_scale_mode(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        scale_logits = output["scale_logits"]
        if self.scale_mode == "learned":
            return output.get("scale_weights", torch.full_like(scale_logits, 1.0 / scale_logits.shape[-1]))
        if self.scale_mode == "uniform":
            return torch.full_like(scale_logits, 1.0 / scale_logits.shape[-1])
        if self.scale_mode == "small":
            weights = torch.zeros_like(scale_logits)
            weights[..., 0] = 1.0
            return weights
        if self.scale_mode == "middle":
            weights = torch.zeros_like(scale_logits)
            weights[..., min(1, weights.shape[-1] - 1)] = 1.0
            return weights
        if self.scale_mode == "large":
            weights = torch.zeros_like(scale_logits)
            weights[..., -1] = 1.0
            return weights
        if self.scale_mode in {"density", "density_soft", "density_only"}:
            prior = self._density_prior_tensor(scale_logits)
            return prior.expand_as(scale_logits)
        if self.scale_mode == "learned_density":
            prior = self._density_prior_tensor(scale_logits).pow(self.density_prior_lambda)
            learned = output.get("scale_weights")
            if learned is None:
                return prior.expand_as(scale_logits)
            weights = learned * prior
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        raise ValueError(f"unknown scale mode: {self.scale_mode}")

    def _structure_logits_for_scale_mode(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.scale_mode == "learned" or "scale_logits" not in output:
            return output["structure_logits"]
        weights = self._scale_weights_for_scale_mode(output)
        return (output["scale_logits"] * weights).sum(dim=-1)

    def _scores_with_scale_mode(
        self,
        output: dict[str, torch.Tensor],
        geometry: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.scale_mode == "learned" or "scale_logits" not in output:
            return output
        scale_weights = self._scale_weights_for_scale_mode(output)
        structure_logits = self._structure_logits_for_scale_mode(output)
        fused = self.model.fuse_scores(geometry, structure_logits, mask)
        updated = {**output, **fused, "structure_logits": structure_logits, "scale_weights": scale_weights}
        if mask is not None:
            updated["logits"] = updated["logits"].masked_fill(~mask, -1e4)
            updated["structure_logits"] = updated["structure_logits"].masked_fill(~mask, -1e4)
            updated["motion_logits"] = updated["motion_logits"].masked_fill(~mask, -1e4)
            updated["residual_logits"] = updated["residual_logits"].masked_fill(~mask, -1e4)
            if "competitive_logits" in updated:
                updated["competitive_logits"] = updated["competitive_logits"].masked_fill(~mask, -1e4)
        return updated

    def _update_density_scale_prior(self, scene: SceneDetections, detections: np.ndarray) -> None:
        if self.structure_radii is None:
            self._density_scale_prior = None
            return
        rho = frame_box_density_rho(scene, detections)
        if rho is not None:
            if self.density_state is None:
                self.density_state = float(rho)
            else:
                alpha = float(np.clip(self.density_alpha, 0.0, 1.0))
                self.density_state = alpha * self.density_state + (1.0 - alpha) * float(rho)
        if self.density_state is None:
            self._density_scale_prior = np.full(len(self.structure_radii), 1.0 / len(self.structure_radii), dtype=np.float32)
            self._density_q = 0.5
            return
        low = self.density_tau_low
        high = self.density_tau_high
        q = 0.5 if high <= low else float(np.clip((self.density_state - low) / (high - low), 0.0, 1.0))
        self._density_q = q
        if self.density_prior_direction == "reverse":
            q = 1.0 - q
        centers = np.linspace(1.0, 0.0, len(self.structure_radii), dtype=np.float32)
        activations = 1.0 - np.abs(q - centers) * 2.0
        activations = np.maximum(activations, 0.0) + max(0.0, self.density_delta)
        logits = activations * self.density_beta
        logits = logits - float(np.max(logits))
        weights = np.exp(logits).astype(np.float32)
        self._density_scale_prior = weights / max(float(weights.sum()), 1e-6)

    def _select_scores(
        self,
        output: dict[str, torch.Tensor],
        geometry: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self._scores_with_scale_mode(output, geometry, mask)
        if self.score_mode == "full":
            return output["logits"]
        if self.score_mode == "residual":
            return output.get("residual_logits", output["logits"])
        if self.score_mode == "competitive":
            if "competitive_logits" not in output:
                raise ValueError(
                    "score_mode=competitive requires a checkpoint/model with fusion_mode=competitive"
                )
            return output["competitive_logits"]
        if self.score_mode == "fixed_fusion":
            return fuse_motion_structure_fixed(
                geometry,
                output["structure_logits"],
                self.fixed_motion_weight,
                mask,
            )["logits"]
        if self.score_mode == "structure":
            return output["structure_logits"]
        if self.score_mode == "structure_zscore":
            return masked_group_zscore(output["structure_logits"], mask)
        if self.score_mode == "geometry":
            return output["geometry_logits"]
        if self.score_mode == "field_l1":
            diff = torch.abs(output["candidate_field"] - output["history_field"][:, None])
            if diff.ndim == 6:
                scale_l1 = diff.mean(dim=(3, 4, 5))
                if "scale_weights" in output:
                    return -(scale_l1 * output["scale_weights"]).sum(dim=-1)
                return -scale_l1.mean(dim=-1)
            if diff.ndim == 5:
                return -diff.mean(dim=(2, 3, 4))
            raise ValueError(f"unexpected field diff shape: {tuple(diff.shape)}")
        if self.score_mode == "distance":
            return -geometry[..., 2]
        if self.score_mode == "last_distance":
            raise ValueError("score_mode=last_distance is handled before model inference")
        if self.score_mode in {"agrf_lite", "agrf_full", "rank_agrf"}:
            raise ValueError(f"score_mode={self.score_mode} is handled by AGRF scoring")
        if self.score_mode == "safe_fixed":
            raise ValueError("score_mode=safe_fixed is handled by safe fusion scoring")
        if self.score_mode in {"expert_select", "expert_pool"}:
            raise ValueError(f"score_mode={self.score_mode} is handled by expert fusion scoring")
        if self.score_mode in {"ambiguity_rerank_strict", "ambiguity_rerank", "ambiguity_rerank_loose"}:
            raise ValueError(f"score_mode={self.score_mode} is handled by ambiguity rerank scoring")
        raise ValueError(f"unknown score mode: {self.score_mode}")

    @staticmethod
    def _pad_float_groups(values: list[list[float]], fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        max_candidates = max(len(group) for group in values)
        padded = np.full((len(values), max_candidates), fill, dtype=np.float32)
        mask = np.zeros((len(values), max_candidates), dtype=np.bool_)
        for group_idx, group in enumerate(values):
            if not group:
                continue
            padded[group_idx, : len(group)] = np.asarray(group, dtype=np.float32)
            mask[group_idx, : len(group)] = True
        return padded, mask

    @staticmethod
    def _masked_zscore_np(values: np.ndarray, mask: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        mask_f = mask.astype(np.float32)
        denom = np.maximum(mask_f.sum(axis=1, keepdims=True), 1.0)
        mean = (values * mask_f).sum(axis=1, keepdims=True) / denom
        centered = values - mean
        var = (centered * centered * mask_f).sum(axis=1, keepdims=True) / denom
        scores = centered / np.sqrt(var + eps)
        return np.where(mask, scores, -1e6).astype(np.float32)

    @staticmethod
    def _masked_margin_np(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid_count = mask.sum(axis=1)
        masked = np.where(mask, scores, -1e6)
        ordered = np.sort(masked, axis=1)
        top1 = ordered[:, -1]
        top2 = ordered[:, -2] if scores.shape[1] > 1 else top1
        margin = np.where(valid_count > 1, top1 - top2, 8.0)
        return np.maximum(margin.astype(np.float32), 0.0)

    @staticmethod
    def _masked_smallest_margin_np(costs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid_count = mask.sum(axis=1)
        masked = np.where(mask, costs, 1e6)
        ordered = np.sort(masked, axis=1)
        top1 = ordered[:, 0]
        top2 = ordered[:, 1] if costs.shape[1] > 1 else top1
        margin = np.where(valid_count > 1, top2 - top1, 8.0)
        return np.maximum(margin.astype(np.float32), 0.0)

    @staticmethod
    def _masked_rank_score_np(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rank_score = np.full_like(scores, -1e6, dtype=np.float32)
        for row_idx in range(scores.shape[0]):
            valid = np.where(mask[row_idx])[0]
            if len(valid) == 0:
                continue
            order = valid[np.argsort(-scores[row_idx, valid])]
            ranks = np.empty(len(order), dtype=np.float32)
            ranks[np.arange(len(order))] = np.arange(1, len(order) + 1, dtype=np.float32)
            raw = 1.0 / ranks
            mapped = np.empty_like(raw)
            mapped[np.argsort(order)] = raw
            rank_score[row_idx, valid] = mapped
        return LocalStructureFieldTracker._masked_zscore_np(rank_score, mask)

    @staticmethod
    def _sigmoid_np(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))

    def _agrf_scores(
        self,
        output: dict[str, torch.Tensor],
        geometry: torch.Tensor,
        candidate_mask: torch.Tensor,
        part: list[dict],
    ) -> np.ndarray:
        output = self._scores_with_scale_mode(output, geometry, candidate_mask)
        structure = masked_group_zscore(output["structure_logits"], candidate_mask).detach().cpu().numpy().astype(np.float32)
        mask = candidate_mask.detach().cpu().numpy().astype(np.bool_)
        last_dist, _ = self._pad_float_groups([g["last_distances"] for g in part], fill=1e6)
        pred_dist, _ = self._pad_float_groups([g["pred_distances"] for g in part], fill=1e6)
        gates = np.asarray([max(float(g["gate"]), 1e-6) for g in part], dtype=np.float32)[:, None]
        last_score = self._masked_zscore_np(-(last_dist / gates), mask)
        pred_score = self._masked_zscore_np(-(pred_dist / gates), mask)

        if self.score_mode == "rank_agrf":
            structure_used = self._masked_rank_score_np(structure, mask)
            last_used = self._masked_rank_score_np(last_score, mask)
            pred_used = self._masked_rank_score_np(pred_score, mask)
        else:
            structure_used = structure
            last_used = last_score
            pred_used = pred_score

        if self.score_mode == "agrf_lite":
            base = last_used
            gamma = np.zeros((len(part), 1), dtype=np.float32)
        else:
            pred_margin = self._masked_margin_np(pred_used, mask)
            last_top = np.argmax(np.where(mask, last_used, -1e6), axis=1)
            pred_top = np.argmax(np.where(mask, pred_used, -1e6), axis=1)
            agreement = (last_top == pred_top).astype(np.float32)
            track_score = np.asarray([min(float(g["track_length"]) / 8.0, 1.0) for g in part], dtype=np.float32)
            gap_score = np.asarray([min(max(float(g["gap"]) - 1.0, 0.0) / 3.0, 1.0) for g in part], dtype=np.float32)
            reliability = 1.20 * pred_margin + 0.80 * agreement + 0.40 * track_score - 1.00 * gap_score - 1.10
            gamma = (0.05 + 0.55 * self._sigmoid_np(reliability)).astype(np.float32)[:, None]
            base = (1.0 - gamma) * last_used + gamma * pred_used

        base_margin = self._masked_margin_np(base, mask)
        structure_margin = self._masked_margin_np(structure_used, mask)
        scale_conf = np.zeros(len(part), dtype=np.float32)
        if "scale_weights" in output:
            scale_weights = output["scale_weights"].detach().cpu().numpy().astype(np.float32)
            if scale_weights.ndim == 3 and scale_weights.shape[-1] > 1:
                entropy = -(scale_weights * np.log(np.maximum(scale_weights, 1e-6))).sum(axis=-1)
                conf = 1.0 - entropy / np.log(float(scale_weights.shape[-1]))
                mask_f = mask.astype(np.float32)
                scale_conf = (conf * mask_f).sum(axis=1) / np.maximum(mask_f.sum(axis=1), 1.0)
        density_q = float(np.clip(self._density_q, 0.0, 1.0))
        ambiguity = np.exp(-base_margin / 0.75).astype(np.float32)
        structure_conf = self._sigmoid_np(1.35 * structure_margin + 0.85 * scale_conf + 0.55 * density_q * structure_margin - 1.15).astype(np.float32)
        eta = 1.15 * ambiguity * structure_conf * (1.0 + 0.25 * density_q * structure_conf)
        if self.score_mode == "rank_agrf":
            eta = np.minimum(eta, 0.95)
        else:
            eta = np.minimum(eta, 1.25)
        final = base + eta[:, None] * structure_used
        return np.where(mask, final, -1e6).astype(np.float32)

    def _safe_fixed_scores(
        self,
        output: dict[str, torch.Tensor],
        geometry: torch.Tensor,
        candidate_mask: torch.Tensor,
        part: list[dict],
    ) -> np.ndarray:
        output = self._scores_with_scale_mode(output, geometry, candidate_mask)
        mask = candidate_mask.detach().cpu().numpy().astype(np.bool_)
        last_dist, _ = self._pad_float_groups([g["last_distances"] for g in part], fill=1e6)
        gates = np.asarray([max(float(g["gate"]), 1e-6) for g in part], dtype=np.float32)[:, None]
        last_score = self._masked_zscore_np(-(last_dist / gates), mask)
        fixed = fuse_motion_structure_fixed(
            geometry,
            output["structure_logits"],
            self.fixed_motion_weight,
            candidate_mask,
        )["logits"].detach().cpu().numpy().astype(np.float32)
        fixed_score = self._masked_zscore_np(fixed, mask)

        structure = masked_group_zscore(output["structure_logits"], candidate_mask).detach().cpu().numpy().astype(np.float32)
        base_margin = self._masked_margin_np(last_score, mask)
        fixed_margin = self._masked_margin_np(fixed_score, mask)
        structure_margin = self._masked_margin_np(structure, mask)
        scale_conf = np.zeros(len(part), dtype=np.float32)
        if "scale_weights" in output:
            scale_weights = output["scale_weights"].detach().cpu().numpy().astype(np.float32)
            if scale_weights.ndim == 3 and scale_weights.shape[-1] > 1:
                entropy = -(scale_weights * np.log(np.maximum(scale_weights, 1e-6))).sum(axis=-1)
                conf = 1.0 - entropy / np.log(float(scale_weights.shape[-1]))
                mask_f = mask.astype(np.float32)
                scale_conf = (conf * mask_f).sum(axis=1) / np.maximum(mask_f.sum(axis=1), 1.0)
        ambiguity = np.exp(-base_margin / 0.60).astype(np.float32)
        density_q = float(np.clip(self._density_q, 0.0, 1.0))
        alternative_conf = self._sigmoid_np(
            1.10 * fixed_margin + 0.75 * structure_margin + 0.50 * scale_conf + 0.35 * density_q - 1.20
        ).astype(np.float32)
        switch = np.minimum(0.95, ambiguity * alternative_conf)
        final = (1.0 - switch[:, None]) * last_score + switch[:, None] * fixed_score
        return np.where(mask, final, -1e6).astype(np.float32)

    def _expert_fusion_scores(
        self,
        output: dict[str, torch.Tensor],
        geometry: torch.Tensor,
        candidate_mask: torch.Tensor,
        part: list[dict],
    ) -> np.ndarray:
        output = self._scores_with_scale_mode(output, geometry, candidate_mask)
        mask = candidate_mask.detach().cpu().numpy().astype(np.bool_)
        last_dist, _ = self._pad_float_groups([g["last_distances"] for g in part], fill=1e6)
        gates = np.asarray([max(float(g["gate"]), 1e-6) for g in part], dtype=np.float32)[:, None]
        last_score = self._masked_zscore_np(-(last_dist / gates), mask)
        fixed = fuse_motion_structure_fixed(
            geometry,
            output["structure_logits"],
            self.fixed_motion_weight,
            candidate_mask,
        )["logits"].detach().cpu().numpy().astype(np.float32)
        fixed_score = self._masked_zscore_np(fixed, mask)

        if self.score_mode == "expert_select":
            last_margin = self._masked_margin_np(last_score, mask)
            fixed_margin = self._masked_margin_np(fixed_score, mask)
            use_fixed = (fixed_margin > last_margin).astype(np.float32)[:, None]
            final = (1.0 - use_fixed) * last_score + use_fixed * fixed_score
        else:
            temperature = 0.35
            stacked = np.stack([last_score, fixed_score], axis=0) / temperature
            stacked = np.where(np.stack([mask, mask], axis=0), stacked, -1e6)
            max_score = np.max(stacked, axis=0, keepdims=False)
            pooled = temperature * (
                max_score + np.log(np.exp(stacked[0] - max_score) + np.exp(stacked[1] - max_score) + 1e-6)
            )
            final = self._masked_zscore_np(pooled, mask)
        return np.where(mask, final, -1e6).astype(np.float32)

    def _ambiguity_rerank_scores(
        self,
        output: dict[str, torch.Tensor],
        geometry: torch.Tensor,
        candidate_mask: torch.Tensor,
        part: list[dict],
    ) -> np.ndarray:
        output = self._scores_with_scale_mode(output, geometry, candidate_mask)
        mask = candidate_mask.detach().cpu().numpy().astype(np.bool_)
        last_dist, _ = self._pad_float_groups([g["last_distances"] for g in part], fill=1e6)
        distance_scale = max(float(self.gate_radius), 1e-6)
        distance_cost = last_dist / distance_scale
        base = -distance_cost
        structure = masked_group_zscore(output["structure_logits"], candidate_mask).detach().cpu().numpy().astype(np.float32)

        if self.score_mode == "ambiguity_rerank_strict":
            topk, ambiguity_tau, ambiguity_width, structure_tau, max_gain = 2, 0.06, 0.03, 0.55, 0.06
        elif self.score_mode == "ambiguity_rerank_loose":
            topk, ambiguity_tau, ambiguity_width, structure_tau, max_gain = 3, 0.16, 0.05, 0.20, 0.16
        else:
            topk, ambiguity_tau, ambiguity_width, structure_tau, max_gain = 3, 0.10, 0.04, 0.35, 0.10

        distance_margin = self._masked_smallest_margin_np(distance_cost, mask)
        structure_margin = self._masked_margin_np(structure, mask)
        scale_conf = np.zeros(len(part), dtype=np.float32)
        if "scale_weights" in output:
            scale_weights = output["scale_weights"].detach().cpu().numpy().astype(np.float32)
            if scale_weights.ndim == 3 and scale_weights.shape[-1] > 1:
                entropy = -(scale_weights * np.log(np.maximum(scale_weights, 1e-6))).sum(axis=-1)
                conf = 1.0 - entropy / np.log(float(scale_weights.shape[-1]))
                mask_f = mask.astype(np.float32)
                scale_conf = (conf * mask_f).sum(axis=1) / np.maximum(mask_f.sum(axis=1), 1.0)

        ambiguity = self._sigmoid_np((ambiguity_tau - distance_margin) / max(ambiguity_width, 1e-6))
        structure_conf = self._sigmoid_np((structure_margin - structure_tau) / 0.35)
        density_q = float(np.clip(self._density_q, 0.0, 1.0))
        density_factor = 0.85 + 0.30 * density_q
        density_factor = 1.0 + self.asr_density_gain * (density_factor - 1.0)
        gain = (
            max_gain
            * ambiguity
            * structure_conf
            * (0.75 + 0.25 * scale_conf)
            * density_factor
        )
        gain = np.minimum(gain.astype(np.float32), max_gain)

        final = base.copy().astype(np.float32)
        masked_base = np.where(mask, base, -1e6)
        for row_idx in range(final.shape[0]):
            if gain[row_idx] <= 1e-6:
                continue
            valid = np.where(mask[row_idx])[0]
            if len(valid) <= 1:
                continue
            order = valid[np.argsort(-masked_base[row_idx, valid])[: min(topk, len(valid))]]
            local_structure = structure[row_idx, order].astype(np.float32)
            if len(order) > 1:
                local_structure = (local_structure - float(local_structure.mean())) / (float(local_structure.std()) + 1e-4)
            final[row_idx, order] = base[row_idx, order] + gain[row_idx] * local_structure
        return np.where(mask, final, -1e6).astype(np.float32)


    @torch.no_grad()
    def _score_groups(self, groups: list[dict]) -> list[np.ndarray]:
        if not groups:
            return []
        if self.score_mode in {"distance", "geometry", "last_distance"}:
            scale = max(float(self.gate_radius), 1e-6)
            return [
                -(np.asarray(group["candidate_distances"], dtype=np.float32) / scale)
                for group in groups
            ]
        group_scores: list[np.ndarray] = []
        for start in range(0, len(groups), self.inference_batch_size):
            part = groups[start : start + self.inference_batch_size]
            history = torch.from_numpy(np.stack([g["history_points"] for g in part])).to(self.device)
            history_mask = torch.from_numpy(np.stack([g["history_mask"] for g in part])).to(self.device)
            candidate_np, candidate_mask_np = self._pad_group([g["candidate_points"] for g in part])
            candidate_neighbor_np, _ = self._pad_group([g["candidate_neighbor_mask"] for g in part])
            geometry_np, _ = self._pad_group([g["geometry"] for g in part])
            candidate = torch.from_numpy(candidate_np).to(self.device)
            candidate_neighbor_mask = torch.from_numpy(candidate_neighbor_np).to(self.device)
            geometry = torch.from_numpy(geometry_np).to(self.device)
            candidate_mask = torch.from_numpy(candidate_mask_np).to(self.device)
            output = self.model(
                history,
                history_mask,
                candidate,
                candidate_neighbor_mask,
                geometry,
                candidate_mask,
            )
            if self.score_mode in {"agrf_lite", "agrf_full", "rank_agrf"}:
                selected_np = self._agrf_scores(output, geometry, candidate_mask, part)
            elif self.score_mode == "safe_fixed":
                selected_np = self._safe_fixed_scores(output, geometry, candidate_mask, part)
            elif self.score_mode in {"expert_select", "expert_pool"}:
                selected_np = self._expert_fusion_scores(output, geometry, candidate_mask, part)
            elif self.score_mode in {"ambiguity_rerank_strict", "ambiguity_rerank", "ambiguity_rerank_loose"}:
                selected_np = self._ambiguity_rerank_scores(output, geometry, candidate_mask, part)
            else:
                selected = self._select_scores(output, geometry, candidate_mask).masked_fill(~candidate_mask, -1e6)
                selected_np = selected.detach().cpu().numpy().astype(np.float32)
            for row, group in zip(selected_np, part):
                group_scores.append(row[: len(group["candidate_indices"])].copy())
        return group_scores

    def _distance_prior_factor(self, distances: np.ndarray, gate: float) -> float:
        if self.distance_prior_mode == "fixed":
            return 1.0
        if self.distance_prior_mode != "adaptive":
            raise ValueError(f"unknown distance_prior_mode: {self.distance_prior_mode}")
        if len(distances) <= 1:
            return 1.0
        ordered = np.sort(distances.astype(np.float32))
        margin = float((ordered[1] - ordered[0]) / max(gate, 1e-6))
        low = float(self.adaptive_distance_margin_low)
        high = float(self.adaptive_distance_margin_high)
        if high <= low:
            scaled = 1.0 if margin >= high else 0.0
        else:
            scaled = float(np.clip((margin - low) / (high - low), 0.0, 1.0))
        min_factor = float(np.clip(self.adaptive_distance_min_factor, 0.0, 1.0))
        return min_factor + (1.0 - min_factor) * scaled

    @torch.no_grad()
    def _score_pairs(self, features: list[tuple]) -> np.ndarray:
        scores = []
        for start in range(0, len(features), self.inference_batch_size):
            part = features[start : start + self.inference_batch_size]
            history = torch.from_numpy(np.stack([p[0] for p in part])).to(self.device)
            history_mask = torch.from_numpy(np.stack([p[1] for p in part])).to(self.device)
            candidate = torch.from_numpy(np.stack([p[2] for p in part]))[:, None].to(self.device)
            candidate_mask = torch.from_numpy(np.stack([p[3] for p in part]))[:, None].to(self.device)
            geometry = torch.from_numpy(np.stack([p[4] for p in part]))[:, None].to(self.device)
            valid = torch.ones((len(part), 1), dtype=torch.bool, device=self.device)
            output = self.model(
                history,
                history_mask,
                candidate,
                candidate_mask,
                geometry,
                valid,
            )
            selected = self._select_scores(output, geometry, valid)[:, 0]
            scores.extend(selected.cpu().tolist())
        return np.asarray(scores, dtype=np.float32)

    def run(self, scene: SceneDetections) -> list[list[float]]:
        tracks: dict[int, Track] = {}
        next_id = 1
        emitted: list[tuple[int, list[float]]] = []
        if len(scene.frames) == 0:
            return []

        for frame in range(int(scene.frames.min()), int(scene.frames.max()) + 1):
            detections = scene.frame_to_indices.get(frame, np.zeros(0, dtype=np.int32))
            self._update_density_scale_prior(scene, detections)
            active = [track for track in tracks.values() if track.missing <= self.max_missing]
            score_groups = []
            for track_pos, track in enumerate(active):
                predicted = track.predict(scene, frame)
                last_position = scene.xy[track.last_index]
                gap = max(1, frame - int(scene.frames[track.last_index]))
                gate = track.dynamic_gate(
                    self.gate_radius,
                    frame,
                    scene,
                    max_radius=self.max_gate_radius,
                    missing_growth=self.missing_gate_growth,
                )
                if len(detections):
                    last_distances_all = np.linalg.norm(scene.xy[detections] - last_position[None], axis=1)
                    pred_distances_all = np.linalg.norm(scene.xy[detections] - predicted[None], axis=1)
                    if self.score_mode in {
                        "last_distance",
                        "ambiguity_rerank_strict",
                        "ambiguity_rerank",
                        "ambiguity_rerank_loose",
                    }:
                        distances = last_distances_all
                    elif self.score_mode in {
                        "agrf_lite",
                        "agrf_full",
                        "rank_agrf",
                        "safe_fixed",
                        "expert_select",
                        "expert_pool",
                    }:
                        distances = np.minimum(last_distances_all, pred_distances_all)
                    else:
                        distances = pred_distances_all
                    candidate_mask = distances <= gate
                    candidate_indices = detections[candidate_mask]
                    candidate_distances = distances[candidate_mask]
                    candidate_last_distances = last_distances_all[candidate_mask]
                    candidate_pred_distances = pred_distances_all[candidate_mask]
                    distance_prior_factor = self._distance_prior_factor(candidate_distances, gate)
                else:
                    candidate_indices = np.zeros(0, dtype=np.int32)
                    candidate_distances = np.zeros(0, dtype=np.float32)
                    candidate_last_distances = np.zeros(0, dtype=np.float32)
                    candidate_pred_distances = np.zeros(0, dtype=np.float32)
                    distance_prior_factor = 1.0
                if len(candidate_indices) == 0:
                    continue
                if self.score_mode == "last_distance":
                    score_groups.append(
                        {
                            "track_pos": track_pos,
                            "candidate_indices": [int(v) for v in candidate_indices],
                            "candidate_distances": [float(v) for v in candidate_distances],
                            "last_distances": [float(v) for v in candidate_last_distances],
                            "pred_distances": [float(v) for v in candidate_pred_distances],
                            "distance_prior_factor": float(distance_prior_factor),
                            "gate": float(gate),
                            "gap": int(gap),
                            "track_length": int(track.length),
                        }
                    )
                    continue
                history_points = None
                history_mask = None
                candidate_points = []
                candidate_neighbor_masks = []
                geometry = []
                for candidate_idx in candidate_indices:
                    pair = self._pair_features(scene, track, int(candidate_idx), frame)
                    if history_points is None:
                        history_points = pair[0]
                        history_mask = pair[1]
                    candidate_points.append(pair[2])
                    candidate_neighbor_masks.append(pair[3])
                    geometry.append(pair[4])
                score_groups.append(
                    {
                        "track_pos": track_pos,
                        "candidate_indices": [int(v) for v in candidate_indices],
                        "candidate_distances": [float(v) for v in candidate_distances],
                        "last_distances": [float(v) for v in candidate_last_distances],
                        "pred_distances": [float(v) for v in candidate_pred_distances],
                        "distance_prior_factor": float(distance_prior_factor),
                        "gate": float(gate),
                        "gap": int(gap),
                        "track_length": int(track.length),
                        "history_points": history_points,
                        "history_mask": history_mask,
                        "candidate_points": candidate_points,
                        "candidate_neighbor_mask": candidate_neighbor_masks,
                        "geometry": geometry,
                    }
                )

            score_matrix = np.full((len(active), len(detections)), -1e6, dtype=np.float32)
            det_slot = {int(det_idx): slot for slot, det_idx in enumerate(detections)}
            if score_groups:
                group_scores = self._score_groups(score_groups)
                for group, scores in zip(score_groups, group_scores):
                    track_pos = int(group["track_pos"])
                    track = active[track_pos]
                    for det_idx, distance, score in zip(
                        group["candidate_indices"],
                        group["candidate_distances"],
                        scores,
                    ):
                        distance_bonus = (
                            self.distance_score_weight
                            * float(group["distance_prior_factor"])
                            * (1.0 - min(1.0, distance / max(float(group["gate"]), 1e-6)))
                        )
                        missing_penalty = self.missing_score_penalty * max(0, track.missing)
                        score_matrix[track_pos, det_slot[det_idx]] = float(score) + distance_bonus - missing_penalty

            matched_tracks = set()
            matched_detections = set()
            if len(active) and len(detections):
                if self.assignment == "greedy":
                    candidates = [
                        (float(score_matrix[r, c]), r, c)
                        for r in range(score_matrix.shape[0])
                        for c in range(score_matrix.shape[1])
                        if score_matrix[r, c] > -1e5
                    ]
                    candidates.sort(reverse=True)
                    pairs = []
                    used_rows = set()
                    used_cols = set()
                    for score, r, c in candidates:
                        if score < self.score_threshold:
                            break
                        if r in used_rows or c in used_cols:
                            continue
                        used_rows.add(r)
                        used_cols.add(c)
                        pairs.append((r, c))
                elif self.assignment == "hungarian":
                    row, col = linear_sum_assignment(-score_matrix)
                    pairs = list(zip(row, col))
                else:
                    raise ValueError(f"unknown assignment: {self.assignment}")
                for r, c in pairs:
                    score = float(score_matrix[int(r), int(c)])
                    if score < self.score_threshold or score <= -1e5:
                        continue
                    track = active[int(r)]
                    detection_idx = int(detections[int(c)])
                    track.update(detection_idx)
                    matched_tracks.add(track.track_id)
                    matched_detections.add(detection_idx)
                    emitted.append((track.track_id, self._mot_row(scene, detection_idx, track.track_id, score)))

            for track in active:
                if track.track_id not in matched_tracks:
                    track.miss()

            for detection_idx in detections:
                detection_idx = int(detection_idx)
                if detection_idx in matched_detections:
                    continue
                track = Track(track_id=next_id, detection_indices=[detection_idx], confidence=0.5 + 0.1 * float(scene.scores[detection_idx]))
                tracks[next_id] = track
                emitted.append((next_id, self._mot_row(scene, detection_idx, next_id, float(scene.scores[detection_idx]))))
                next_id += 1

        lengths = {track_id: len(track.detection_indices) for track_id, track in tracks.items()}
        rows = [row for track_id, row in emitted if lengths.get(track_id, 0) >= max(self.min_track_length, self.mature_birth_length)]
        rows.sort(key=lambda row: (int(row[0]), int(row[1])))
        return rows

    @staticmethod
    def _mot_row(scene: SceneDetections, index: int, track_id: int, score: float) -> list[float]:
        x, y = scene.xy[index] - scene.wh[index] * 0.5
        w, h = scene.wh[index]
        return [
            int(scene.frames[index]),
            track_id,
            float(x),
            float(y),
            float(w),
            float(h),
            float(score),
            -1,
            -1,
            -1,
        ]


def attach_track_ids(rows_with_ids: Iterable[tuple[int, list[float]]]) -> list[list[float]]:
    output = []
    for track_id, row in rows_with_ids:
        copied = list(row)
        copied[1] = track_id
        output.append(copied)
    return output


def load_model(checkpoint_path: Path, device: str) -> CandidateGroupStructureNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {"grid_size": 24, "width": 24})
    if "num_scales" in config:
        model_config = {
            key: value
            for key, value in config.items()
            if key in {"grid_size", "width", "geometry_dim", "num_scales", "fusion_mode"}
        }
        model = MultiScaleCandidateGroupStructureNet(**model_config).to(device)
    else:
        model_config = {
            key: value
            for key, value in config.items()
            if key in {"grid_size", "width", "geometry_dim", "fusion_mode"}
        }
        model = CandidateGroupStructureNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def load_structure_radii(checkpoint_path: Path) -> tuple[float, ...] | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    radii = config.get("radii")
    if radii is None:
        return None
    return tuple(float(value) for value in radii)


def load_neighborhood_config(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    return {
        "structure_radii": tuple(float(v) for v in config["radii"]) if "radii" in config else None,
        "structure_radius": float(config.get("radius", 20.0)),
        "neighbor_mode": str(config.get("neighborhood", "radius")),
        "knn": tuple(int(v) for v in config["ks"]) if "ks" in config else None,
    }
