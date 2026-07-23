from __future__ import annotations

from typing import List, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def iou_matrix(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    if gt.size == 0 or pr.size == 0:
        return np.zeros((len(gt), len(pr)), dtype=float)
    gx1, gy1 = gt[:, 2], gt[:, 3]
    gx2, gy2 = gt[:, 2] + gt[:, 4], gt[:, 3] + gt[:, 5]
    px1, py1 = pr[:, 2], pr[:, 3]
    px2, py2 = pr[:, 2] + pr[:, 4], pr[:, 3] + pr[:, 5]
    inter_x1 = np.maximum(gx1[:, None], px1[None, :])
    inter_y1 = np.maximum(gy1[:, None], py1[None, :])
    inter_x2 = np.minimum(gx2[:, None], px2[None, :])
    inter_y2 = np.minimum(gy2[:, None], py2[None, :])
    iw = np.maximum(0.0, inter_x2 - inter_x1)
    ih = np.maximum(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    garea = np.maximum(0.0, gt[:, 4] * gt[:, 5])
    parea = np.maximum(0.0, pr[:, 4] * pr[:, 5])
    union = garea[:, None] + parea[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def center_distance_matrix(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    if gt.size == 0 or pr.size == 0:
        return np.zeros((len(gt), len(pr)), dtype=float)
    gc = np.stack([gt[:, 2] + gt[:, 4] / 2.0, gt[:, 3] + gt[:, 5] / 2.0], axis=1)
    pc = np.stack([pr[:, 2] + pr[:, 4] / 2.0, pr[:, 3] + pr[:, 5] / 2.0], axis=1)
    diff = gc[:, None, :] - pc[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    # Greedy fallback
    pairs: List[Tuple[int, int]] = []
    used_r, used_c = set(), set()
    flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    for _, i, j in sorted(flat, key=lambda x: x[0]):
        if i not in used_r and j not in used_c:
            pairs.append((i, j)); used_r.add(i); used_c.add(j)
    return pairs


def match_point(gt: np.ndarray, pr: np.ndarray, dist_th: float = 3.0) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if len(gt) == 0:
        return [], [], list(range(len(pr)))
    if len(pr) == 0:
        return [], list(range(len(gt))), []
    dist = center_distance_matrix(gt, pr)
    big = 1e6
    cost = dist.copy()
    cost[dist > dist_th] = big
    pairs = []
    for i, j in _hungarian(cost):
        if dist[i, j] <= dist_th:
            pairs.append((i, j, float(dist[i, j])))
    mg = {i for i, _, _ in pairs}
    mp = {j for _, j, _ in pairs}
    return pairs, [i for i in range(len(gt)) if i not in mg], [j for j in range(len(pr)) if j not in mp]


def match_iou(gt: np.ndarray, pr: np.ndarray, iou_th: float = 0.5) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if len(gt) == 0:
        return [], [], list(range(len(pr)))
    if len(pr) == 0:
        return [], list(range(len(gt))), []
    iou = iou_matrix(gt, pr)
    big = 1e6
    cost = 1.0 - iou
    cost[iou < iou_th] = big
    pairs = []
    for i, j in _hungarian(cost):
        if iou[i, j] >= iou_th:
            pairs.append((i, j, float(iou[i, j])))
    mg = {i for i, _, _ in pairs}
    mp = {j for _, j, _ in pairs}
    return pairs, [i for i in range(len(gt)) if i not in mg], [j for j in range(len(pr)) if j not in mp]
