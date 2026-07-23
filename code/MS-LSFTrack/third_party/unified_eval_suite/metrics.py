from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Tuple
import math
import numpy as np

from .io import rows_by_frame
from .matching import match_point, match_iou


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r) if p + r else 0.0


def _global_id_assignment(co: Dict[Tuple[int, int], int]) -> int:
    if not co:
        return 0
    gt_ids = sorted({g for g, _ in co})
    pr_ids = sorted({p for _, p in co})
    gmap = {g: i for i, g in enumerate(gt_ids)}
    pmap = {p: i for i, p in enumerate(pr_ids)}
    mat = np.zeros((len(gt_ids), len(pr_ids)), dtype=float)
    for (g, p), v in co.items():
        mat[gmap[g], pmap[p]] = v
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        r, c = linear_sum_assignment(-mat)
        return int(mat[r, c].sum())
    except Exception:
        # greedy fallback
        total = 0
        used_g, used_p = set(), set()
        vals = [(mat[i, j], i, j) for i in range(mat.shape[0]) for j in range(mat.shape[1])]
        for v, i, j in sorted(vals, reverse=True):
            if i not in used_g and j not in used_p:
                total += int(v); used_g.add(i); used_p.add(j)
        return total


def evaluate(gt: np.ndarray, pred: np.ndarray, mode: str = 'point', dist_th: float = 3.0, iou_th: float = 0.5) -> Dict[str, float]:
    gt_by_f = rows_by_frame(gt)
    pr_by_f = rows_by_frame(pred)
    frames = sorted(set(gt_by_f.keys()) | set(pr_by_f.keys()))
    matcher: Callable
    if mode == 'point':
        matcher = lambda g, p: match_point(g, p, dist_th)
    elif mode == 'iou':
        matcher = lambda g, p: match_iou(g, p, iou_th)
    else:
        raise ValueError(f'unknown mode: {mode}')

    total_gt = int(len(gt))
    total_pred = int(len(pred))
    tp = fp = fn = idsw = 0
    match_quality_sum = 0.0
    match_quality_count = 0
    co: Dict[Tuple[int, int], int] = defaultdict(int)
    gt_match_flags: Dict[int, Dict[int, bool]] = defaultdict(dict)
    prev_match: Dict[int, int] = {}
    matched_gt_frames: Dict[int, int] = defaultdict(int)
    gt_total_frames: Dict[int, int] = defaultdict(int)
    pred_total_frames: Dict[int, int] = defaultdict(int)

    for r in gt:
        gt_total_frames[int(round(r[1]))] += 1
    for r in pred:
        pred_total_frames[int(round(r[1]))] += 1

    for fr in frames:
        g = gt_by_f.get(fr, np.zeros((0, 7)))
        p = pr_by_f.get(fr, np.zeros((0, 7)))
        pairs, ug, up = matcher(g, p)
        tp += len(pairs); fn += len(ug); fp += len(up)
        matched_gids_this_frame = set()
        for gi, pi, q in pairs:
            gid = int(round(g[gi, 1])); pid = int(round(p[pi, 1]))
            co[(gid, pid)] += 1
            matched_gt_frames[gid] += 1
            matched_gids_this_frame.add(gid)
            if gid in prev_match and prev_match[gid] != pid:
                idsw += 1
            prev_match[gid] = pid
            match_quality_sum += q
            match_quality_count += 1
        for r in g:
            gid = int(round(r[1]))
            gt_match_flags[gid][fr] = gid in matched_gids_this_frame

    # Fragmentation: matched -> unmatched gap -> matched
    frag = 0
    for gid, flags in gt_match_flags.items():
        state = 'start'
        for fr in sorted(flags):
            matched = flags[fr]
            if state == 'start':
                state = 'matched' if matched else 'unmatched'
            elif state == 'matched' and not matched:
                state = 'lost'
            elif state == 'lost' and matched:
                frag += 1
                state = 'matched'
            elif state == 'unmatched' and matched:
                state = 'matched'

    detp = _safe_div(tp, tp + fp)
    detr = _safe_div(tp, tp + fn)
    detf1 = _f1(detp, detr)
    deta = _safe_div(tp, tp + fp + fn)
    mota = 1.0 - _safe_div(fn + fp + idsw, total_gt)

    idtp = _global_id_assignment(co)
    idfp = total_pred - idtp
    idfn = total_gt - idtp
    idp = _safe_div(idtp, idtp + idfp)
    idr = _safe_div(idtp, idtp + idfn)
    idf1 = _f1(idp, idr)

    # Association HOTA-style approximation
    ass_vals = []
    assr_vals = []
    gt_match_total = defaultdict(int)
    pr_match_total = defaultdict(int)
    for (g, p), c in co.items():
        gt_match_total[g] += c
        pr_match_total[p] += c
    for (g, p), c in co.items():
        tpa = c
        fna = gt_match_total[g] - c
        fpa = pr_match_total[p] - c
        val = _safe_div(tpa, tpa + fna + fpa)
        recall_val = _safe_div(tpa, tpa + fna)
        ass_vals += [val] * c
        assr_vals += [recall_val] * c
    assa = float(np.mean(ass_vals)) if ass_vals else 0.0
    assr = float(np.mean(assr_vals)) if assr_vals else 0.0
    hota = math.sqrt(max(0.0, deta * assa))

    # Track coverage
    mt = pt_count = ml = 0
    for gid, total in gt_total_frames.items():
        ratio = _safe_div(matched_gt_frames.get(gid, 0), total)
        if ratio >= 0.8:
            mt += 1
        elif ratio > 0.2:
            pt_count += 1
        else:
            ml += 1

    pred_tracks = len(pred_total_frames)
    gt_tracks = len(gt_total_frames)
    avg_pred_len = float(np.mean(list(pred_total_frames.values()))) if pred_total_frames else 0.0
    avg_gt_len = float(np.mean(list(gt_total_frames.values()))) if gt_total_frames else 0.0

    pred_purities = []
    for pid, plen in pred_total_frames.items():
        best = max([c for (g, p), c in co.items() if p == pid] or [0])
        pred_purities.append(_safe_div(best, plen))
    pred_purity = float(np.mean(pred_purities)) if pred_purities else 0.0
    pure_pred_rate = float(np.mean([x >= 0.8 for x in pred_purities])) if pred_purities else 0.0

    motp = _safe_div(match_quality_sum, match_quality_count)
    return {
        'TP': tp, 'FP': fp, 'FN': fn,
        'IDs': idsw, 'IDSW': idsw, 'Frag': frag,
        'DetP': detp, 'DetR': detr, 'DetF1': detf1, 'DetA': deta,
        'MOTA': mota,
        'IDTP': idtp, 'IDFP': idfp, 'IDFN': idfn,
        'IDP': idp, 'IDR': idr, 'IDF1': idf1,
        'AssA': assa, 'AssR': assr, 'HOTA': hota,
        'MOTP': motp,
        'MT': mt, 'PT': pt_count, 'ML': ml,
        'Pred_tracks': pred_tracks, 'GT_tracks': gt_tracks,
        'AvgPredLen': avg_pred_len, 'AvgGTLen': avg_gt_len,
        'PredPurity': pred_purity, 'PurePredRate': pure_pred_rate,
        'num_gt_dets': total_gt, 'num_pred_dets': total_pred,
    }


def prefix_metrics(m: Dict[str, float], prefix: str, motp_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in m.items():
        if k in {'TP', 'FP', 'FN', 'IDs', 'IDSW', 'Frag', 'MT', 'PT', 'ML', 'Pred_tracks', 'GT_tracks', 'AvgPredLen', 'AvgGTLen', 'PredPurity', 'PurePredRate', 'num_gt_dets', 'num_pred_dets'}:
            if k in {'TP', 'FP', 'FN', 'IDs', 'IDSW', 'Frag'}:
                out[f'{prefix}-{k if k != "IDSW" else "IDs"}'] = v
            else:
                out[k] = v
        elif k == 'MOTP':
            out[motp_name] = v
        else:
            out[f'{prefix}-{k}'] = v
    return out
