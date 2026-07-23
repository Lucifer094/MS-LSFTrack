from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import numpy as np

from .io import read_detections, read_gt, read_predictions, list_scene_files
from .matching import match_point


def detection_quality(gt: np.ndarray, det: np.ndarray, dist_th: float = 3.0) -> Dict[str, float]:
    from .io import rows_by_frame
    gt_by_f = rows_by_frame(gt)
    dt_by_f = rows_by_frame(det)
    frames = sorted(set(gt_by_f.keys()) | set(dt_by_f.keys()))
    covered_gt = 0; total_gt = len(gt); matched_det = 0; total_det = len(det)
    for fr in frames:
        g = gt_by_f.get(fr, np.zeros((0,7)))
        d = dt_by_f.get(fr, np.zeros((0,7)))
        pairs, _, _ = match_point(g, d, dist_th)
        covered_gt += len(pairs)
        matched_det += len(pairs)
    fp_det = total_det - matched_det
    return {
        'num_gt': total_gt,
        'num_det': total_det,
        'matched_gt': covered_gt,
        'matched_det': matched_det,
        'det_recall': covered_gt / total_gt if total_gt else 0.0,
        'fp_det': fp_det,
        'fp_rate': fp_det / total_det if total_det else 0.0,
    }


def track_stats(pred: np.ndarray) -> Dict[str, float]:
    if pred.size == 0:
        return {'Pred_tracks': 0, 'Pred_dets': 0, 'AvgPredLen': 0.0, 'ShortTrackRate_3': 0.0, 'ShortTrackRate_5': 0.0, 'ShortTrackRate_10': 0.0}
    ids = pred[:,1].astype(int)
    lens = [int(np.sum(ids == i)) for i in np.unique(ids)]
    return {
        'Pred_tracks': len(lens),
        'Pred_dets': int(len(pred)),
        'AvgPredLen': float(np.mean(lens)) if lens else 0.0,
        'MedianPredLen': float(np.median(lens)) if lens else 0.0,
        'ShortTrackRate_3': float(np.mean([x <= 3 for x in lens])) if lens else 0.0,
        'ShortTrackRate_5': float(np.mean([x <= 5 for x in lens])) if lens else 0.0,
        'ShortTrackRate_10': float(np.mean([x <= 10 for x in lens])) if lens else 0.0,
    }
