from __future__ import annotations
from typing import Dict, Iterable, Any


def _get(row: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        if k in row and row.get(k, '') not in ('', None):
            return row.get(k, '')
    return ''


def normalize_runtime_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize runtime summaries saved by different tracker scripts.

    BoxMOT clean scripts usually save fps_total/fps_update and total/update time.
    SMIA-v1 clean saves time_sec/fps.  The eval suite uses fps_update as the
    canonical FPS field for paper tables, so we map legacy `fps` to both
    fps_update and fps_total when the more specific fields are absent.
    """
    out = dict(row)

    fps_update = _get(row, 'fps_update', 'fps', 'avg_fps', 'FPS')
    fps_total = _get(row, 'fps_total', 'fps', 'avg_fps', 'FPS')
    total_time = _get(row, 'total_time_sec', 'time_sec', 'elapsed_sec')
    update_time = _get(row, 'update_time_sec', 'time_sec', 'elapsed_sec')
    total_frames = _get(row, 'total_frames', 'num_frames', 'frames')

    if fps_update != '': out['fps_update'] = fps_update
    if fps_total != '': out['fps_total'] = fps_total
    if total_time != '': out['total_time_sec'] = total_time
    if update_time != '': out['update_time_sec'] = update_time
    if total_frames != '': out['total_frames'] = total_frames

    # Keep the original fps too, useful for debugging legacy outputs.
    if 'fps' not in out and fps_update != '':
        out['fps'] = fps_update
    return out


def merge_runtime(summary: Dict[str, Any], runtime_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(runtime_rows)
    if not rows:
        return summary
    rt = normalize_runtime_row(rows[0])
    for key in ['fps_update', 'fps_total', 'fps', 'total_time_sec', 'update_time_sec', 'total_frames', 'total_dets', 'total_outputs']:
        if key in rt and rt.get(key, '') not in ('', None):
            summary[key] = rt.get(key, '')
    return summary
