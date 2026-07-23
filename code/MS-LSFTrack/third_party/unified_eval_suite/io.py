from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def scene_name_from_path(p: str | Path) -> str:
    return Path(p).stem


def read_split(data_root: str | Path, split: str) -> List[str]:
    root = Path(data_root)
    candidates = [
        root / 'splits' / f'{split}.txt',
        root / 'splits' / split,
        root / f'{split}.txt',
    ]
    for p in candidates:
        if p.exists():
            scenes: List[str] = []
            for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
                s = line.strip().split()[0] if line.strip() else ''
                if s and not s.startswith('#'):
                    scenes.append(s)
            return scenes
    # Fallback: infer from directories named like 00001
    scenes = sorted([x.name for x in root.iterdir() if x.is_dir() and x.name.isdigit()]) if root.exists() else []
    return scenes


def find_image_dir(data_root: str | Path, scene: str, split: Optional[str] = None) -> Optional[Path]:
    root = Path(data_root)
    candidates: List[Path] = []
    if split:
        candidates += [root / split / scene / 'img1', root / split / scene / 'images']
    candidates += [root / scene / 'img1', root / scene / 'images', root / 'images' / scene, root / split / scene if split else root / scene]
    for d in candidates:
        if d and d.exists() and d.is_dir():
            imgs = [x for x in d.iterdir() if x.suffix.lower() in IMG_EXTS]
            if imgs:
                return d
    return None


def list_images(img_dir: str | Path) -> List[Path]:
    p = Path(img_dir)
    if not p.exists():
        return []
    return sorted([x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS])


def read_mot_txt(path: str | Path, kind: str = 'pred') -> np.ndarray:
    """Read MOT-like txt.

    Returns float array with columns [frame,id,x,y,w,h,score].
    Supports:
      - 10/9/7 cols: frame,id,x,y,w,h,score,...
      - 6 cols detection: frame,x,y,w,h,score; id=-1
    """
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 7), dtype=float)
    rows: List[List[float]] = []
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [x.strip() for x in line.replace(' ', ',').split(',') if x.strip() != '']
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            if len(vals) >= 7:
                frame, oid, x, y, w, h, score = vals[:7]
            elif len(vals) == 6:
                frame, x, y, w, h, score = vals
                oid = -1.0
            else:
                continue
            rows.append([frame, oid, x, y, w, h, score])
    if not rows:
        return np.zeros((0, 7), dtype=float)
    arr = np.asarray(rows, dtype=float)
    # Sort by frame then id
    idx = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[idx]


def write_mot_txt(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for r in arr:
            frame, oid, x, y, w, h, score = r[:7]
            f.write(f'{int(round(frame))},{int(round(oid))},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.6f},-1,-1,-1\n')


def boxes_to_centers(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    return np.stack([arr[:, 2] + arr[:, 4] / 2.0, arr[:, 3] + arr[:, 5] / 2.0], axis=1)


def rows_by_frame(arr: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if arr.size == 0:
        return out
    frames = np.unique(arr[:, 0].astype(int))
    for fr in frames:
        out[int(fr)] = arr[arr[:, 0].astype(int) == int(fr)]
    return out


def count_frames(arrs: Sequence[np.ndarray]) -> int:
    max_fr = 0
    for arr in arrs:
        if arr.size:
            max_fr = max(max_fr, int(np.max(arr[:, 0])))
    return max_fr


def candidate_gt_paths(data_root: str | Path, scene: str, split: Optional[str] = None) -> List[Path]:
    root = Path(data_root)
    paths: List[Path] = []
    if split:
        paths += [
            root / split / scene / 'gt' / 'gt.txt',
            root / split / scene / 'gt.txt',
            root / 'gt' / split / f'{scene}.txt',
            root / 'annotations' / split / f'{scene}.txt',
            root / 'labels' / split / f'{scene}.txt',
            root / 'track_label' / split / f'{scene}.txt',
        ]
    paths += [
        root / scene / 'gt' / 'gt.txt',
        root / scene / 'gt.txt',
        root / 'gt' / f'{scene}.txt',
        root / 'annotations' / f'{scene}.txt',
        root / 'labels' / f'{scene}.txt',
        root / 'track_label' / f'{scene}.txt',
        root / 'IR-DMSTrack_MOT' / scene / 'gt' / 'gt.txt',
    ]
    return paths


def find_gt_file(data_root: str | Path, scene: str, split: Optional[str] = None) -> Optional[Path]:
    for p in candidate_gt_paths(data_root, scene, split):
        if p.exists():
            return p
    return None


def read_gt(data_root: str | Path, scene: str, split: Optional[str] = None) -> np.ndarray:
    p = find_gt_file(data_root, scene, split)
    if p is None:
        return np.zeros((0, 7), dtype=float)
    arr = read_mot_txt(p, kind='gt')
    # If GT score is zero/missing in some MOT files, force score=1 for valid boxes
    if arr.size:
        arr[:, 6] = np.where(arr[:, 6] <= 0, 1.0, arr[:, 6])
    return arr


def unified_det_path(unified_root: str | Path, det_version: str, split: str, scene: str) -> Path:
    return Path(unified_root) / 'detections' / det_version / split / f'{scene}.txt'


def unified_track_path(unified_root: str | Path, det_version: str, tracker: str, split: str, scene: str) -> Path:
    return Path(unified_root) / 'tracks' / det_version / tracker / split / f'{scene}.txt'


def read_detections(unified_root: str | Path, det_version: str, split: str, scene: str) -> np.ndarray:
    return read_mot_txt(unified_det_path(unified_root, det_version, split, scene), kind='det')


def read_predictions(unified_root: str | Path, det_version: str, tracker: str, split: str, scene: str) -> np.ndarray:
    return read_mot_txt(unified_track_path(unified_root, det_version, tracker, split, scene), kind='pred')


def list_scene_files(folder: str | Path) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([x.stem for x in p.glob('*.txt') if not x.name.endswith('.partial')])


def read_csv_dicts(path: str | Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: str | Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
