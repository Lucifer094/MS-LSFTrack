from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_root: Path
    unified_root: Path
    det_version: str
    detection_root: Path
    gt_root: Path
    split_root: Path
    default_split: str = "test"
    point_dist_th: float = 3.0
    iou_th: float = 0.5
    notes: str = ""


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"Reading {path} requires PyYAML. Install pyyaml or use a JSON config."
        ) from exc
    data = yaml.safe_load(text)
    return data or {}


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_path(value: Any) -> Path:
    text = str(value)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return match.group(0)

    return Path(os.path.expanduser(_ENV_PATTERN.sub(replace, text)))


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = _load_mapping(Path(path))
    required = ["name", "data_root", "unified_root", "det_version"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"dataset config missing keys: {missing}")
    data_root = _expand_path(raw["data_root"])
    unified_root = _expand_path(raw["unified_root"])
    det_version = str(raw["det_version"])
    detection_root = _expand_path(
        raw.get("detection_root", data_root / "detecion_label")
    )
    gt_root = _expand_path(raw.get("gt_root", data_root))
    split_root = _expand_path(raw.get("split_root", data_root / "splits"))
    return DatasetConfig(
        name=str(raw["name"]),
        data_root=data_root,
        unified_root=unified_root,
        det_version=det_version,
        detection_root=detection_root,
        gt_root=gt_root,
        split_root=split_root,
        default_split=str(raw.get("default_split", "test")),
        point_dist_th=float(raw.get("point_dist_th", 3.0)),
        iou_th=float(raw.get("iou_th", 0.5)),
        notes=str(raw.get("notes", "")),
    )
