from __future__ import annotations
from pathlib import Path
from typing import List, Dict

from .io import read_csv_dicts, write_csv_dicts, ensure_dir


def load_summary(path: str | Path) -> Dict:
    rows = read_csv_dicts(path)
    return rows[0] if rows else {}


def merge_summaries(paths: List[Path], out_path: Path) -> None:
    rows = []
    keys = []
    for p in paths:
        r = load_summary(p)
        if r:
            rows.append(r)
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
    write_csv_dicts(out_path, rows, keys)
