from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PUBLIC_REID_MODEL_NAMES = {
    "osnet_x0_25_msmt17.pt",
    "osnet_x0_5_msmt17.pt",
    "osnet_x1_0_msmt17.pt",
    "lmbn_n_duke.pt",
    "clip_market1501.pt",
}


def instantiate_reid_backend(weights: Path, device: str) -> None:
    try:
        from boxmot.reid.core.auto_backend import ReidAutoBackend
    except Exception as exc:
        raise ImportError("boxmot is required to prepare public ReID weights") from exc

    # BoxMOT resolves public model names by the checkpoint filename through its
    # ReID registry, so using output-dir/model_name keeps the cache explicit.
    ReidAutoBackend(weights=weights, device=device, half=False)


def copy_if_needed(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return dst
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ReID weights for BoxMOT baselines.")
    parser.add_argument("--model", default="osnet_x0_25_msmt17.pt", help="Public BoxMOT model name or local checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("MS_LSF_WEIGHT_ROOT", "./weights")) / "reid")
    parser.add_argument("--device", default="0")
    parser.add_argument("--copy-local", action="store_true", help="Copy a local checkpoint into output-dir instead of relying on BoxMOT cache.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    if model_path.is_file():
        target = copy_if_needed(model_path, args.output_dir / model_path.name) if args.copy_local else model_path
        print(f"[DONE] local ReID weights={target}")
        return

    if args.model not in PUBLIC_REID_MODEL_NAMES:
        print(f"[WARN] {args.model} is not in the known public BoxMOT ReID model list.")
        print("[WARN] BoxMOT may still resolve it if your installed version has a matching registry entry.")

    target = args.output_dir / args.model
    instantiate_reid_backend(target, args.device)
    if not target.is_file():
        raise FileNotFoundError(f"BoxMOT did not create the expected ReID weights file: {target}")

    marker = args.output_dir / "PUBLIC_REID_MODEL.txt"
    marker.write_text(
        f"model={args.model}\n"
        f"weights={target}\n"
        "This model was resolved through the installed BoxMOT ReID registry.\n",
        encoding="utf-8",
    )
    print(f"[DONE] public ReID model resolved by BoxMOT: {target}")
    print(f"[INFO] For baseline scripts you may use REID_WEIGHTS={target}")


if __name__ == "__main__":
    main()
