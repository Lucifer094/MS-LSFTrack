# Reproduction

This document lists the commands needed to reproduce MS-LSFTrack tracking results and baseline evaluations.

## Environment Variables

```bash
export MS_LSF_DATA_ROOT=/path/to/MS-LSFTrack-Datasets
export MS_LSF_OUTPUT_ROOT=/path/to/MS-LSFTrack-Outputs
export MS_LSF_RUN_ROOT=/path/to/MS-LSFTrack-Runs
export MS_LSF_CACHE_ROOT=/path/to/MS-LSFTrack-Cache
export MS_LSF_WEIGHT_ROOT=/path/to/MS-LSFTrack-Weights
```

If `MS_LSF_WEIGHT_ROOT` is not set, scripts use `./weights`. The bundled ReID weight is available at `weights/reid/osnet_x0_25_msmt17.pt`.

## Dataset Configs

Download links and placement instructions are provided in [DATASETS.md](DATASETS.md).

```text
configs/datasets/irdmstrack_v3_det_label.yaml
configs/datasets/gmot40_small_det_label.yaml
configs/datasets/irsatvideo_leo_resunet_rfr.yaml
```

The dataset directory must follow [DATASET_FORMAT.md](../DATASET_FORMAT.md).

## Run Released Checkpoints

IR-DMSTrack-v3:

```bash
bash scripts/run_mslsftrack.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  /path/to/MS-LSFTrack-results/checkpoints/IR-DMSTrack-v3/best.pt \
  MS-LSFTrack \
  test

bash scripts/evaluate_tracker.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  MS-LSFTrack \
  test
```

GMOT-40-small-target:

```bash
bash scripts/run_mslsftrack.sh \
  configs/datasets/gmot40_small_det_label.yaml \
  /path/to/MS-LSFTrack-results/checkpoints/GMOT-40-small-target/best.pt \
  MS-LSFTrack \
  test

bash scripts/evaluate_tracker.sh \
  configs/datasets/gmot40_small_det_label.yaml \
  MS-LSFTrack \
  test
```

IRSatVideo-LEO:

```bash
bash scripts/run_mslsftrack.sh \
  configs/datasets/irsatvideo_leo_resunet_rfr.yaml \
  /path/to/MS-LSFTrack-results/checkpoints/IRSatVideo-LEO/best.pt \
  MS-LSFTrack \
  test

bash scripts/evaluate_tracker.sh \
  configs/datasets/irsatvideo_leo_resunet_rfr.yaml \
  MS-LSFTrack \
  test
```

## Train From Scratch

Example on IR-DMSTrack-v3:

```bash
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split train --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split val --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split test --output-dir "$MS_LSF_CACHE_ROOT"

python tools/train_assoc.py \
  --model multiscale_lsf \
  --neighbor-mode radius \
  --radii 10,15,20 \
  --train-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/train.pkl" \
  --val-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/val.pkl" \
  --test-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/test.pkl" \
  --output "$MS_LSF_RUN_ROOT/MS-LSFTrack" \
  --device cuda
```

After training, pass the generated `best.pt` to `scripts/run_mslsftrack.sh`.

## Run BoxMOT Baselines

Non-ReID trackers:

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers bytetrack ocsort sfsort
```

ReID trackers:

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers strongsort botsort deepocsort hybridsort boosttrack \
  --reid-weights weights/reid/osnet_x0_25_msmt17.pt
```

## Evaluate Existing Tracks

```bash
python tools/check_results.py --dataset <dataset.yaml> --split test --trackers <tracker>
python tools/evaluate_tracks.py --dataset <dataset.yaml> --split test --mode point --trackers <tracker>
python tools/evaluate_tracks.py --dataset <dataset.yaml> --split test --mode iou --trackers <tracker>
python tools/compare_trackers.py --dataset <dataset.yaml> --split test --trackers <tracker>
```
