# MS-LSFTrack

MS-LSFTrack is an association-only multi-object tracking framework for dense small-target tracking. It keeps detector outputs fixed and focuses on learning local spatial structure cues for online track association.

This public version matches the paper-v1 frozen setting:

```text
tail-anchor ASR
+ tail-node local structure
+ density-guided multi-scale local structure field
```

The repository contains training, online tracking, baseline running, quantitative evaluation, qualitative visualization helpers, and data/TS-AMID diagnostics. It does not include detector training code or the full datasets.
The public release also bundles a public ReID weight under `weights/reid/` for BoxMOT baselines.

## Paper Model

The released paper setting is:

```text
Structure branch:
  structure_radii = 10,15,20
  scale_mode = learned_density
  density_prior_direction = normal
  density_prior_lambda = 0.5

Motion-structure fusion:
  score_mode = ambiguity_rerank
  asr_density_gain = 0.0

ASR anchor:
  last observation / track tail

Structure history:
  tail-node local structure
```

Density is used only to guide multi-scale structure fusion. It is not used as an extra gain in the final motion-structure ASR fusion.

## Repository Layout

```text
configs/                 Dataset and experiment configs
mslsftrack/              Core dataset/model/tracker implementation
tools/                   Python entry points for training, tracking, evaluation, diagnostics
scripts/                 Shell wrappers for paper reproduction and common workflows
third_party/             Unified MOT evaluation and visualization utilities
docs/                    Reproduction, dataset, code/result mapping documentation
weights/                 Bundled public weights for baseline reproduction
DATASET_FORMAT_CN.md     Clean-layout dataset format
requirements.txt         Python dependencies
```

The most important paper-reproduction files are:

```text
scripts/paper_run_ms_lsftrack.sh
scripts/paper_evaluate_tracker.sh
scripts/paper_run_boxmot_baselines.sh
tools/build_cache.py
tools/train_assoc.py
tools/run_tracker.py
tools/evaluate_tracks.py
tools/run_baseline_trackers.py
```

See [docs/CODE_AND_RESULTS_MAP_CN.md](docs/CODE_AND_RESULTS_MAP_CN.md) for a full code/result map.

## Installation

```bash
conda create -n mslsftrack python=3.10 -y
conda activate mslsftrack
pip install -r requirements.txt
```

If you want to reproduce BoxMOT baselines, install a compatible `boxmot` version separately. The release package already includes `weights/reid/osnet_x0_25_msmt17.pt` for ReID-based trackers.

## Dataset Layout

Datasets should follow the clean association-only layout:

```text
DatasetRoot/
├── detecion_label/{train,val,test}/{scene}.txt
├── track_label/{scene}.txt
├── sequence/{scene}/...
└── splits/{train,val,test,all}.txt
```

The spelling `detecion_label` follows the current prepared datasets. See [DATASET_FORMAT_CN.md](DATASET_FORMAT_CN.md) for details.

Set paths with environment variables:

```bash
export MS_LSF_DATA_ROOT=/path/to/MS-LSFTrack-Datasets
export MS_LSF_OUTPUT_ROOT=/path/to/MS-LSFTrack-Outputs
export MS_LSF_RUN_ROOT=/path/to/MS-LSFTrack-Runs
export MS_LSF_CACHE_ROOT=/path/to/MS-LSFTrack-Cache
export MS_LSF_WEIGHT_ROOT=/path/to/MS-LSFTrack-Weights
```

## Run the Paper Tracker

With released checkpoints:

```bash
bash scripts/paper_run_ms_lsftrack.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  /path/to/MS-LSFTrack-results/checkpoints/ms_lsf_ird_v3/best.pt \
  ms_lsf_ird_v3_paper \
  test
```

Evaluate both point and IoU metrics:

```bash
bash scripts/paper_evaluate_tracker.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  ms_lsf_ird_v3_paper \
  test
```

The wrapper expands to the paper setting:

```bash
python tools/run_tracker.py \
  --score-mode ambiguity_rerank \
  --scale-mode learned_density \
  --structure-radii-override 10,15,20 \
  --density-prior-direction normal \
  --density-prior-lambda 0.5 \
  --asr-density-gain 0.0
```

## Train From Scratch

Build train/val/test association caches:

```bash
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split train --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split val --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split test --output-dir "$MS_LSF_CACHE_ROOT"
```

Train the multi-scale LSF association model:

```bash
python tools/train_assoc.py \
  --model multiscale_lsf \
  --neighbor-mode radius \
  --radii 10,15,20 \
  --train-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/train.pkl" \
  --val-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/val.pkl" \
  --test-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/test.pkl" \
  --output "$MS_LSF_RUN_ROOT/ms_lsf_ird_v3" \
  --device cuda
```

Run online tracking with the trained checkpoint using `scripts/paper_run_ms_lsftrack.sh`.

## Baselines

Run non-ReID BoxMOT baselines:

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers bytetrack ocsort sfsort
```

Run ReID-based baselines:

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers strongsort botsort deepocsort hybridsort boosttrack \
  --reid-weights weights/reid/osnet_x0_25_msmt17.pt
```

The paper comparison also includes PuTR results if you provide PuTR predictions in the same MOT output format.

## Released Results

The companion result package is organized as:

```text
MS-LSFTrack-results/
├── summary/       Paper tables and ablations
├── raw_outputs/   Final tracks, point/IoU eval summaries, runtime summaries
├── checkpoints/   Released MS-LSFTrack checkpoints
├── configs/       Dataset/experiment configs used for the frozen run
└── manifests/     Freeze manifest
```

See the result package `README_CN.md` for exact tracker names and table paths.

## Notes

- This repository is the paper-v1 public code. Development experiments about predicted-position ASR and trajectory-level structure memory are intentionally excluded.
- The project focuses on association; detector training is outside this repository.
- Before public upload, choose and add an explicit open-source license.
