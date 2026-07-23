# MS-LSFTrack

MS-LSFTrack is an association-only multi-object tracking framework for dense small-target tracking. It uses fixed detector outputs and improves online track association with a density-guided multi-scale local structure field.

## Highlights

- Detection-agnostic online tracker for small-target MOT.
- Multi-scale local structure field with radii `10,15,20`.
- Density-guided scale weighting with `density_prior_lambda=0.5`.
- Ambiguity-aware motion-structure association.
- Unified evaluation for point-distance and IoU metrics.
- Scripts for MS-LSFTrack, BoxMOT baselines, training, evaluation and result aggregation.

## Repository Layout

```text
configs/        Dataset and experiment configs
mslsftrack/     Core dataset, model and tracker implementation
tools/          Python entry points for training, tracking and evaluation
scripts/        Shell wrappers for common reproduction commands
third_party/    Lightweight MOT evaluation and visualization utilities
docs/           Method and reproduction documentation
weights/        Public ReID weight for BoxMOT baselines
```

## Installation

```bash
conda create -n mslsftrack python=3.10 -y
conda activate mslsftrack
pip install -r requirements.txt
```

BoxMOT baselines require an additional compatible `boxmot` installation. A public ReID weight for ReID-based BoxMOT trackers is bundled at:

```text
weights/reid/osnet_x0_25_msmt17.pt
```

## Datasets

Datasets should use the clean association layout described in [DATASET_FORMAT.md](DATASET_FORMAT.md):

```text
DatasetRoot/
├── detecion_label/{train,val,test}/{scene}.txt
├── track_label/{scene}.txt
├── sequence/{scene}/...
└── splits/{train,val,test,all}.txt
```

Set the following paths before running experiments:

```bash
export MS_LSF_DATA_ROOT=/path/to/MS-LSFTrack-Datasets
export MS_LSF_OUTPUT_ROOT=/path/to/MS-LSFTrack-Outputs
export MS_LSF_RUN_ROOT=/path/to/MS-LSFTrack-Runs
export MS_LSF_CACHE_ROOT=/path/to/MS-LSFTrack-Cache
export MS_LSF_WEIGHT_ROOT=/path/to/MS-LSFTrack-Weights
```

Released dataset configs:

```text
configs/datasets/irdmstrack_v3_det_label.yaml
configs/datasets/gmot40_small_det_label.yaml
configs/datasets/irsatvideo_leo_resunet_rfr.yaml
```

## Run MS-LSFTrack

Use a released checkpoint from the companion results package:

```bash
bash scripts/run_mslsftrack.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  /path/to/MS-LSFTrack-results/checkpoints/ms_lsf_ird_v3/best.pt \
  ms_lsf_ird_v3 \
  test
```

Evaluate the generated tracks with both point and IoU metrics:

```bash
bash scripts/evaluate_tracker.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  ms_lsf_ird_v3 \
  test
```

The wrapper uses the released setting:

```text
score_mode = ambiguity_rerank
scale_mode = learned_density
structure_radii = 10,15,20
density_prior_direction = normal
density_prior_lambda = 0.5
asr_density_gain = 0.0
```

## Train From Scratch

Build association caches:

```bash
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split train --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split val --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split test --output-dir "$MS_LSF_CACHE_ROOT"
```

Train the multi-scale association model:

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

## Baselines

Run non-ReID BoxMOT trackers:

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers bytetrack ocsort sfsort
```

Run ReID-based BoxMOT trackers:

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers strongsort botsort deepocsort hybridsort boosttrack \
  --reid-weights weights/reid/osnet_x0_25_msmt17.pt
```

The comparison tables also include PuTR results after converting PuTR outputs to the same MOT text format.

## Results

The companion results package contains released checkpoints, final tracks, evaluation summaries and paper tables:

```text
MS-LSFTrack-results/
├── checkpoints/
├── raw_outputs/
├── summary/
├── configs/
└── manifests/
```

See [docs/RESULTS.md](docs/RESULTS.md) and the results package README for details.

## Documentation

- [docs/METHOD.md](docs/METHOD.md): model overview.
- [docs/REPRODUCE.md](docs/REPRODUCE.md): full reproduction commands.
- [docs/BASELINES.md](docs/BASELINES.md): baseline tracker commands.
- [docs/RESULTS.md](docs/RESULTS.md): result package layout.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
