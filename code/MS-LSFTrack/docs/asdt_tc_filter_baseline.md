# ASDT TC-Filtering Weak Baseline

This baseline adapts only the public `A-Simple-Detector-is-a-Strong-Tracker`
TC-Filtering idea to the clean-layout MOT detections used by MS-LSFTrack.

The released ASDT code provides data preparation, frame-dynamics data
generation, and TC-Filtering post-processing examples. It does not provide a
drop-in multi-object tracker or official multi-target ID association module for
the three MS-LSFTrack datasets. Therefore this script should not be described as
an official ASDT MOT reproduction in the paper.

For downstream debugging, `tools/run_asdt_tracker.py` keeps the reproducible part
that can be aligned across datasets: detection-score filtering, per-frame top-K
filtering, normalized motion gating, Hungarian online assignment, and short-gap
linear interpolation.

Recommended use:

- Use it as a weak heuristic baseline or ablation-style sanity check.
- Do not include it as a main 2025/2026 open-source MOT tracker comparison.
- For ID metrics such as HOTA, IDF1 and IDs, prefer trackers that natively output
  stable multi-object IDs.

## Inputs

The script reads the existing clean-layout detection files:

```text
${MS_LSF_DATA_ROOT}/<dataset>/detecion_label/<split>/<scene>.txt
```

Each row is interpreted as MOT-style detection data:

```text
frame,-1,x,y,w,h,score,-1,-1,-1
```

## Outputs

The output is aligned with all existing MS-LSFTrack evaluation tools:

```text
${MS_LSF_OUTPUT_ROOT}/<dataset>/tracks/<det_version>/<tracker>/<split>/<scene>.txt
${MS_LSF_OUTPUT_ROOT}/<dataset>/runtime/<det_version>/<tracker>/<split>_summary.csv
```

Track rows are written as:

```text
frame,track_id,x,y,w,h,score,-1,-1,-1
```

## AutoDL Batch Run

From the `MS-LSFTrack-Open` directory:

```bash
export MS_LSF_DATA_ROOT=/path/to/MS-LSFTrack-Datasets
export MS_LSF_OUTPUT_ROOT=/path/to/MS-LSFTrack-Outputs
bash scripts/autodl_run_asdt_tracker.sh
```

Useful overrides:

```bash
OVERWRITE=1 TRACKER_NAME=asdt_tc_filter bash scripts/autodl_run_asdt_tracker.sh
MAX_BBOXES_PER_FRAME=3 SCORE_THRESHOLD=0.05 bash scripts/autodl_run_asdt_tracker.sh
MAX_MOVEMENT=0.09 MAX_MISSING=3 INTERPOLATE_GAP=3 bash scripts/autodl_run_asdt_tracker.sh
```

## Single Dataset Run

```bash
python tools/run_asdt_tracker.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --tracker-name asdt_tc_filter \
  --overwrite
```

Run the same command with:

```text
configs/datasets/gmot40_small_det_label.yaml
configs/datasets/irsatvideo_leo_resunet_rfr.yaml
```

## Evaluation

After generating tracks, reuse the existing checks and metrics:

```bash
python tools/check_results.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers asdt_tc_filter

python tools/evaluate_tracks.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers asdt_tc_filter
```

The same tracker name can be passed into the paper-table comparison scripts.
