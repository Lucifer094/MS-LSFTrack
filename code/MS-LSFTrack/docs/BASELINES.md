# Baselines

The repository provides a common runner for BoxMOT trackers on the clean dataset layout.

## Supported Trackers

```text
bytetrack
ocsort
sfsort
strongsort
botsort
deepocsort
hybridsort
boosttrack
```

PuTR results can be evaluated after converting its predictions to the same MOT-style output format.

## Output Format

Each tracker should write one file per scene:

```text
frame,track_id,x,y,w,h,score,-1,-1,-1
```

Output path:

```text
${MS_LSF_OUTPUT_ROOT}/<dataset>/tracks/<det_version>/<tracker>/<split>/<scene>.txt
```

Runtime summaries are written to:

```text
${MS_LSF_OUTPUT_ROOT}/<dataset>/runtime/<det_version>/<tracker>/<split>_summary.csv
```

## Run Non-ReID Trackers

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers bytetrack ocsort sfsort
```

## Run ReID Trackers

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers strongsort botsort deepocsort hybridsort boosttrack \
  --reid-weights weights/reid/osnet_x0_25_msmt17.pt
```

Equivalent wrapper:

```bash
TRACKERS="strongsort botsort deepocsort hybridsort boosttrack" \
REID_WEIGHTS=weights/reid/osnet_x0_25_msmt17.pt \
bash scripts/run_baselines.sh configs/datasets/irdmstrack_v3_det_label.yaml test
```

## Evaluate

```bash
bash scripts/evaluate_all.sh configs/datasets/irdmstrack_v3_det_label.yaml test
```
