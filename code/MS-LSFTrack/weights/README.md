# Public Weights

This directory stores the model weights needed for public reproduction. The code repository only bundles the public ReID weight that is directly required by the BoxMOT baselines; the MS-LSFTrack checkpoints trained for the paper are released in the companion results package under `MS-LSFTrack-results/checkpoints/`.

## ReID baseline weight

```text
weights/reid/osnet_x0_25_msmt17.pt
```

This weight can be used for the BoxMOT trackers that rely on ReID embeddings:

```text
StrongSORT
BoT-SORT
DeepOCSORT
HybridSORT
BoostTrack
```

Example:

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers strongsort botsort deepocsort hybridsort boosttrack \
  --reid-weights weights/reid/osnet_x0_25_msmt17.pt
```

If you want to prepare a different ReID weight, use:

```bash
bash scripts/autodl_prepare_reid.sh <model_name_or_path>
```

or place the new checkpoint directly under:

```text
weights/reid/
```
