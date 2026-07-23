# MS-LSFTrack 论文实验运行手册

本手册对应公开版论文 v1。推荐优先阅读 `docs/REPRODUCE_PAPER_CN.md`，本文件保留更具体的分步命令。

## 1. 环境变量

```bash
export MS_LSF_DATA_ROOT=/path/to/MS-LSFTrack-Datasets
export MS_LSF_OUTPUT_ROOT=/path/to/MS-LSFTrack-Outputs
export MS_LSF_RUN_ROOT=/path/to/MS-LSFTrack-Runs
export MS_LSF_CACHE_ROOT=/path/to/MS-LSFTrack-Cache
export MS_LSF_WEIGHT_ROOT=/path/to/MS-LSFTrack-Weights
```

公开代码包里已经包含 `weights/reid/osnet_x0_25_msmt17.pt`，因此 ReID baselines 可以直接使用仓库内的公开权重。

## 2. 最终模型参数

```text
structure_radii = 10,15,20
scale_mode = learned_density
density_prior_direction = normal
density_prior_lambda = 0.5
score_mode = ambiguity_rerank
asr_density_gain = 0.0
```

推荐使用脚本：

```bash
bash scripts/paper_run_ms_lsftrack.sh DATASET_CONFIG CHECKPOINT TRACKER_NAME test
bash scripts/paper_evaluate_tracker.sh DATASET_CONFIG TRACKER_NAME test
```

## 3. 构建缓存

```bash
python tools/build_cache.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split train \
  --output-dir "$MS_LSF_CACHE_ROOT"
```

`val` 和 `test` 同理。

## 4. 训练

```bash
python tools/train_assoc.py \
  --model multiscale_lsf \
  --fusion-mode residual \
  --neighbor-mode radius \
  --radii 10,15,20 \
  --train-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/train.pkl" \
  --val-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/val.pkl" \
  --test-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/test.pkl" \
  --output "$MS_LSF_RUN_ROOT/ms_lsf_ird_v3" \
  --device cuda
```

## 5. 在线跟踪

```bash
bash scripts/paper_run_ms_lsftrack.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  "$MS_LSF_RUN_ROOT/ms_lsf_ird_v3/best.pt" \
  ms_lsf_ird_v3_paper \
  test
```

## 6. 评价

```bash
bash scripts/paper_evaluate_tracker.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  ms_lsf_ird_v3_paper \
  test
```

评价结果会写入：

```text
${MS_LSF_OUTPUT_ROOT}/{Dataset}/eval/point_3px/{det_version}/{tracker}/test_summary.csv
${MS_LSF_OUTPUT_ROOT}/{Dataset}/eval/iou_0.5/{det_version}/{tracker}/test_summary.csv
```

## 7. 对比算法

非 ReID baseline：

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers bytetrack ocsort sfsort
```

ReID baseline：

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers strongsort botsort deepocsort hybridsort boosttrack \
  --reid-weights weights/reid/osnet_x0_25_msmt17.pt
```

## 8. 消融

公开结果包已经包含论文主线消融结果：

```text
MS-LSFTrack-results/summary/pre_module_ablation_main_20260706/
```

若需要重新跑消融，请使用 `tools/run_tracking_ablation.py` 和 `configs/experiments/ablation_switches.yaml`，并确保所有结果使用 `test` split 出轨迹和评价。
