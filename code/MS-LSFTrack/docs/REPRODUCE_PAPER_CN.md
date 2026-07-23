# 论文实验复现说明

本文档说明如何复现当前公开版论文结果。公开版对应 paper-v1 frozen setting，不包含后续 dev 探索实验。

## 1. 环境变量

```bash
export MS_LSF_DATA_ROOT=/path/to/MS-LSFTrack-Datasets
export MS_LSF_OUTPUT_ROOT=/path/to/MS-LSFTrack-Outputs
export MS_LSF_RUN_ROOT=/path/to/MS-LSFTrack-Runs
export MS_LSF_CACHE_ROOT=/path/to/MS-LSFTrack-Cache
export MS_LSF_WEIGHT_ROOT=/path/to/MS-LSFTrack-Weights
```

公开代码包已经附带 `weights/reid/osnet_x0_25_msmt17.pt`，直接克隆代码仓库后即可用于 ReID baselines；如果想换成其他公开权重，再使用 `tools/prepare_reid_weights.py`。

## 2. 数据集配置

公开版提供三个数据集配置：

```text
configs/datasets/irdmstrack_v3_det_label.yaml
configs/datasets/gmot40_small_det_label.yaml
configs/datasets/irsatvideo_leo_resunet_rfr.yaml
```

数据集目录需要符合 `DATASET_FORMAT_CN.md` 中的 clean layout。

## 3. 使用公开 checkpoint 复现本文方法

IR-DMSTrack-v3：

```bash
bash scripts/paper_run_ms_lsftrack.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  /path/to/MS-LSFTrack-results/checkpoints/ms_lsf_ird_v3/best.pt \
  ablate_ird_v3_fusion_den101520_l05_asr_default_no_fusion_density \
  test

bash scripts/paper_evaluate_tracker.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  ablate_ird_v3_fusion_den101520_l05_asr_default_no_fusion_density \
  test
```

GMOT-40-small-target：

```bash
bash scripts/paper_run_ms_lsftrack.sh \
  configs/datasets/gmot40_small_det_label.yaml \
  /path/to/MS-LSFTrack-results/checkpoints/ms_lsf_gmot/best.pt \
  ablate_gmot_fusion_den101520_l05_asr_default_no_fusion_density \
  test

bash scripts/paper_evaluate_tracker.sh \
  configs/datasets/gmot40_small_det_label.yaml \
  ablate_gmot_fusion_den101520_l05_asr_default_no_fusion_density \
  test
```

IRSatVideo-LEO：

```bash
bash scripts/paper_run_ms_lsftrack.sh \
  configs/datasets/irsatvideo_leo_resunet_rfr.yaml \
  /path/to/MS-LSFTrack-results/checkpoints/ms_lsf_irsat/best.pt \
  ms_lsf_irsat_final_asr_no_density \
  test

bash scripts/paper_evaluate_tracker.sh \
  configs/datasets/irsatvideo_leo_resunet_rfr.yaml \
  ms_lsf_irsat_final_asr_no_density \
  test
```

## 4. 论文最终参数

`scripts/paper_run_ms_lsftrack.sh` 固定使用：

```text
score_mode = ambiguity_rerank
scale_mode = learned_density
structure_radii = 10,15,20
density_prior_direction = normal
density_prior_lambda = 0.5
asr_density_gain = 0.0
```

## 5. 从零训练

以 IR-DMSTrack-v3 为例：

```bash
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split train --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split val --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split test --output-dir "$MS_LSF_CACHE_ROOT"

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

训练完成后使用 `scripts/paper_run_ms_lsftrack.sh` 出轨迹并评价。

## 6. 对比方法

非 ReID 方法：

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers bytetrack ocsort sfsort
```

ReID 方法：

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers strongsort botsort deepocsort hybridsort boosttrack \
  --reid-weights weights/reid/osnet_x0_25_msmt17.pt
```

PuTR 结果需要单独运行 PuTR 后转换为同样的 MOT 结果格式，再使用 `tools/evaluate_tracks.py` 评价。

## 7. 结果表位置

公开结果包中的主结果表：

```text
MS-LSFTrack-results/summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_paper_table.csv
MS-LSFTrack-results/summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_comparison.md
```

消融表：

```text
MS-LSFTrack-results/summary/pre_module_ablation_main_20260706/
```

原始轨迹与评价结果：

```text
MS-LSFTrack-results/raw_outputs/
```
