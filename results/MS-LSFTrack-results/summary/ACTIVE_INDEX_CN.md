# 当前公开结果索引

更新时间：2026-07-17

本结果包只保留当前论文 v1 使用的公开口径。IR-DMSTrack 数据集统一使用 `IR-DMSTrack-v3`。

## 当前最终模型口径

```text
Structure branch:
  structure_radii = 10,15,20
  scale_mode = learned_density
  density_direction = normal
  density_lambda = 0.5

Motion-structure fusion:
  score_mode = ambiguity_rerank
  asr_density_gain = 0.0
```

含义：

- density 只用于结构分支内部的多尺度融合。
- motion 与 structure 的最终融合使用 ASR-default，不再额外加入 fusion density gain。
- 消融、总对比、论文表格均按 IR-DMSTrack-v3 口径整理。

## 主线消融目录

```text
summary/pre_module_ablation_main_20260706
```

主要文件：

| File | 用途 |
| --- | --- |
| `01_single_scale_structure_only_v3_gmot.csv` | Structure Only 单尺度消融，说明不同数据集需要的结构尺度不同 |
| `02_structure_only_multiscale_density_v3_gmot.csv` | Structure Only 多尺度与 density 引导消融 |
| `03_motion_structure_fusion_main_asr_no_density_v3_gmot.csv` | 固定结构分支后，运动-结构融合方式消融 |
| `04_final_module_ablation_asr_no_density_v3_gmot.csv` | 最终模块消融 |
| `FINAL_MODEL_AND_MODULE_ABLATION_CN.md` | 最终模型与模块消融中文说明 |
| `FINAL_STATUS_ASR_NO_DENSITY_CN.md` | 当前最终状态说明 |

## 最终总对比目录

```text
summary/final_comparison_asr_no_density_20260706
```

主要文件：

| File | 用途 |
| --- | --- |
| `final_all_datasets_test_paper_table.csv` | 最终完整大表 |
| `final_all_datasets_test_concise.csv` | 核心指标简表 |
| `final_all_datasets_test_macro_average.csv` | 多数据集宏平均 |
| `final_all_datasets_test_comparison.md` | 中文可读版总对比 |

## 原始输出目录

```text
raw_outputs/IR-DMSTrack-v3
raw_outputs/GMOT-40-small-target
raw_outputs/IRSatVideo-LEO
```

`raw_outputs/` 只保留最终 MS-LSFTrack 模型的轨迹、point/IoU 评价和运行时间。对比算法与消融实验结果以 `summary/` 中的表格数值为准。

每个数据集下包含：

```text
tracks/
eval/
runtime/
```

## 当前不要直接引用

- IR-DMSTrack 相关表格、配置和原始输出均使用 `IR-DMSTrack-v3`。
- `final_comparison_asr_20260705` 属于旧最终口径，不作为当前最终总对比表。
- `paper_ablation_v3_gmot` 属于旧整理口径；当前主线以 `pre_module_ablation_main_20260706` 为准。
