# Paper v1 Public Manifest: Tail-Anchor ASR

整理日期：2026-07-17

本清单对应当前论文 v1 公开结果包。公开口径只包含论文中使用的稳定版本，不包含后续关于预测位置 ASR、轨迹级结构记忆、预测结构场等探索实验。

## 当前论文 v1 真实机制

```text
Dataset:
  IR-DMSTrack-v3

Structure branch:
  structure_radii = 10,15,20
  scale_mode = learned_density
  density_direction = normal
  density_lambda = 0.5

Motion-structure fusion:
  score_mode = ambiguity_rerank
  asr_density_gain = 0.0

ASR anchor:
  tail / last observation anchor

Structure history:
  tail-node local structure
```

论文 v1 方法应如实表述为：

```text
tail-anchor ASR + tail-node local structure + density-guided multi-scale local structure field
```

其中密度信息用于结构分支内部的多尺度融合，不进入最终运动-结构 ASR 融合增益。

## 结果包目录

```text
MS-LSFTrack-results/
├── summary/
│   ├── final_comparison_asr_no_density_20260706/
│   └── pre_module_ablation_main_20260706/
├── raw_outputs/
│   ├── IR-DMSTrack-v3/
│   ├── GMOT-40-small-target/
│   └── IRSatVideo-LEO/
├── checkpoints/
│   ├── ms_lsf_ird_v3/
│   ├── ms_lsf_gmot/
│   └── ms_lsf_irsat/
├── configs/
└── manifests/
```

## 最终总对比表

```text
summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_concise.csv
summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_paper_table.csv
summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_macro_average.csv
summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_comparison.md
```

最终模型对应的 tracker/result 名：

```text
IR-DMSTrack-v3:
  ablate_ird_v3_fusion_den101520_l05_asr_default_no_fusion_density

GMOT-40-small-target:
  ablate_gmot_fusion_den101520_l05_asr_default_no_fusion_density

IRSatVideo-LEO:
  ms_lsf_irsat_final_asr_no_density
```

最终模型在总表中的 IoU 口径结果：

| Dataset | IoU-HOTA | IoU-IDF1 | IoU-MOTA | FPS |
| --- | ---: | ---: | ---: | ---: |
| IR-DMSTrack-v3 | 0.437860 | 0.395256 | 0.523635 | 17.636 |
| GMOT-40-small-target | 0.362329 | 0.281197 | 0.233448 | 29.602 |
| IRSatVideo-LEO | 0.006941 | 0.000641 | -1.467676 | 119.519 |

注意：IRSatVideo-LEO 的 IoU 指标整体不适合作为主要叙事口径，论文中更适合结合 point 指标解释。

## 主线消融表

```text
summary/pre_module_ablation_main_20260706/01_single_scale_structure_only_v3_gmot.csv
summary/pre_module_ablation_main_20260706/02_structure_only_multiscale_density_v3_gmot.csv
summary/pre_module_ablation_main_20260706/03_motion_structure_fusion_main_asr_no_density_v3_gmot.csv
summary/pre_module_ablation_main_20260706/04_final_module_ablation_asr_no_density_v3_gmot.csv
summary/pre_module_ablation_main_20260706/FINAL_MODEL_AND_MODULE_ABLATION_CN.md
summary/pre_module_ablation_main_20260706/FINAL_STATUS_ASR_NO_DENSITY_CN.md
```

其中 `03_motion_structure_fusion_fixed_structure_v3_gmot.csv` 保留为回溯记录，不建议作为论文主表直接展示。

## 原始输出

`raw_outputs/` 只保留最终 MS-LSFTrack 模型的原始输出。对比算法与消融实验的公开包主体以 `summary/` 中的表格数值为准，不额外打包全部中间 tracker 输出。

每个数据集在 `raw_outputs/<dataset>/` 下保留：

```text
tracks
eval/iou_0.5
eval/point_3px
runtime
```

## Checkpoints

```text
checkpoints/ms_lsf_ird_v3/best.pt
checkpoints/ms_lsf_gmot/best.pt
checkpoints/ms_lsf_irsat/best.pt
```

## 已知设计限制

- 当前 ASR 的基础距离排序和歧义判断使用轨迹尾节点距离，不是预测位置距离。
- 当前历史端结构只使用轨迹尾节点所在帧的局部结构，不是轨迹级结构记忆，也没有结构外推/预测。
- 预测位置 ASR、轨迹级结构记忆、预测结构场等属于后续探索，不进入当前论文 v1 公开代码与结果。
