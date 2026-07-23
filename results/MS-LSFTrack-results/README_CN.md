# MS-LSFTrack 公开结果包

本结果包对应当前论文 v1 冻结口径：

```text
tail-anchor ASR + tail-node local structure + density-guided multi-scale local structure field
```

## 目录结构

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

## 最终 MS-LSFTrack tracker 名

| Dataset | Tracker name |
| --- | --- |
| IR-DMSTrack-v3 | `ablate_ird_v3_fusion_den101520_l05_asr_default_no_fusion_density` |
| GMOT-40-small-target | `ablate_gmot_fusion_den101520_l05_asr_default_no_fusion_density` |
| IRSatVideo-LEO | `ms_lsf_irsat_final_asr_no_density` |

## 最终模型 IoU 结果

| Dataset | IoU-HOTA | IoU-IDF1 | IoU-MOTA | FPS |
| --- | ---: | ---: | ---: | ---: |
| IR-DMSTrack-v3 | 0.437860 | 0.395256 | 0.523635 | 17.636 |
| GMOT-40-small-target | 0.362329 | 0.281197 | 0.233448 | 29.602 |
| IRSatVideo-LEO | 0.006941 | 0.000641 | -1.467676 | 119.519 |

注意：IRSatVideo-LEO 的 IoU 指标整体很低，论文写作时更适合结合 point 指标解释。

## 主表

```text
summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_concise.csv
summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_paper_table.csv
summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_macro_average.csv
summary/final_comparison_asr_no_density_20260706/final_all_datasets_test_comparison.md
```

## 消融表

```text
summary/pre_module_ablation_main_20260706/01_single_scale_structure_only_v3_gmot.csv
summary/pre_module_ablation_main_20260706/02_structure_only_multiscale_density_v3_gmot.csv
summary/pre_module_ablation_main_20260706/03_motion_structure_fusion_main_asr_no_density_v3_gmot.csv
summary/pre_module_ablation_main_20260706/04_final_module_ablation_asr_no_density_v3_gmot.csv
```

其中 `03_motion_structure_fusion_fixed_structure_v3_gmot.csv` 保留完整回溯行，不建议作为论文主表展示。

## 原始输出

`raw_outputs/` 只保留最终 MS-LSFTrack 模型的原始输出，用于复查本文方法的最终轨迹、point/IoU 评价结果和运行时间。对比算法与消融实验的公开包主体以 `summary/` 中的表格数值为准，不额外打包全部中间 tracker 输出。

每个数据集保留：

```text
tracks/      MOT 风格预测轨迹
eval/        point_3px 与 iou_0.5 评价
runtime/     每场景和总运行时间
```

## Checkpoints

```text
checkpoints/ms_lsf_ird_v3/best.pt
checkpoints/ms_lsf_gmot/best.pt
checkpoints/ms_lsf_irsat/best.pt
```

这些 checkpoint 可配合代码仓库中的 `scripts/paper_run_ms_lsftrack.sh` 复现实验。
