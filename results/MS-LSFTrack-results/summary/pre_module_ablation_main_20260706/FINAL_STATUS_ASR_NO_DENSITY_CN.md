# ASR-default no fusion density 最终状态整理

## 最终模型

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

含义：密度只用于结构分支的多尺度融合，不进入运动-结构融合增益。

## 已完成的主线消融

| 步骤 | 文件 | 状态 |
| --- | --- | --- |
| 单尺度结构消融 | `01_single_scale_structure_only_v3_gmot.csv` | 完成 |
| 多尺度结构/密度消融 | `02_structure_only_multiscale_density_v3_gmot.csv` | 完成 |
| 运动-结构融合消融主线表 | `03_motion_structure_fusion_main_asr_no_density_v3_gmot.csv` | 完成，最终融合已定 |
| 最终模块消融 | `04_final_module_ablation_asr_no_density_v3_gmot.csv` | 完成 |

旧的完整融合记录仍保留在 `03_motion_structure_fusion_fixed_structure_v3_gmot.csv`，其中包含 nearest anchor、fixed m=0.25、ASR with fusion density 等回溯行，但不作为主线展示表。

## 模块消融结论边界

最终模型相对 nearest/no-module：

| dataset | HOTA | IDF1 | MOTA |
| --- | ---: | ---: | ---: |
| IR-DMSTrack-v3 | +0.0002 | +0.0012 | +0.0019 |
| GMOT-40-small-target | +0.0024 | +0.0017 | -0.0002 |

最终模型相对 motion only 和 structure only 都有更明显优势。因此论文表述建议是：

- 可以写：最终模型在保留运动锚点稳定性的同时，通过局部结构重排提升 HOTA/IDF1，并显著优于单独 motion only / structure only。
- 不要写：最终模型在所有数据集、所有指标上都显著超过 nearest/no-module。

## 最终总对比表

新总表目录：

`summary/final_comparison_asr_no_density_20260706`

其中 MS-LSFTrack 三个数据集均已替换为最终 ASR-default no fusion density 口径：

| dataset | source tracker |
| --- | --- |
| IR-DMSTrack-v3 | `ablate_ird_v3_fusion_den101520_l05_asr_default_no_fusion_density` |
| GMOT-40-small-target | `ablate_gmot_fusion_den101520_l05_asr_default_no_fusion_density` |
| IRSatVideo-LEO | `ms_lsf_irsat_final_asr_no_density` |

总表文件：

| 文件 | 内容 |
| --- | --- |
| `final_all_datasets_test_concise.csv` | 带 source path 的完整简表 |
| `final_all_datasets_test_paper_table.csv` | 论文用简洁表 |
| `final_all_datasets_test_macro_average.csv` | 三数据集宏平均 |
| `final_all_datasets_test_comparison.md` | Markdown 展示表 |

## 当前还需要注意

1. 以后运行最终模型时必须显式设置 `--asr-density-gain 0.0`，否则默认值仍可能复现旧的 fusion-density 行。
2. 最终大表的 GMOT 上 PuTR 的 IoU-HOTA 高于 MS-LSFTrack，但 MS-LSFTrack 的 IDF1/MOTA 更高；论文中需要按指标如实表述。
3. IRSatVideo-LEO 的 IoU 指标整体很低且 MOTA 为负，主要看 point 指标时 MS-LSFTrack 优势明显；写作时要注意解释评估模式差异。
4. 旧目录 `final_comparison_asr_20260705` 不是当前最终口径，不要再作为最终表使用。
