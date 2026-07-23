# 最终模型与模块消融整理

最终模型固定为：

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

即：密度只用于结构分支多尺度融合，不进入运动-结构融合增益。

## 模块消融主表

| dataset | module_setting | score_mode | uses_motion_prediction | uses_structure | uses_density_scale_fusion | uses_asr | IoU-HOTA | IoU-IDF1 | IoU-MOTA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IR-DMSTrack-v3 | Nearest / no module | last_distance | no | no | no | no | 0.4377 | 0.3941 | 0.5217 |
| IR-DMSTrack-v3 | Motion only | distance | yes | no | no | no | 0.4193 | 0.3778 | 0.4713 |
| IR-DMSTrack-v3 | Structure only | structure_zscore | no | yes | yes | no | 0.4192 | 0.3738 | 0.5128 |
| IR-DMSTrack-v3 | Final motion + structure | ambiguity_rerank | yes | yes | yes | yes | 0.4379 | 0.3953 | 0.5236 |
| GMOT-40-small-target | Nearest / no module | last_distance | no | no | no | no | 0.3599 | 0.2795 | 0.2337 |
| GMOT-40-small-target | Motion only | distance | yes | no | no | no | 0.3430 | 0.2638 | 0.2239 |
| GMOT-40-small-target | Structure only | structure_zscore | no | yes | yes | no | 0.3228 | 0.2540 | 0.2279 |
| GMOT-40-small-target | Final motion + structure | ambiguity_rerank | yes | yes | yes | yes | 0.3623 | 0.2812 | 0.2334 |

## 相对增量

| dataset | module_setting | delta_HOTA_vs_nearest | delta_IDF1_vs_nearest | delta_MOTA_vs_nearest | delta_HOTA_vs_motion_only | delta_IDF1_vs_motion_only | delta_MOTA_vs_motion_only | delta_HOTA_vs_structure_only | delta_IDF1_vs_structure_only | delta_MOTA_vs_structure_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IR-DMSTrack-v3 | Nearest / no module | 0.0000 | 0.0000 | 0.0000 | 0.0184 | 0.0163 | 0.0504 | 0.0185 | 0.0203 | 0.0089 |
| IR-DMSTrack-v3 | Motion only | -0.0184 | -0.0163 | -0.0504 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0040 | -0.0415 |
| IR-DMSTrack-v3 | Structure only | -0.0185 | -0.0203 | -0.0089 | -0.0001 | -0.0040 | 0.0415 | 0.0000 | 0.0000 | 0.0000 |
| IR-DMSTrack-v3 | Final motion + structure | 0.0002 | 0.0012 | 0.0019 | 0.0185 | 0.0174 | 0.0523 | 0.0186 | 0.0215 | 0.0108 |
| GMOT-40-small-target | Nearest / no module | 0.0000 | 0.0000 | 0.0000 | 0.0169 | 0.0157 | 0.0098 | 0.0371 | 0.0254 | 0.0057 |
| GMOT-40-small-target | Motion only | -0.0169 | -0.0157 | -0.0098 | 0.0000 | 0.0000 | 0.0000 | 0.0202 | 0.0097 | -0.0041 |
| GMOT-40-small-target | Structure only | -0.0371 | -0.0254 | -0.0057 | -0.0202 | -0.0097 | 0.0041 | 0.0000 | 0.0000 | 0.0000 |
| GMOT-40-small-target | Final motion + structure | 0.0024 | 0.0017 | -0.0002 | 0.0193 | 0.0174 | 0.0096 | 0.0395 | 0.0272 | 0.0055 |

## 当前判断

- v3 上 final 相比 nearest：HOTA +0.0002，IDF1 +0.0012，MOTA +0.0019；相比 motion only 和 structure only 均明显更好。
- GMOT 上 final 相比 nearest：HOTA +0.0024，IDF1 +0.0017，MOTA -0.0002；相比 motion only 和 structure only 的 HOTA/IDF1/MOTA 整体更优。
- 因此最终模型不能写成在每个指标都压过 nearest，但可以写成：在保留运动锚点稳定性的同时，通过结构重排提升 HOTA/IDF1，并显著优于单独 motion only / structure only。

输出 CSV：`summary/pre_module_ablation_main_20260706/04_final_module_ablation_asr_no_density_v3_gmot.csv`
