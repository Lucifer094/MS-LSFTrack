# 模块消融前主线实验整理（2026-07-06）

本目录只整理模块消融之前应当确定的三步实验。所有正式行均使用 `test` split 出轨迹，并使用 `test` split 评估；当前主指标只看 `IoU-HOTA`、`IoU-IDF1`、`IoU-MOTA`。

## 文件

| 文件 | 内容 |
| --- | --- |
| `00_pre_module_ablation_status.csv` | 模块消融前的状态审计 |
| `01_single_scale_structure_only_v3_gmot.csv` | 实验一：单尺度结构信息消融 |
| `02_structure_only_multiscale_density_v3_gmot.csv` | 实验二：Structure Only 下多尺度融合与密度引导消融 |
| `03_motion_structure_fusion_fixed_structure_v3_gmot.csv` | 实验三完整记录，保留 nearest / fixed m=0.25 / ASR with fusion density 等回溯行 |
| `03_motion_structure_fusion_main_asr_no_density_v3_gmot.csv` | 实验三主线展示版：已去掉 nearest anchor、fixed m=0.25、ASR with fusion density |

## 最终已定设置

结构分支固定为：

```text
scale_pool = 10,15,20
scale_mode = learned_density
density_direction = normal
density_lambda = 0.5
```

运动-结构融合固定为：

```text
score_mode = ambiguity_rerank
asr_density_gain = 0.0
```

也就是 `ASR-default no fusion density`：密度只用于结构分支的多尺度融合，不再用于运动-结构融合增益。

## 实验一：单尺度结构信息

目的：只使用局部结构信息进行关联，验证不同结构半径会导致不同关联效果，从而说明尺度不是无关变量。

| dataset | structure_radius | IoU-HOTA | IoU-IDF1 | IoU-MOTA | best_HOTA | best_IDF1 | best_MOTA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IR-DMSTrack-v3 | 5 | 0.2260 | 0.1704 | 0.2729 | no | no | no |
| IR-DMSTrack-v3 | 10 | 0.3490 | 0.3028 | 0.4874 | no | no | no |
| IR-DMSTrack-v3 | 15 | 0.4057 | 0.3599 | 0.5088 | no | no | yes |
| IR-DMSTrack-v3 | 20 | 0.4110 | 0.3657 | 0.5035 | no | no | no |
| IR-DMSTrack-v3 | 25 | 0.4111 | 0.3662 | 0.4970 | yes | yes | no |
| IR-DMSTrack-v3 | 30 | 0.4073 | 0.3640 | 0.4914 | no | no | no |
| GMOT-40-small-target | 5 | 0.2767 | 0.2036 | 0.1991 | no | no | no |
| GMOT-40-small-target | 10 | 0.3188 | 0.2478 | 0.2219 | no | yes | no |
| GMOT-40-small-target | 15 | 0.3065 | 0.2310 | 0.2250 | no | no | no |
| GMOT-40-small-target | 20 | 0.3135 | 0.2405 | 0.2231 | no | no | no |
| GMOT-40-small-target | 25 | 0.3188 | 0.2384 | 0.2262 | yes | no | yes |
| GMOT-40-small-target | 30 | 0.3067 | 0.2276 | 0.2241 | no | no | no |

结论：v3 在 20/25 附近最好，GMOT 在 10/25 附近更好，说明不同数据集对局部结构尺度的偏好不完全一致。这个实验只用于证明尺度敏感性，不用于决定运动-结构融合方式。

## 实验二：结构分支的多尺度/密度融合

固定：`Structure Only`，`radii=10,15,20`，不使用运动信息。

| dataset | variant | density_direction | density_lambda | IoU-HOTA | IoU-IDF1 | IoU-MOTA | delta_HOTA_vs_learned | delta_IDF1_vs_learned | delta_MOTA_vs_learned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IR-DMSTrack-v3 | uniform average of three structure scales | n/a | n/a | 0.4123 | 0.3680 | 0.5140 | -0.0063 | -0.0062 | 0.0032 |
| IR-DMSTrack-v3 | learned scale weights from structure branch | normal | n/a | 0.4186 | 0.3742 | 0.5107 | 0.0000 | 0.0000 | 0.0000 |
| IR-DMSTrack-v3 | density-only scale prior | normal | 1.0 | 0.4056 | 0.3606 | 0.5096 | -0.0130 | -0.0136 | -0.0011 |
| IR-DMSTrack-v3 | learned weights corrected by density prior | normal | 0.25 | 0.4183 | 0.3748 | 0.5116 | -0.0003 | 0.0006 | 0.0009 |
| IR-DMSTrack-v3 | learned weights corrected by density prior | normal | 0.5 | 0.4192 | 0.3738 | 0.5128 | 0.0006 | -0.0004 | 0.0021 |
| GMOT-40-small-target | uniform average of three structure scales | n/a | n/a | 0.3196 | 0.2452 | 0.2272 | 0.0016 | -0.0019 | 0.0002 |
| GMOT-40-small-target | learned scale weights from structure branch | normal | n/a | 0.3180 | 0.2471 | 0.2270 | 0.0000 | 0.0000 | 0.0000 |
| GMOT-40-small-target | density-only scale prior | normal | 1.0 | 0.2989 | 0.2279 | 0.2250 | -0.0191 | -0.0192 | -0.0019 |
| GMOT-40-small-target | learned weights corrected by density prior | normal | 0.25 | 0.3210 | 0.2492 | 0.2270 | 0.0030 | 0.0021 | -0.0000 |
| GMOT-40-small-target | learned weights corrected by density prior | normal | 0.5 | 0.3228 | 0.2540 | 0.2279 | 0.0048 | 0.0069 | 0.0010 |

结论：`density-only` 不能替代学习权重；更合理的是 `learned+density`。综合 v3 和 GMOT，当前结构分支固定为 `scale_pool=10,15,20`，`scale_mode=learned_density`，`density_direction=normal`，`density_lambda=0.5`。

## 实验三：固定结构分支后的运动-结构融合

这里使用主线展示版，已经按最终选择去掉 `nearest anchor`、`fixed m=0.25`、`ASR with fusion density`。完整记录仍保留在 `03_motion_structure_fusion_fixed_structure_v3_gmot.csv`。

| dataset | variant | variant_group | IoU-HOTA | IoU-IDF1 | IoU-MOTA | delta_HOTA_vs_final_asr | delta_IDF1_vs_final_asr | delta_MOTA_vs_final_asr | selected_final_fusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IR-DMSTrack-v3 | structure_only_selected | structure only reference | 0.4192 | 0.3738 | 0.5128 | -0.0186 | -0.0215 | -0.0108 | no |
| IR-DMSTrack-v3 | fixed_m0p5 | fusion mechanism | 0.4436 | 0.4017 | 0.5126 | 0.0057 | 0.0065 | -0.0110 | no |
| IR-DMSTrack-v3 | fixed_m0p75 | fusion mechanism | 0.4316 | 0.3910 | 0.4982 | -0.0063 | -0.0042 | -0.0255 | no |
| IR-DMSTrack-v3 | residual | fusion mechanism | 0.4090 | 0.3647 | 0.5074 | -0.0289 | -0.0306 | -0.0162 | no |
| IR-DMSTrack-v3 | competitive | fusion mechanism | 0.4108 | 0.3656 | 0.5132 | -0.0271 | -0.0296 | -0.0104 | no |
| IR-DMSTrack-v3 | asr_default_no_fusion_density | selected final fusion | 0.4379 | 0.3953 | 0.5236 | 0.0000 | 0.0000 | 0.0000 | yes |
| GMOT-40-small-target | structure_only_selected | structure only reference | 0.3228 | 0.2540 | 0.2279 | -0.0395 | -0.0272 | -0.0055 | no |
| GMOT-40-small-target | fixed_m0p5 | fusion mechanism | 0.3476 | 0.2742 | 0.2256 | -0.0147 | -0.0070 | -0.0078 | no |
| GMOT-40-small-target | fixed_m0p75 | fusion mechanism | 0.3393 | 0.2665 | 0.2255 | -0.0231 | -0.0147 | -0.0079 | no |
| GMOT-40-small-target | residual | fusion mechanism | 0.3498 | 0.2672 | 0.2322 | -0.0126 | -0.0140 | -0.0012 | no |
| GMOT-40-small-target | competitive | fusion mechanism | 0.3270 | 0.2577 | 0.2239 | -0.0353 | -0.0235 | -0.0095 | no |
| GMOT-40-small-target | asr_default_no_fusion_density | selected final fusion | 0.3623 | 0.2812 | 0.2334 | 0.0000 | 0.0000 | 0.0000 | yes |

结论：

1. `ASR-default no fusion density` 作为最终融合方式：运动作为主锚点，结构只在运动歧义时介入；密度只保留在结构分支多尺度融合中。
2. `fixed m=0.25` 虽然在 v3 上 HOTA/IDF1 高，但已经从主线展示表去掉，因为它在 GMOT 上不稳，且固定权重不适合作为最终机制。
3. `ASR with fusion density` 也从主线展示表去掉，因为当前结果显示融合增益里的密度调制收益很小且混合，不作为最终方法。

## 下一步：模块消融

后续模块消融应固定上述最终结构分支和最终融合方式，只做模块层面的对比：

| 行 | 含义 |
| --- | --- |
| nearest / no module | 不用运动预测、不用结构，只做 last-position nearest |
| motion only | 只用运动预测距离 |
| structure only | 只用已确定的密度引导多尺度结构分支 |
| final motion+structure | `ASR-default no fusion density` + 已确定结构分支 |

不要在模块消融里再重新比较尺度、密度、fixed fusion 或 ASR with fusion density。
