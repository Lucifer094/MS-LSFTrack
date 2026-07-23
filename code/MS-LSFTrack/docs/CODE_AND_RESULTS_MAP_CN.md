# 代码与结果映射

本文档用于区分公开代码中不同文件的用途，避免把诊断脚本、消融脚本和论文主实验入口混在一起。

## 1. 核心方法代码

这些文件实现 MS-LSFTrack 的数据、网络和在线关联逻辑。

| 文件 | 用途 |
| --- | --- |
| `mslsftrack/config.py` | 读取数据集 YAML/JSON 配置，展开环境变量 |
| `mslsftrack/mot_io.py` | MOT 风格检测、GT、预测结果读写 |
| `mslsftrack/datasets.py` | 构建关联训练 query、邻域点提取、缓存数据集 |
| `mslsftrack/models.py` | 单尺度/多尺度 LSF 网络、结构分支、运动-结构融合模块 |
| `mslsftrack/tracker.py` | 在线跟踪器、密度引导多尺度、ASR 融合、结果输出 |

## 2. 论文定量实验代码

这些脚本用于生成论文中的定量结果，包括本文方法和对比方法。

| 文件 | 用途 |
| --- | --- |
| `tools/build_cache.py` | 构建 train/val/test 关联缓存 |
| `tools/train_assoc.py` | 训练 MS-LSFTrack 关联网络 |
| `tools/run_tracker.py` | 使用训练好的 checkpoint 在线生成轨迹 |
| `tools/evaluate_tracks.py` | 统一 point / IoU 跟踪评价 |
| `tools/run_baseline_trackers.py` | 运行 ByteTrack、OC-SORT、SF-SORT、StrongSORT、BoT-SORT、DeepOCSORT、HybridSORT、BoostTrack |
| `tools/build_overall_comparison.py` | 从各数据集结果构建总体对比表 |
| `tools/compare_trackers.py` | 汇总多个 tracker 的评价结果 |
| `scripts/paper_run_ms_lsftrack.sh` | 论文最终 MS-LSFTrack 设置的推荐入口 |
| `scripts/paper_evaluate_tracker.sh` | point + IoU 评价入口 |
| `scripts/paper_run_boxmot_baselines.sh` | BoxMOT baseline 推荐入口 |

论文最终 MS-LSFTrack 设置固定为：

```text
score_mode = ambiguity_rerank
scale_mode = learned_density
structure_radii = 10,15,20
density_prior_direction = normal
density_prior_lambda = 0.5
asr_density_gain = 0.0
```

## 3. 论文消融实验代码

这些脚本用于尺度、密度、多尺度融合、模块消融等表格的复查或重新生成。

| 文件 | 用途 |
| --- | --- |
| `tools/run_tracking_ablation.py` | 批量运行跟踪级消融实验 |
| `tools/evaluate_assoc_ablation.py` | 候选级关联消融与 smoke 诊断 |
| `tools/plot_ablation.py` | 消融结果绘图 |
| `tools/make_ablation_metric_details.py` | 消融指标细节整理 |
| `tools/make_ablation_readme.py` | 消融结果说明文件生成 |
| `configs/experiments/ablation_switches.yaml` | 论文版消融开关说明 |

公开结果包中的消融表来自冻结版：

```text
MS-LSFTrack-results/summary/pre_module_ablation_main_20260706
```

## 4. 定性展示与可视化代码

这些代码用于画轨迹、对比图、数据分布和 TS-AMID 相关分析。它们不是训练/主表必须入口，但可用于论文定性图或补充材料。

| 文件 | 用途 |
| --- | --- |
| `third_party/unified_eval_suite/visualize.py` | 检测、GT、预测轨迹可视化 |
| `tools/analyze_ts_amid_difficulty.py` | TS-AMID / box 分布 / 难度划分分析 |
| `tools/diagnose_dataset_density.py` | 数据集密度与候选歧义诊断 |
| `tools/diagnose_detection_motion.py` | 检测和运动可分性诊断 |
| `tools/make_gmot_tsamid_balanced_splits.py` | GMOT TS-AMID 均衡划分辅助 |

## 5. 辅助代码

| 文件 | 用途 |
| --- | --- |
| `tools/check_results.py` | 检查 tracker 输出和评价结果是否齐全 |
| `tools/prepare_clean_dataset_layout.py` | 整理数据集为 clean layout |
| `tools/prepare_reid_weights.py` | 准备 BoxMOT ReID 权重 |
| `scripts/autodl_prepare_reid.sh` | AutoDL 环境 ReID 权重准备 |
| `scripts/autodl_run_baselines.sh` | AutoDL baseline 运行辅助 |
| `scripts/autodl_evaluate_all.sh` | AutoDL 批量评价辅助 |

## 6. 公开权重

| 文件 | 用途 |
| --- | --- |
| `weights/reid/osnet_x0_25_msmt17.pt` | BoxMOT ReID baseline 的公开权重，可直接用于 StrongSORT、BoT-SORT、DeepOCSORT、HybridSORT、BoostTrack |

## 7. 结果包映射

| 结果目录 | 内容 |
| --- | --- |
| `summary/final_comparison_asr_no_density_20260706` | 三数据集最终对比总表 |
| `summary/pre_module_ablation_main_20260706` | 论文主线消融表 |
| `raw_outputs/IR-DMSTrack-v3` | IR-DMSTrack-v3 最终轨迹、评价、runtime |
| `raw_outputs/GMOT-40-small-target` | GMOT-40-small-target 最终轨迹、评价、runtime |
| `raw_outputs/IRSatVideo-LEO` | IRSatVideo-LEO 最终轨迹、评价、runtime |
| `checkpoints/ms_lsf_ird_v3` | IR-DMSTrack-v3 训练得到的 MS-LSFTrack checkpoint |
| `checkpoints/ms_lsf_gmot` | GMOT checkpoint |
| `checkpoints/ms_lsf_irsat` | IRSatVideo-LEO checkpoint |

## 8. 不包含的内容

公开代码不包含：

- 检测器训练代码；
- 原始数据集；
- 后续 dev 预测位置 ASR / 结构预测 / 轨迹级结构记忆实验；
- 旧的失败或废弃消融结果；
- IDE 配置文件和本机路径。
