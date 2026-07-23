# tools 目录说明

## 论文定量实验主入口

| 文件 | 说明 |
| --- | --- |
| `build_cache.py` | 从 clean-layout 数据集构建关联训练缓存 |
| `train_assoc.py` | 训练单尺度/多尺度 LSF 关联网络 |
| `run_tracker.py` | 在线关联并输出 MOT 风格轨迹 |
| `evaluate_tracks.py` | 统一 point / IoU 指标评价 |
| `run_baseline_trackers.py` | 运行 BoxMOT 对比算法 |
| `build_overall_comparison.py` | 汇总最终对比表 |
| `compare_trackers.py` | 汇总多个 tracker 的评价结果 |

推荐优先使用 `scripts/paper_run_ms_lsftrack.sh` 和 `scripts/paper_evaluate_tracker.sh`，它们已经固定论文最终参数。

## 消融与表格整理

| 文件 | 说明 |
| --- | --- |
| `run_tracking_ablation.py` | 跟踪级消融批量运行 |
| `evaluate_assoc_ablation.py` | 候选级关联消融与 smoke test |
| `plot_ablation.py` | 消融可视化 |
| `make_ablation_metric_details.py` | 消融细节表整理 |
| `make_ablation_readme.py` | 消融结果说明生成 |

## 定性/数据诊断

| 文件 | 说明 |
| --- | --- |
| `analyze_ts_amid_difficulty.py` | TS-AMID、box 分布、难度划分分析 |
| `diagnose_dataset_density.py` | 数据集密度/候选歧义诊断 |
| `diagnose_detection_motion.py` | 检测与运动可分性诊断 |
| `diagnose_score_modes.py` | score mode 调试诊断 |
| `make_gmot_tsamid_balanced_splits.py` | GMOT 难度均衡划分辅助 |

## 辅助工具

| 文件 | 说明 |
| --- | --- |
| `prepare_clean_dataset_layout.py` | 数据集格式整理 |
| `prepare_reid_weights.py` | BoxMOT ReID 权重准备；公开包已附带 `weights/reid/osnet_x0_25_msmt17.pt` 时可直接跳过 |
| `check_results.py` | 检查输出完整性 |
| `run_assoc_pipeline.py` | 端到端训练/跟踪/评价流水线 |
| `run_asdt_tracker.py` | 兼容旧 ASDT tracker 的辅助入口 |
