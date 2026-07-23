# 2025/2026 多目标跟踪对比方法接入建议

这个项目的评价协议要求 tracker 输出稳定多目标 ID：

```text
frame,track_id,x,y,w,h,score,-1,-1,-1
```

因此，RGB-T 单目标跟踪器、Anti-UAV 单目标跟踪器，以及只发布检测后处理代码的方法，都不适合作为主表 MOT 对比方法。它们可以作为检测前端或补充讨论，但不能直接用于 HOTA、IDF1、IDs 等 ID 指标。

## 已确认不适合作为主 MOT 的方法

- `A-Simple-Detector-is-a-Strong-Tracker`：公开代码包含 frame dynamics 数据生成和 TC-Filtering 检测后处理，没有官方多目标 ID 关联器。本仓库的 `tools/run_asdt_tracker.py` 只是弱启发式适配，不建议写成官方复现。
- STTrack、SUTrack、UETrack 等 RGB-T/SOT 方法：单目标跟踪器本身不输出多目标 ID，除非额外设计逐目标初始化和冲突处理，否则不进入主 MOT 表。

## 当前最稳的主对比

优先使用已经接入统一协议的 BoxMOT 系列，因为它们可以直接消费三数据集的检测文件并输出 MOT-style track 文件：

```text
bytetrack ocsort sfsort strongsort botsort deepocsort hybridsort boosttrack
```

运行非 ReID 方法：

```bash
TRACKERS="bytetrack ocsort sfsort" \
bash scripts/autodl_run_baselines.sh configs/datasets/irdmstrack_v3_det_label.yaml test
```

运行 ReID 方法：

```bash
TRACKERS="strongsort botsort deepocsort hybridsort boosttrack" \
REID_WEIGHTS=osnet_x0_25_msmt17.pt \
bash scripts/autodl_run_baselines.sh configs/datasets/irdmstrack_v3_det_label.yaml test
```

公开代码包已经包含 `weights/reid/osnet_x0_25_msmt17.pt`，上述脚本会优先从 `${MS_LSF_WEIGHT_ROOT:-${PWD}/weights}/reid/` 中解析这个文件。

同一脚本可替换为：

```text
configs/datasets/gmot40_small_det_label.yaml
configs/datasets/irsatvideo_leo_resunet_rfr.yaml
```

## 可作为后续新增的 2025 MOT 方法

这些是真正 MOT 方法，有公开代码，但接入成本高于 BoxMOT，需要先适配数据集和检测输入输出：

| 方法 | 年份 | 代码 | 接入判断 |
|---|---:|---|---|
| TrackTrack | CVPR 2025 | https://github.com/kamkyu94/TrackTrack | 真 MOT，官方代码。建议优先研究其 inference/eval 输入格式，再写转换器导出到本项目统一 track txt。 |
| MOTIP | CVPR 2025 | https://github.com/MCG-NJU/MOTIP | 真 MOT，ID prediction 范式。较强，可能需要训练或至少适配 embedding/trajectory 输入。适合作为强 baseline。 |
| ThermalTrack | CVPRW 2025 | https://github.com/StadlerDaniel/ThermalTrack | 热红外 MOT，方向相关。对象是行人，不是点状小目标，迁移到密集小目标时效果未必强，但论文审稿认可度不错。 |

建议优先级：

1. `TrackTrack`：最新 CVPR 2025，通用在线 MOT，作为新方法对比最自然。
2. `ThermalTrack`：热红外 MOT，领域相关，但要说明其原始任务是 thermal pedestrian MOT。
3. `MOTIP`：较强且训练适配成本更高，适合作为补充强 baseline。

## 接入验收标准

新增外部 tracker 只有满足以下条件才进入主表：

- 每个 scene 输出一个 `{scene}.txt`；
- 每行是 `frame,track_id,x,y,w,h,score,-1,-1,-1`；
- `frame` 使用 1-based；
- `track_id` 是跨帧稳定 ID，不是每帧重新编号；
- 输出路径为 `{MS_LSF_OUTPUT_ROOT}/<dataset>/tracks/<det_version>/<tracker>/<split>/<scene>.txt`；
- runtime 写入 `{MS_LSF_OUTPUT_ROOT}/<dataset>/runtime/<det_version>/<tracker>/<split>_summary.csv`。

完成后统一用：

```bash
python tools/check_results.py --dataset <dataset.yaml> --split test --trackers <tracker>
python tools/evaluate_tracks.py --dataset <dataset.yaml> --split test --trackers <tracker> --mode point
python tools/evaluate_tracks.py --dataset <dataset.yaml> --split test --trackers <tracker> --mode iou
```
