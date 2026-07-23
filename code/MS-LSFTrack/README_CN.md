# MS-LSFTrack

MS-LSFTrack 是面向密集小目标多目标跟踪的检测后关联框架。代码固定检测输入，重点研究如何利用局部空间结构信息改善在线轨迹关联。
公开包还额外整理了 `weights/reid/` 下的公共 ReID 权重，方便 BoxMOT 对比方法直接复现。

本公开版本对应当前论文 v1 冻结口径：

```text
tail-anchor ASR
+ tail-node local structure
+ density-guided multi-scale local structure field
```

不包含后续 dev 探索中的预测位置 ASR、轨迹级结构记忆或结构预测实验。

## 最终论文模型

```text
Structure branch:
  structure_radii = 10,15,20
  scale_mode = learned_density
  density_prior_direction = normal
  density_prior_lambda = 0.5

Motion-structure fusion:
  score_mode = ambiguity_rerank
  asr_density_gain = 0.0

ASR anchor:
  last observation / track tail

Structure history:
  tail-node local structure
```

密度信息只用于结构分支的多尺度融合，不进入最终运动-结构 ASR 融合增益。

## 目录

```text
configs/                 数据集与实验配置
mslsftrack/              核心数据、模型、在线 tracker
tools/                   训练、跟踪、评价、诊断入口
scripts/                 论文复现与常用流程脚本
third_party/             统一 MOT 评价与可视化工具
docs/                    复现说明、代码/结果映射
weights/                 用于基线复现的公开权重
DATASET_FORMAT_CN.md     数据集 clean layout 说明
requirements.txt         Python 依赖
```

最常用的论文复现入口：

```text
scripts/paper_run_ms_lsftrack.sh
scripts/paper_evaluate_tracker.sh
scripts/paper_run_boxmot_baselines.sh
tools/build_cache.py
tools/train_assoc.py
tools/run_tracker.py
tools/evaluate_tracks.py
tools/run_baseline_trackers.py
```

完整代码与结果映射见：

```text
docs/CODE_AND_RESULTS_MAP_CN.md
```

## 环境

```bash
conda create -n mslsftrack python=3.10 -y
conda activate mslsftrack
pip install -r requirements.txt
```

如果要复现 BoxMOT 对比方法，需要额外安装兼容版本的 `boxmot`。公开包已包含 `weights/reid/osnet_x0_25_msmt17.pt`，可直接用于 StrongSORT、BoT-SORT、DeepOCSORT、HybridSORT、BoostTrack。

## 数据集格式

数据集使用统一 clean layout：

```text
DatasetRoot/
├── detecion_label/{train,val,test}/{scene}.txt
├── track_label/{scene}.txt
├── sequence/{scene}/...
└── splits/{train,val,test,all}.txt
```

其中 `detecion_label` 的拼写沿用当前整理好的数据集。详细格式见：

```text
DATASET_FORMAT_CN.md
```

推荐用环境变量指定路径：

```bash
export MS_LSF_DATA_ROOT=/path/to/MS-LSFTrack-Datasets
export MS_LSF_OUTPUT_ROOT=/path/to/MS-LSFTrack-Outputs
export MS_LSF_RUN_ROOT=/path/to/MS-LSFTrack-Runs
export MS_LSF_CACHE_ROOT=/path/to/MS-LSFTrack-Cache
export MS_LSF_WEIGHT_ROOT=/path/to/MS-LSFTrack-Weights
```

## 运行论文最终模型

使用公开结果包中的 checkpoint：

```bash
bash scripts/paper_run_ms_lsftrack.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  /path/to/MS-LSFTrack-results/checkpoints/ms_lsf_ird_v3/best.pt \
  ms_lsf_ird_v3_paper \
  test
```

同时评估 point 和 IoU：

```bash
bash scripts/paper_evaluate_tracker.sh \
  configs/datasets/irdmstrack_v3_det_label.yaml \
  ms_lsf_ird_v3_paper \
  test
```

该脚本实际使用的关键参数是：

```bash
python tools/run_tracker.py \
  --score-mode ambiguity_rerank \
  --scale-mode learned_density \
  --structure-radii-override 10,15,20 \
  --density-prior-direction normal \
  --density-prior-lambda 0.5 \
  --asr-density-gain 0.0
```

## 从零训练

构建缓存：

```bash
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split train --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split val --output-dir "$MS_LSF_CACHE_ROOT"
python tools/build_cache.py --dataset configs/datasets/irdmstrack_v3_det_label.yaml --split test --output-dir "$MS_LSF_CACHE_ROOT"
```

训练多尺度 LSF 关联网络：

```bash
python tools/train_assoc.py \
  --model multiscale_lsf \
  --neighbor-mode radius \
  --radii 10,15,20 \
  --train-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/train.pkl" \
  --val-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/val.pkl" \
  --test-cache "$MS_LSF_CACHE_ROOT/IR-DMSTrack-v3/det_label/test.pkl" \
  --output "$MS_LSF_RUN_ROOT/ms_lsf_ird_v3" \
  --device cuda
```

训练完成后使用 `scripts/paper_run_ms_lsftrack.sh` 在线出轨迹。

## 结果包

配套结果包建议单独发布：

```text
MS-LSFTrack-results/
├── summary/       论文总表与消融表
├── raw_outputs/   最终 tracks、point/IoU eval、runtime
├── checkpoints/   公开 checkpoint
├── configs/       冻结运行配置
└── manifests/     冻结清单
```

## 注意

- 本仓库只整理当前论文 v1 公开版本，不包含 dev 预测位置/结构预测实验。
- 本仓库不包含检测器训练代码。
- 上传 GitHub 前请自行决定并添加开源 license。
