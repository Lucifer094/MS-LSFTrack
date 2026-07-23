# 公开权重说明

本目录用于保存公开复现需要的模型权重。当前代码仓库只打包与论文对比方法复现直接相关的公共 ReID 权重；本文 MS-LSFTrack 训练得到的 checkpoint 放在配套结果包 `MS-LSFTrack-results/checkpoints/` 中。

## ReID baseline 权重

```text
weights/reid/osnet_x0_25_msmt17.pt
```

该权重用于 BoxMOT 中需要 ReID 分支的对比方法：

```text
StrongSORT
BoT-SORT
DeepOCSORT
HybridSORT
BoostTrack
```

运行示例：

```bash
python tools/run_baseline_trackers.py \
  --dataset configs/datasets/irdmstrack_v3_det_label.yaml \
  --split test \
  --trackers strongsort botsort deepocsort hybridsort boosttrack \
  --reid-weights weights/reid/osnet_x0_25_msmt17.pt
```

如果需要替换为其他 ReID 权重，可以继续使用：

```bash
bash scripts/autodl_prepare_reid.sh <model_name_or_path>
```

或直接把新的权重放到：

```text
weights/reid/
```
