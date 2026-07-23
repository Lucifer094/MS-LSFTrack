# MS-LSFTrack GitHub Release Package

本目录是面向 GitHub 公开的整理包，来源于当前论文 v1 冻结版本。

## 目录

```text
MS-LSFTrack-GitHub-Release-20260717/
├── code/
│   └── MS-LSFTrack/          # 建议上传到 GitHub 的代码仓库，内含 weights/reid/ 公开权重
└── results/
    └── MS-LSFTrack-results/  # 建议作为 release asset / 网盘补充下载的结果包
```

## 公开口径

本发布包只包含当前论文 v1 口径，不包含后续 dev 探索：

```text
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

一句话表述：

```text
tail-anchor ASR + tail-node local structure + density-guided multi-scale local structure field
```

预测位置、轨迹级结构记忆、结构预测等 dev 实验没有进入本公开包。

## 上传建议

1. 将 `code/MS-LSFTrack` 作为 GitHub 仓库主体上传。
2. 将 `results/MS-LSFTrack-results` 压缩后作为 GitHub Release asset、网盘或其他下载链接提供。
3. `code/MS-LSFTrack/weights/reid/osnet_x0_25_msmt17.pt` 已随代码包一并整理，可直接用于 BoxMOT ReID baselines。
4. 数据集本体不包含在本包中，只提供 clean-layout 格式和配置文件。
5. 上传前请自行决定开源许可证；本整理包没有替你选择 license。
