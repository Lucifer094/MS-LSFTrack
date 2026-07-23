# MS-LSFTrack 数据集统一格式

本项目专注于多目标跟踪中的关联问题。为了公平比较不同关联方法，数据集中需要同时提供图像、轨迹真值、标准检测结果和 train/val/test 划分。

## 1. 总体目录

每个数据集统一整理为四个目录：

```text
DatasetRoot/
├── detecion_label/
│   ├── train/{scene}.txt
│   ├── val/{scene}.txt
│   └── test/{scene}.txt
├── track_label/
│   └── {scene}.txt
├── sequence/
│   └── {scene}/image files
└── splits/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── all.txt
```

说明：

- `detecion_label` 保存标准检测结果。这里沿用当前已整理数据集的拼写。
- `track_label` 保存轨迹真值。
- `sequence` 保存图像，便于可视化、复查和后续扩展。
- `splits` 保存场景级划分，每行一个 scene 名称。

## 2. 检测结果格式

路径：

```text
DatasetRoot/detecion_label/{split}/{scene}.txt
```

推荐使用 MOT 风格 10 列：

```text
frame,-1,x,y,w,h,score,-1,-1,-1
```

字段含义：

| 字段 | 含义 |
|---|---|
| `frame` | 帧号，推荐从 1 开始 |
| `-1` | 检测结果没有真实 ID，统一填 `-1` |
| `x,y,w,h` | 左上角坐标、宽、高 |
| `score` | 检测置信度 |
| 后三列 | 占位，统一填 `-1` |

如果原始检测为 6 列：

```text
frame,x,y,w,h,score
```

发布或训练前建议转换成 10 列。

## 3. 轨迹真值格式

路径：

```text
DatasetRoot/track_label/{scene}.txt
```

推荐使用 MOT 风格：

```text
frame,id,x,y,w,h,score,class,visibility
```

也可以使用 10 列占位格式：

```text
frame,id,x,y,w,h,1,-1,-1,-1
```

## 4. 图像格式

路径：

```text
DatasetRoot/sequence/{scene}/
```

推荐命名：

```text
000001.png
000002.png
...
```

代码训练关联模型时主要使用检测和 GT；图像用于可视化、检查、后续扩展和开源完整性。

## 5. 划分文件

路径：

```text
DatasetRoot/splits/train.txt
DatasetRoot/splits/val.txt
DatasetRoot/splits/test.txt
DatasetRoot/splits/all.txt
```

每行一个 scene 名称：

```text
00001
00002
AfricaWest-1_38
```

要求 scene 名称与 `sequence/{scene}`、`track_label/{scene}.txt`、`detecion_label/{split}/{scene}.txt` 保持一致。

## 6. 当前三个数据集

当前配置默认按 AutoDL/Linux 路径读取数据集根目录：

```text
/path/to/MS-LSFTrack-Datasets
```

也可以通过环境变量覆盖，例如本机 Windows：

```powershell
$env:MS_LSF_DATA_ROOT="D:\MyProject\DenseTargetTrack\MS-LSFTrack-Datasets"
```

| 数据集 | 场景数 | 检测源 | 缺失项 |
|---|---:|---|---:|
| IR-DMSTrack | 500 | `yolov12n` | 0 |
| GMOT-40-small-target | 40 | `det_label` | 0 |
| IRSatVideo-LEO | 200 | `ResUNet_RFR` | 0 |

对应配置：

```text
configs/datasets/irdmstrack_v3_det_label.yaml
configs/datasets/gmot40_small_det_label.yaml
configs/datasets/irsatvideo_leo_resunet_rfr.yaml
```

GMOT-40-small-target 的 train/val/test 按 TS-AMID 做了场景难度均衡划分：

| split | scenes | mean TS-AMID |
|---|---:|---:|
| train | 24 | 1.777294 |
| val | 8 | 1.772027 |
| test | 8 | 1.775977 |

## 7. 论文中可使用的表述

英文：

> To focus on the association problem, we provide standardized detection results for each sequence and evaluate all trackers using the same detection inputs.

中文：

> 为了专注于多目标跟踪中的关联问题，我们为每个序列提供统一的检测结果，并在相同检测输入下评估所有跟踪方法。
