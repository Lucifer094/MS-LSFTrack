# Dataset Format

MS-LSFTrack uses fixed detections and focuses on the association stage of multi-object tracking. Each dataset should provide images, ground-truth tracks, detections and scene-level splits.

## Directory Layout

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

The directory name `detecion_label` is kept for compatibility with the released configs.

## Detection Files

Path:

```text
DatasetRoot/detecion_label/{split}/{scene}.txt
```

Recommended MOT-style format:

```text
frame,-1,x,y,w,h,score,-1,-1,-1
```

Fields:

| Field | Description |
| --- | --- |
| `frame` | 1-based frame index |
| `-1` | placeholder ID for detections |
| `x,y,w,h` | top-left box coordinate, width and height |
| `score` | detection confidence |
| trailing `-1` values | MOT-style placeholders |

Six-column detections are also accepted by the conversion tools:

```text
frame,x,y,w,h,score
```

## Ground Truth Files

Path:

```text
DatasetRoot/track_label/{scene}.txt
```

Recommended MOT-style format:

```text
frame,id,x,y,w,h,score,class,visibility
```

The common 10-column placeholder format is also supported:

```text
frame,id,x,y,w,h,1,-1,-1,-1
```

## Images

Path:

```text
DatasetRoot/sequence/{scene}/
```

Recommended file naming:

```text
000001.png
000002.png
...
```

Training and evaluation use detections and annotations. Images are used by visualization and diagnostic tools.

## Split Files

Path:

```text
DatasetRoot/splits/train.txt
DatasetRoot/splits/val.txt
DatasetRoot/splits/test.txt
DatasetRoot/splits/all.txt
```

Each line contains one scene name:

```text
00001
00002
AfricaWest-1_38
```

Scene names must match `sequence/{scene}`, `track_label/{scene}.txt` and `detecion_label/{split}/{scene}.txt`.

## Released Configs

```text
configs/datasets/irdmstrack_v3_det_label.yaml
configs/datasets/gmot40_small_det_label.yaml
configs/datasets/irsatvideo_leo_resunet_rfr.yaml
```
