# MS-LSFTrack Results

This package contains the released checkpoints, raw outputs and summary tables for MS-LSFTrack.

## Layout

```text
MS-LSFTrack-results/
├── checkpoints/
├── raw_outputs/
├── summary/
├── configs/
└── manifests/
```

## Dataset Downloads

The dataset files are not included in this results package. Download the prepared archives below and extract them under `MS_LSF_DATA_ROOT`.

| Dataset | Archive | Download | Extraction code | Expected directory |
| --- | --- | --- | --- | --- |
| IR-DMSTrack-v3 | `IR-DMSTrack.zip` | [Baidu Netdisk](https://pan.baidu.com/s/1nXpu_Ts6t3ANxCXyrYR4SQ) | `sbn4` | `$MS_LSF_DATA_ROOT/IR-DMSTrack-v3` |
| GMOT-40-small-target | `GMOT-40-small-target.zip` | [Baidu Netdisk](https://pan.baidu.com/s/1oue_l778O-Mk_FJPIjKsEA) | `j7hk` | `$MS_LSF_DATA_ROOT/GMOT-40-small-target` |
| IRSatVideo-LEO | `IRSatVideo-LEO.zip` | [Baidu Netdisk](https://pan.baidu.com/s/1To1X7HW0Rk-49BNZPG19-A) | `mgwr` | `$MS_LSF_DATA_ROOT/IRSatVideo-LEO` |

The released `IR-DMSTrack.zip` archive corresponds to `IR-DMSTrack-v3` in the configs and result tables.

## Checkpoints

```text
checkpoints/IR-DMSTrack-v3/best.pt
checkpoints/GMOT-40-small-target/best.pt
checkpoints/IRSatVideo-LEO/best.pt
```

Use these checkpoints with `scripts/run_mslsftrack.sh` from the code repository.

## Released Method Name

All released MS-LSFTrack raw outputs use `MS-LSFTrack` as the tracker name. Dataset-specific checkpoints are separated by dataset directory under `checkpoints/`.

## Final IoU Metrics

| Dataset | IoU-HOTA | IoU-IDF1 | IoU-MOTA | FPS |
| --- | ---: | ---: | ---: | ---: |
| IR-DMSTrack-v3 | 0.437860 | 0.395256 | 0.523635 | 17.636 |
| GMOT-40-small-target | 0.362329 | 0.281197 | 0.233448 | 29.602 |
| IRSatVideo-LEO | 0.006941 | 0.000641 | -1.467676 | 119.519 |

## Tables

Final comparison:

```text
summary/final_comparison/
```

Ablations:

```text
summary/ablations/
```

## Raw Outputs

For each dataset, `raw_outputs/` contains:

```text
tracks/
eval/
runtime/
```
