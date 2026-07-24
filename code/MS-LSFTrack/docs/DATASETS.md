# Datasets

This repository does not include dataset files. Download the prepared archives below and extract them under `MS_LSF_DATA_ROOT`.

| Dataset | Archive | Download | Extraction code | Expected directory |
| --- | --- | --- | --- | --- |
| IR-DMSTrack-v3 | `IR-DMSTrack.zip` | [Baidu Netdisk](https://pan.baidu.com/s/1nXpu_Ts6t3ANxCXyrYR4SQ) | `sbn4` | `$MS_LSF_DATA_ROOT/IR-DMSTrack-v3` |
| GMOT-40-small-target | `GMOT-40-small-target.zip` | [Baidu Netdisk](https://pan.baidu.com/s/1oue_l778O-Mk_FJPIjKsEA) | `j7hk` | `$MS_LSF_DATA_ROOT/GMOT-40-small-target` |
| IRSatVideo-LEO | `IRSatVideo-LEO.zip` | [Baidu Netdisk](https://pan.baidu.com/s/1To1X7HW0Rk-49BNZPG19-A) | `mgwr` | `$MS_LSF_DATA_ROOT/IRSatVideo-LEO` |

The released `IR-DMSTrack.zip` archive corresponds to `IR-DMSTrack-v3` in the configs and result tables.

After extraction, set:

```bash
export MS_LSF_DATA_ROOT=/path/to/MS-LSFTrack-Datasets
```

The resulting directory should look like:

```text
$MS_LSF_DATA_ROOT/
├── IR-DMSTrack-v3/
├── GMOT-40-small-target/
└── IRSatVideo-LEO/
```

Each dataset directory must follow the clean association layout described in [../DATASET_FORMAT.md](../DATASET_FORMAT.md).

```text
DatasetRoot/
├── detecion_label/{train,val,test}/{scene}.txt
├── track_label/{scene}.txt
├── sequence/{scene}/...
└── splits/{train,val,test,all}.txt
```

The directory name `detecion_label` is kept for compatibility with the released configs.

Please use the datasets for academic research and follow the license or usage terms of the corresponding original datasets when applicable.
