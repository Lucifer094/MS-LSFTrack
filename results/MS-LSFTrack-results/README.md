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

## Checkpoints

```text
checkpoints/ms_lsf_ird_v3/best.pt
checkpoints/ms_lsf_gmot/best.pt
checkpoints/ms_lsf_irsat/best.pt
```

Use these checkpoints with `scripts/run_mslsftrack.sh` from the code repository.

## Final Tracker Names

| Dataset | Tracker name |
| --- | --- |
| IR-DMSTrack-v3 | `ms_lsf_ird_v3` |
| GMOT-40-small-target | `ms_lsf_gmot` |
| IRSatVideo-LEO | `ms_lsf_irsat` |

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
