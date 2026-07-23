# Results Package

The companion results package contains released checkpoints, final predictions, evaluation summaries and table files.

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

These checkpoints can be passed directly to `scripts/run_mslsftrack.sh`.

## Final Tracker Names

| Dataset | Tracker name in released raw outputs |
| --- | --- |
| IR-DMSTrack-v3 | `ms_lsf_ird_v3` |
| GMOT-40-small-target | `ms_lsf_gmot` |
| IRSatVideo-LEO | `ms_lsf_irsat` |

## Summary Tables

Final comparison:

```text
summary/final_comparison/final_all_datasets_test_paper_table.csv
summary/final_comparison/final_all_datasets_test_concise.csv
summary/final_comparison/final_all_datasets_test_macro_average.csv
summary/final_comparison/final_all_datasets_test_comparison.md
```

Ablation tables:

```text
summary/ablations/01_single_scale_structure_only_v3_gmot.csv
summary/ablations/02_structure_only_multiscale_density_v3_gmot.csv
summary/ablations/03_motion_structure_fusion_v3_gmot.csv
summary/ablations/04_module_ablation_v3_gmot.csv
```

Raw outputs:

```text
raw_outputs/<dataset>/tracks/
raw_outputs/<dataset>/eval/
raw_outputs/<dataset>/runtime/
```
