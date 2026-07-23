# Ablation Tables

This directory contains the released ablation tables.

## Files

| File | Content |
| --- | --- |
| `01_single_scale_structure_only_v3_gmot.csv` | Tests different single structure radii under structure-only association |
| `02_structure_only_multiscale_density_v3_gmot.csv` | Tests multi-scale structure fusion and density guidance |
| `03_motion_structure_fusion_v3_gmot.csv` | Tests motion-structure fusion choices after fixing the structure branch |
| `04_module_ablation_v3_gmot.csv` | Tests the contribution of the main modules |

The main reported metrics are `IoU-HOTA`, `IoU-IDF1` and `IoU-MOTA`.
