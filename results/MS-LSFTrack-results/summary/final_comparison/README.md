# Final Comparison

This directory contains the released all-dataset comparison tables.

## MS-LSFTrack Setting

```text
score_mode = ambiguity_rerank
scale_mode = learned_density
structure_radii = 10,15,20
density_prior_direction = normal
density_prior_lambda = 0.5
asr_density_gain = 0.0
```

## MS-LSFTrack Tracker Names

| Dataset | Tracker |
| --- | --- |
| IR-DMSTrack-v3 | `ms_lsf_ird_v3` |
| GMOT-40-small-target | `ms_lsf_gmot` |
| IRSatVideo-LEO | `ms_lsf_irsat` |

## Files

| File | Content |
| --- | --- |
| `final_all_datasets_test_concise.csv` | Concise metric table with source tracker names |
| `final_all_datasets_test_paper_table.csv` | Compact reporting table |
| `final_all_datasets_test_macro_average.csv` | Macro averages across datasets |
| `final_all_datasets_test_comparison.md` | Markdown table view |
