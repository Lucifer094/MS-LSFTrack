# Final comparison with ASR-default no fusion density

This directory rebuilds the final comparison table using the selected final MS-LSFTrack setting:

```text
score_mode = ambiguity_rerank
scale_mode = learned_density
structure_radii = 10,15,20
density_prior_direction = normal
density_prior_lambda = 0.5
asr_density_gain = 0.0
```

The baseline rows are reused from existing formal `test` summaries. The MS-LSFTrack rows are replaced with the final ASR-no-density trackers:

- IR-DMSTrack-v3: `ablate_ird_v3_fusion_den101520_l05_asr_default_no_fusion_density`
- GMOT-40-small-target: `ablate_gmot_fusion_den101520_l05_asr_default_no_fusion_density`
- IRSatVideo-LEO: `ms_lsf_irsat_final_asr_no_density`

Files:

- `final_all_datasets_test_concise.csv`: full source paths and concise metrics.
- `final_all_datasets_test_paper_table.csv`: compact table for paper use.
- `final_all_datasets_test_macro_average.csv`: macro averages.
- `final_all_datasets_test_comparison.md`: markdown view.
