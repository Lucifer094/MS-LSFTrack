# Release Manifest

Released MS-LSFTrack configuration:

```text
structure_radii = 10,15,20
scale_mode = learned_density
density_prior_direction = normal
density_prior_lambda = 0.5
score_mode = ambiguity_rerank
asr_density_gain = 0.0
```

Datasets:

```text
IR-DMSTrack-v3
GMOT-40-small-target
IRSatVideo-LEO
```

Released checkpoints:

```text
checkpoints/ms_lsf_ird_v3/best.pt
checkpoints/ms_lsf_gmot/best.pt
checkpoints/ms_lsf_irsat/best.pt
```

Summary tables:

```text
summary/final_comparison/
summary/ablations/
```

Raw output directories:

```text
raw_outputs/IR-DMSTrack-v3/
raw_outputs/GMOT-40-small-target/
raw_outputs/IRSatVideo-LEO/
```
