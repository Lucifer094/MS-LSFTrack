# Method Overview

MS-LSFTrack is a detection-based online MOT association framework for dense small-target scenes. It keeps the detector fixed and estimates the association between active tracks and current-frame detections.

## Local Structure Field

For each active track and each candidate detection, MS-LSFTrack builds a local structure field from neighboring detections. Neighbor offsets are normalized and rasterized into a fixed grid, then encoded by a shared CNN. The resulting feature comparison produces a structure compatibility score.

## Density-Guided Multi-Scale Structure

The released model uses three neighborhood radii:

```text
10,15,20
```

Each radius provides one local structure field. The network predicts scale weights and applies a density prior to guide the final scale mixture:

```text
scale_mode = learned_density
density_prior_direction = normal
density_prior_lambda = 0.5
```

This lets the tracker adapt the local structure cue to different target densities while keeping a fixed set of candidate radii.

## Motion-Structure Association

The online tracker combines motion distance and structure compatibility with an ambiguity-aware reranking strategy:

```text
score_mode = ambiguity_rerank
asr_density_gain = 0.0
```

The motion cue ranks likely candidates, and the structure cue is used to resolve ambiguous local associations.

## Released Configuration

```text
structure_radii = 10,15,20
scale_mode = learned_density
density_prior_direction = normal
density_prior_lambda = 0.5
score_mode = ambiguity_rerank
asr_density_gain = 0.0
```

The helper script `scripts/run_mslsftrack.sh` uses this configuration by default.
