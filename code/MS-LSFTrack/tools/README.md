# Tools

Main entry points:

| File | Purpose |
| --- | --- |
| `build_cache.py` | Build association caches from clean-layout datasets |
| `train_assoc.py` | Train the MS-LSFTrack association model |
| `run_tracker.py` | Run online MS-LSFTrack tracking |
| `evaluate_tracks.py` | Evaluate tracks with point-distance or IoU metrics |
| `run_baseline_trackers.py` | Run BoxMOT baseline trackers |
| `compare_trackers.py` | Summarize tracker metrics |
| `check_results.py` | Check whether tracker outputs are complete |

Analysis and ablation utilities:

| File | Purpose |
| --- | --- |
| `run_tracking_ablation.py` | Run tracking-level ablations |
| `evaluate_assoc_ablation.py` | Evaluate association-level ablations |
| `plot_ablation.py` | Plot ablation results |
| `analyze_ts_amid_difficulty.py` | Analyze TS-AMID and box statistics |
| `diagnose_dataset_density.py` | Inspect dataset density and ambiguity |
| `diagnose_detection_motion.py` | Inspect detection and motion separability |

Preparation utilities:

| File | Purpose |
| --- | --- |
| `prepare_clean_dataset_layout.py` | Convert datasets to the clean layout |
| `prepare_reid_weights.py` | Prepare ReID weights for BoxMOT baselines |
| `run_assoc_pipeline.py` | Run cache building, training, tracking and evaluation as one pipeline |
