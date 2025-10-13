# Plot for the paper figure. Unified.

First, generate plotable files from the human data and prior-aggregation simulation data. Get a unified mean, std format.
```bash
cd assets
 
python build_aggregated_panel_metrics.py --human_eye human_data/human_eye_movement_metrics.json --human_mcq human_data/human_mcq_acc_metrics.json --human_fr human_data/human_free_recall_metrics.json --sim_eye simulation_data/simulation_eye_movement_metrics.json --sim_comp simulation_data/comprehension_results_20251006-150327.json --out aggregated_panel_metrics.json
```

Plot.
```bash
cd plot

python plot_eye_comp_from_aggregated_metrics.py 
```

# Plot for the baseline comparisons. Unified.

Generate usable data metrics in json.
```bash
cd assets
python build_aggregated_panel_metrics_baseline.py --folder simulation_data_baselines/ 
```

Plot
```bash
python plot_eye_comp_and_baselines_from_aggregated_metrics.py 
```