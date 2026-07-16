To generate the noise experiment plots in the IROS 2026 paper, run: 
    - `generate_noise_comparison_table.py`, which expects the results path to the 
    results in OpenVINS format. This runs error_comparison, generates the ATE table,
    then generates a table of results by noise level.
    - `evaluate_vins_noise_experiment.py` generates the time-series plots 
    for multiple noise levels.

These scripts were then refactoed into `generate_result_csvs.py`, 
which generates a NEES and a ATE CSV from a result folder in
the same format as expected by OpenVINS.

The ATE CSV should have the same output as  the `error_comparison`, 
while the NEES csv should have the mean att/pos NEES for each algorithm/dataset combo.

The ATE CSV generated has the following columns:
```
algorithm, dataset, mean_ori, std_ori, mean_pos, std_pos, num_runs
```

The NEES CSV generated has the following columns:
```
algorithm, dataset, mean_ori_nees, mean_pos_nees, num_runs
```

Then, run the script `generate_tables_from_csv` to recreate the ATE and NEES tables for various noise levels.