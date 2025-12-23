# experiment.py

import pickle
import os
from pipeline import ProvenanceRunner
from config import (
    LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, grid_path, GRID_RESOLUTION, path, 
    use_env_var, env_vars, features_dict, feature_types, test_method, outfile, lr, plot_isoscapes_flag, prior_type, verbose, completeness_required,
    mean, geo_dict, cardinal_site_clusters_dict, central_peripheral_sites, geographic, experiment_variable, options
)

# Where to save experiment results
RESULTS_PATH = f"{outfile}/{experiment_variable}_experiment_results"
os.makedirs(RESULTS_PATH, exist_ok=True)


# Learning rates to sweep

# Storage for all LR experiments
all_results = {}

# -------------------------------------------------------------
# RUN EXPERIMENTS
# -------------------------------------------------------------
for option in options:
    print(f"\n=====================================")
    print(f" Running {test_method} for {experiment_variable}={option} ")
    print(f"=====================================\n")

    # Create runner instance
    override_kwargs = {f"{experiment_variable}_override": option}
    runner = ProvenanceRunner(**override_kwargs)

    # Override the learning rate
    #runner.n_iterns = n_iters       # <--- important override

    # Run ONLY random K-fold cross validation
    fold_results = runner.run()

    # Store results
    all_results[option] = fold_results

    # Save results after each loop
    save_path = f"{RESULTS_PATH}/{experiment_variable}_experiment.pkl"

    # Ensure the save directory exists (fixes PermissionError)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)

print(f"\nExperiment complete. Results saved to:\n {save_path}\n")




