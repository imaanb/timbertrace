import os
import pickle
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from backup import PerFeatureLinearKernel, run_provenance_pipeline
from config import (
    LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, grid_path, GRID_RESOLUTION, path, 
    use_env_var, env_vars, features_dict, feature_types, test_method, outfile, 
    n_iters, lr, plot_isoscapes_flag, prior_type, verbose, completeness_required,
    mean, geo_dict, cardinal_site_clusters_dict, central_peripheral_sites, geographic, plot_hpd
)



class ProvenanceRunner:
    

    def __init__(self, n_iters_override = None, lr_override = None):
        self.BBOX = (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

        # Load sample CSV
        self.samples = self._load_samples(path)
        

        # Environmental grid
        self.env_df = self._load_env(grid_path)

        # Isotope/feature list
        self.features = self._extract_features()

        # Select valid candidate samples
        self.test_candidates = self._filter_candidates()
        self.n_iters = n_iters_override if n_iters_override is not None else n_iters
        self.lr = lr_override if lr_override is not None else lr


    class FeatureNormalizer:
        """
        Performs per-feature min-max normalization: 
        - Fit on TRAIN ONLY
        - Normalize ONLY non-NaN rows
        - Replace constant columns with 0
        - Test normalization uses TRAIN min/max
        """
        def __init__(self, feature_list):
            self.features = feature_list
            self.mins = None
            self.maxs = None

        def fit(self, df):
            mins = {}
            maxs = {}

            for iso in self.features:
                col = df[iso]
                valid = col.dropna()
                if valid.empty:
                    mins[iso] = 0.0
                    maxs[iso] = 0.0
                else:
                    mins[iso] = valid.min()
                    maxs[iso] = valid.max()

            self.mins = mins
            self.maxs = maxs

        def transform(self, df):
            df = df.copy()
            for iso in self.features:
                mask = ~df[iso].isna()
                mn = self.mins[iso]
                mx = self.maxs[iso]

                if mx > mn:
                    df.loc[mask, iso] = (df.loc[mask, iso] - mn) / (mx - mn)
                else:
                    df.loc[mask, iso] = 0.0

            return df

        def fit_transform(self, df):
            self.fit(df)
            return self.transform(df)
    # ------------------------------------------------------------------
    # LOADING + CLEANUP
    # ------------------------------------------------------------------

    def _load_samples(self, path):
        samples = pd.read_csv(path)
        samples.columns = samples.columns.str.strip()
        samples = samples.replace("NA", np.nan)
        for col in samples.columns:
            samples[col] = samples[col].astype(str).str.strip()
            if col != 'matchname':
                samples[col] = pd.to_numeric(samples[col], errors='coerce')

        return samples

    def _load_env(self, grid_path):
        env_df = pd.read_csv(grid_path)
        env_df.columns = env_df.columns.str.strip()
        env_df = env_df.rename(columns={'elevation': 'Elevation'})

        # Normalize environmental variables
        for var in env_vars:
            mn, mx = env_df[var].min(), env_df[var].max()
            env_df[var] = (env_df[var] - mn) / (mx - mn) if mx > mn else 0.0

        return env_df

    """
    def _extract_features(self):
        # Nested lists inside features_dict → flatten
        feature_list = [
            features_dict[ft] for ft in feature_types
        ]
        flat = []
        for f in feature_list:
            flat.extend(f if isinstance(f, list) else [f])
        return flat
    """

    def _extract_features(self):
        flat = []
        for ft in feature_types:
            items = features_dict[ft]
            if isinstance(items, list):
                flat.extend(items)
            else:
                flat.append(items)
        return flat

    def _filter_candidates(self):
        feature_mask = (
            self.samples[self.features].notna().mean(axis=1)
            >= (completeness_required / 100)
        )
        return self.samples[feature_mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    def normalize_features(self):
        """
        Vectorized normalization of all features using min/max per column.
        Safe for NaNs and constant-value columns.
        """

        df = self.samples

        # Compute min/max per feature on non-NaN values
        mins = df[self.features].min(skipna=True)
        maxs = df[self.features].max(skipna=True)

        # Avoid division by zero: replace zeros with NaN, then fill with 1
        ranges = (maxs - mins).replace(0, np.nan)

        # Apply vectorized normalization
        df[self.features] = (df[self.features] - mins) / ranges

        # Columns where range was zero → fill with 0
        df[self.features] = df[self.features].fillna(0.0)

        self.samples = df



    # ------------------------------------------------------------------
    # MAIN EXECUTION LOGIC
    # ------------------------------------------------------------------

    def run(self, return_result  = True):
        """Entry point — calls the correct evaluation method."""
        #self.normalize_features()

        if not geographic:
            if test_method == "random":
                result = self.run_random_test(return_result = return_result)
            elif test_method == "Kfold_RD":
                result = self.run_kfold_random(return_result = return_result)
        else:
            result = self.run_geographic_cv(return_result = return_result)

        if return_result:
            return result
    # ------------------------------------------------------------------
    # RANDOM 5-SAMPLE DIAGNOSTIC
    # ------------------------------------------------------------------

    def run_random_test(self, return_result = True):
        os.makedirs(f"{outfile}/random/individual_maps", exist_ok=True)

        test_samples = self.test_candidates.iloc[
            np.random.choice(len(self.test_candidates),20,replace=False)
        ].reset_index(drop=False)

        # Avoid leakage
        train_samples = self.samples[~self.samples['matchname'].isin(test_samples['matchname'])]
        
        normalizer =self.FeatureNormalizer(self.features)

        # Fit on TRAIN only
        train_samples = normalizer.fit_transform(train_samples)

        # Transform TEST using TRAIN min/max
        test_samples = normalizer.transform(test_samples)
 

        results = run_provenance_pipeline(
            train_samples, self.features, self.env_df, test_samples,
            outfile=f"{outfile}/random",
            use_environmental_vars=use_env_var,
            env_var_start_col=self.samples.columns.get_loc(env_vars[0]),
            grid_resolution=GRID_RESOLUTION,
            n_iters=self.n_iters,
            lr=lr,
            completeness_required=completeness_required,
            plot_isoscapes_flag=plot_isoscapes_flag,
            bbox=self.BBOX,
            prior_type=prior_type,
            verbose=verbose,
            mean=mean,
        )

        print("Done. Results summary:")
        for k, v in results.items():
            print(k, v["distance_deg"])

        if return_result:
            return results

    # ------------------------------------------------------------------
    # RANDOM KFOLD CROSS-VALIDATION
    # ------------------------------------------------------------------

    def run_kfold_random(self, return_result = True):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        

        for fold, (train_idx, test_idx) in enumerate(kf.split(self.test_candidates)):

            fold_dir = f"{outfile}/{test_method}/fold{fold+1}"
            os.makedirs(f"{fold_dir}/individual_maps", exist_ok=True)

            train_samples = self.test_candidates.iloc[train_idx].reset_index(drop=True)
            test_samples = self.test_candidates.iloc[test_idx][
                ["Latitude", "Longitude"] + self.features
            ].reset_index(drop=True)
            normalizer =self.FeatureNormalizer(self.features)

            # Fit on TRAIN only
            train_samples = normalizer.fit_transform(train_samples)

            # Transform TEST using TRAIN min/max
            test_samples = normalizer.transform(test_samples)

            fold_results = run_provenance_pipeline(
                train_samples, self.features, self.env_df, test_samples,
                outfile=fold_dir,
                use_environmental_vars=use_env_var,
                env_var_start_col=self.samples.columns.get_loc(env_vars[0]),
                grid_resolution=GRID_RESOLUTION,
                n_iters=self.n_iters,
                lr=lr,
                completeness_required=completeness_required,
                plot_isoscapes_flag=plot_isoscapes_flag,
                bbox=self.BBOX,
                prior_type=prior_type,
                verbose=verbose,
                mean=mean,
            )

            # Save results after every fold
            #results_list.append(fold_results)
            with open(f"{outfile}/{test_method}/all_folds_results.pkl", "wb") as f:
                pickle.dump(fold_results, f)

        
        # Calculate and print average dist_deg per fold
        avg_dist_deg = []
        for fold in range(5):
            fold_file = f"{outfile}/{test_method}/all_folds_results.pkl"
            with open(fold_file, "rb") as f:
                fold_results = pickle.load(f)
            fold_distances = [v["distance_deg"] for v in fold_results.values()]
            avg_dist_deg.append(np.mean(fold_distances))
            print(f"Fold {fold+1} average dist_deg: {avg_dist_deg[-1]:.4f}")

    

    # ------------------------------------------------------------------
    # GEOGRAPHIC CROSS-VALIDATION
    # ------------------------------------------------------------------

    def run_geographic_cv(self, return_result = True):
        # Determine wind direction grouping
        wd = {
            "KF_GCN": "northern",
            "KF_GCE": "eastern",
            "KF_GCS": "southern",
            "KF_GCW": "western",
        }[test_method]

        sites = cardinal_site_clusters_dict[wd]
        results_list = []

        for site in sites:
            test_tree_ids = set(geo_dict[site])
            test_samples = self.test_candidates[
                self.test_candidates["matchname"].isin(test_tree_ids)
            ]
            train_samples = self.samples[
                ~self.samples["matchname"].isin(test_tree_ids)
            ]

            site_dir = f"{outfile}/{test_method}/site_{site}"
            os.makedirs(f"{site_dir}/individual_maps", exist_ok=True)
            normalizer =self.FeatureNormalizer(self.features)

            # Fit on TRAIN only
            train_samples = normalizer.fit_transform(train_samples)

            # Transform TEST using TRAIN min/max
            test_samples = normalizer.transform(test_samples)

            fold_results = run_provenance_pipeline(
                train_samples, self.features, self.env_df, test_samples,
                outfile=site_dir,
                use_environmental_vars=use_env_var,
                env_var_start_col=self.samples.columns.get_loc(env_vars[0]),
                grid_resolution=GRID_RESOLUTION,
                n_iters=self.n_iters,
                lr=lr,
                completeness_required=completeness_required,
                plot_isoscapes_flag=plot_isoscapes_flag,
                bbox=self.BBOX,
                prior_type=prior_type,
                verbose=verbose,
                mean=mean,
                plot_hpd=plot_hpd,
            )

            results_list.append((site, fold_results))
        if return_result:
            return results_list
        print("\nGeographic CV complete.")

    # ------------------------------------------------------------------
    


# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    runner = ProvenanceRunner()
    runner.run()
