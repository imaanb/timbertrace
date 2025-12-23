# provenance_gptorch_optionA_group_tests.py
# GP per-isotope provenance model with Natural Earth (Cartopy) land/ocean/borders
# Features:
#  - Per-isotope GP (drop NaNs only for that isotope)
#  - Individual isotope likelihood maps
#  - Combined posterior map (product of isotope likelihoods * prior)
#  - Sea excluded from solution space via Natural Earth land mask
#  - Group test samples support (Pandas DataFrame). Each test sample must have all isotopes.
#
# Requirements: torch, gpytorch, numpy, scipy, matplotlib, pandas, cartopy, shapely
# Use conda-forge for cartopy/shapely if possible.

import os
import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

import pyproj


# geospatial
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.geometry import Point, shape
from shapely.prepared import prep

from config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, GRID_RESOLUTION
# triangulation / interpolation
import matplotlib.tri as mtri
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output/gpr", exist_ok=True)

# ---------------------------
# Bounding box (Congo basin) provided earlier
# ---------------------------

BBOX = (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

# ---------------------------
# GP model (Single isotope)
# ---------------------------
class SingleIsoGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_coords=2, p_atm=0):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #matern = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=list(range(0, n_coords)))
        matern = gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=2,active_dims=[0, 1])
        matern = gpytorch.kernels.ScaleKernel(matern)
        if p_atm > 0:
            atm_active = list(range(n_coords, n_coords + p_atm))
            linear = gpytorch.kernels.LinearKernel(active_dims=atm_active)
            linear = gpytorch.kernels.ScaleKernel(linear)
            self.covar_module = matern + linear
        else:
            self.covar_module = matern

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def train_single_gp(train_x, train_y, n_iters=300, lr=0.05, n_coords=2, p_atm=0, verbose=False):
    """
    train_x : torch.Tensor (n, d)
    train_y : torch.Tensor (n,)
    returns (model, likelihood) or (None, None) if training not possible
    """
    

    if train_x.shape[0] < 3:
        # Not enough points to train a GP reliably
        if verbose:
            print("Not enough training points for GP (need >=3). Skipping.")
        return None, None

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SingleIsoGPModel(train_x, train_y, likelihood, n_coords=n_coords, p_atm=p_atm).to(device)
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(n_iters):
        optimizer.zero_grad()
        output = model(train_x)  # This should work now with the same data
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if verbose and (i % (max(1, n_iters // 10)) == 0 or i == n_iters-1):
            try:
                noise_val = likelihood.noise.item()
            except Exception:
                noise_val = float('nan')
            print(f"Iter {i+1}/{n_iters} - Loss: {loss.item():.3f} | noise: {noise_val:.6f}")
    
    model.eval(); likelihood.eval()
    return model, likelihood

@torch.no_grad()
def predict_gp_on_grid(model_lik_tuple, A_coords, U_grid=None):
    """
    Predict single GP on the full grid. Returns (means, vars) arrays (n_grid,).
    If model is None, returns arrays filled with nan.
    """
    model, lik = model_lik_tuple
    n_grid = A_coords.shape[0]
    if model is None or lik is None:
        return np.full(n_grid, np.nan), np.full(n_grid, np.nan)

    if U_grid is None:
        Xq = torch.from_numpy(A_coords.astype(np.float32)).to(device)
        #print("Predict X shape:", Xq.shape)

    else:
        X_full = np.hstack([A_coords, U_grid])
        Xq = torch.from_numpy(X_full.astype(np.float32)).to(device)
        #print("Predict X shape:", Xq.shape)

    with gpytorch.settings.fast_pred_var():
        pred = lik(model(Xq))
        mu = pred.mean.cpu().numpy()
        var = pred.variance.cpu().numpy()

    return mu, var

# ---------------------------
# Natural Earth helpers (Option A)
# ---------------------------
def get_natural_earth_geoms(name, category='physical', resolution='10m'):
    shp = shapereader.natural_earth(resolution=resolution, category=category, name=name)
    reader = shapereader.Reader(shp)
    geoms = [rec.geometry for rec in reader.records()]
    return geoms

def build_land_mask_natural_earth(A_coords, resolution='10m', bbox=None):
    """
    A_coords: (n,2) array [lat, lon]
    bbox: (lat_min, lat_max, lon_min, lon_max) optional; used for selecting test indices to speed up
    Returns boolean mask (n,) True if point considered land.
    """
    land_geoms = get_natural_earth_geoms('land', category='physical', resolution=resolution)
    prepared = [prep(shape(g)) if not hasattr(g, 'prepared') else g for g in land_geoms]

    # build points
    pts = [Point(lon, lat) for lat, lon in A_coords]
    mask = np.zeros(len(A_coords), dtype=bool)

    # If bbox provided, preselect indices
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        idxs = np.where(
            (A_coords[:,0] >= lat_min) & (A_coords[:,0] <= lat_max) & (A_coords[:,1] >= lon_min) & (A_coords[:,1] <= lon_max)
        )[0]
    else:
        idxs = np.arange(len(A_coords))

    for prep_geom in prepared:
        # only test indices not already marked land
        idxs_test = idxs[~mask[idxs]]
        for idx in idxs_test:
            p = pts[idx]
            try:
                if prep_geom.contains(p) or prep_geom.touches(p):
                    mask[idx] = True
            except Exception:
                # fallback
                try:
                    geom = prep_geom.context if hasattr(prep_geom, 'context') else prep_geom
                    if geom.contains(p) or geom.touches(p):
                        mask[idx] = True
                except Exception:
                    pass
    return mask

# ---------------------------
# Plot helpers
# ---------------------------
def add_coast_and_borders(ax):
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')
    ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.6)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.3)

def save_tricontour_on_land(A_coords, values, land_mask, extent, outpath,  cmap='viridis', levels=50, title=None):
    """
    Triangulate and save a tricontourf over LAND points only.
    A_coords: (n,2) [lat, lon]
    values: (n,) array; must have values for land points (nan or zero for ocean)
    land_mask: boolean mask over A_coords indicating land
    """
    land_coords = A_coords[land_mask]  # [lat, lon]
    land_vals = values[land_mask]

    # If no valid land values, save an empty map
    if len(land_coords) == 0 or np.all(np.isnan(land_vals)):
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        add_coast_and_borders(ax)
        if extent: ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(title or 'No data')
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()
        return

    # Triangulation expects x=lon, y=lat
    triang = mtri.Triangulation(land_coords[:,1], land_coords[:,0])
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    add_coast_and_borders(ax)
    if extent: ax.set_extent(extent, crs=ccrs.PlateCarree())

    # tricontourf with land-only triangulation & values
    tcf = ax.tricontourf(triang, land_vals, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    plt.colorbar(tcf, ax=ax, label='Value')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close()

# ---------------------------
# Core pipeline: train per-isotope GPs (drop NaNs per isotope), predict on grid, compute likelihoods/posterior
# ---------------------------
def train_per_isotope_gps(samples_df, isotopes, use_environmental_vars=False, env_var_start_col=301, n_iters=400, lr=0.05, verbose=False):
    """
    Train one GP per isotope using only rows where that isotope is not NaN.
    Returns dict: isotope -> (model, lik, X_train_coords (n,2), U_train or None)
    """
    trained = {}
    for iso in isotopes:
        df_iso = samples_df[~samples_df[iso].isna()].copy() 
        if df_iso.shape[0] < 3:
            if verbose:
                print(f"[{iso}] Only {df_iso.shape[0]} non-NaN samples -> skipping GP training.")
            trained[iso] = (None, None, None, None)
            continue
            
        X_coords = np.vstack([df_iso["Latitude"].values, df_iso["Longitude"].values]).T
        

        y = df_iso[iso].values.astype(np.float32)
        
        if use_environmental_vars and env_var_start_col < df_iso.shape[1]:
            U = df_iso.iloc[:, env_var_start_col:].values.astype(np.float32)
            X_full = np.hstack([X_coords, U])
            p_atm = U.shape[1]
            if verbose:
                print(f"[{iso}] Training X shape: {X_full.shape}, p_atm={p_atm}")
        else:
            U = None
            X_full = X_coords
            p_atm = 0
            if verbose:
                print(f"[{iso}] Training X shape: {X_full.shape}, p_atm={p_atm}")

        # Convert to torch tensors
        X_full_t = torch.from_numpy(X_full.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(device)
        
        # Train the GP
        model, lik = train_single_gp(X_full_t, y_t, n_iters=n_iters, lr=lr, n_coords=2, p_atm=p_atm, verbose=verbose)
        trained[iso] = (model, lik, X_coords, U)

    return trained
def predict_all_isotopes_on_grid(trained_gps, A_coords, U_grid=None):
    """
    Given dict trained_gps: iso -> (model, lik, Xcoords, U)
    Returns:
      means: (n_grid, m) numpy (m = len(isotopes) order of keys)
      vars_: (n_grid, m)
      iso_list: list of isotope names (order)
    """
    iso_list = list(trained_gps.keys())
    n_grid = A_coords.shape[0]
    m = len(iso_list)
    means = np.zeros((n_grid, m), dtype=np.float64)
    vars_ = np.zeros((n_grid, m), dtype=np.float64)
    for j, iso in enumerate(iso_list):
        model, lik, Xcoords, Utrain = trained_gps[iso]
        mu, var = predict_gp_on_grid((model, lik), A_coords, U_grid)
        means[:, j] = mu
        vars_[:, j] = var
    return means, vars_, iso_list

def compute_likelihood_maps_for_sample(means, vars_, iso_list, y_obs):
    """
    Compute per-isotope likelihood (pdf) maps for a single test sample y_obs (length m)
    Returns likelihoods: (n_grid, m)
    """
    n_grid, m = means.shape
    liks = np.ones((n_grid, m), dtype=np.float64)
    for j in range(m):
        mu = means[:, j]
        var = vars_[:, j]
        var = np.maximum(var, 1e-10)
        lik = norm.pdf(y_obs[j], loc=mu, scale=np.sqrt(var))
        liks[:, j] = lik
    return liks

# ---------------------------
# High-level runner: supports multiple test samples (DataFrame)
# ---------------------------
def run_provenance_pipeline(samples_df,isotopes, env_df, test_samples_df,
                            use_environmental_vars=False, env_var_start_col=291,
                            grid_resolution=200, n_iters=1000, lr=0.05,
                            plot_isoscapes_flag=True, bbox=BBOX, prior_type='flat',
                            verbose=False):
    """
    samples_df: DataFrame with columns ['Latitude','Longitude', <isotope columns...>, ...env vars...]
    isotopes: list of isotope column names
    test_samples_df: DataFrame with rows to test; must include Latitude, Longitude and all isotopes (no NaNs)
    Returns dict with results for each test sample.
    """

    # Prepare prediction grid based on bbox
    lat_min, lat_max, lon_min, lon_max = bbox
    decimals = 9
    decimals = 9  # number of decimal places

    lat_lin = np.round(np.linspace(lat_min, lat_max, grid_resolution), decimals)
    lon_lin = np.round(np.linspace(lon_min, lon_max, grid_resolution), decimals)

    Lon_grid, Lat_grid = np.meshgrid(lon_lin, lat_lin)
    
    A_coords = np.column_stack([Lat_grid.ravel(), Lon_grid.ravel()])
    # Save A_coords (the grid) to a txt file
    np.savetxt("output/gpr/A_grid.txt", A_coords, fmt="%.9f", header="Latitude Longitude")
    n_grid = A_coords.shape[0]
    extent = [lon_min, lon_max, lat_min, lat_max]

    if use_environmental_vars:
        # Match A_coords to env_df grid cells using KDTree (nearest neighbor)
        coords_env = env_df[['latitude', 'longitude']].values
        tree = KDTree(coords_env)
        dist, idx = tree.query(A_coords, k=1)
        env_vars = env_df.iloc[idx[:, 0]].reset_index(drop=True)
        U_grid = env_vars.iloc[:, env_var_start_col:].values.astype(np.float32)
       
    else:
        U_grid = None

    # Build land mask (Natural Earth), clipped to bbox for speed
    if verbose: print("Building land mask (Natural Earth)...")
    land_mask = build_land_mask_natural_earth(A_coords, resolution='10m', bbox=bbox)
    if verbose:
        print(f"Grid pts: {n_grid}, land pts: {land_mask.sum()}, ocean pts: {n_grid - land_mask.sum()}")

    # Train per-isotope GPs using only non-NaN rows per isotope
    if verbose: print("Training per-isotope GPs (dropping NaNs per isotope)...")
    trained_gps = train_per_isotope_gps(samples_df, isotopes, use_environmental_vars=use_environmental_vars,
                                        env_var_start_col=env_var_start_col, n_iters=n_iters, lr=lr, verbose=verbose)

    # Predict per-isotope means/vars on grid
    if verbose: print("Predicting per-isotope isoscapes on grid...")
    means, vars_, iso_list = predict_all_isotopes_on_grid(trained_gps, A_coords, U_grid)

    results = {}
    # Precompute prior over grid
    if prior_type == 'flat':
        prior = np.ones(n_grid, dtype=np.float64)
    else:
        prior = np.ones(n_grid, dtype=np.float64)
    prior = prior / prior.sum()

    # For each test sample (group processing)
    for idx, row in test_samples_df.reset_index(drop=True).iterrows():
        sample_name = f"sample_{idx}"
        y_obs = np.array([row[iso] for iso in iso_list], dtype=np.float64)
        test_lat = row['Latitude']; test_lon = row['Longitude']

        if verbose:
            print(f"\nProcessing test sample {idx}: lat={test_lat}, lon={test_lon}, isotopes={y_obs}")

        # Per-isotope likelihood maps (unnormalized)
        lik_maps = compute_likelihood_maps_for_sample(means, vars_, iso_list, y_obs)  # shape (n_grid, m)

        # Save per-isotope likelihood maps (both raw and normalized over land)
        iso_outputs = {}
        for j, iso in enumerate(iso_list):
            lik = lik_maps[:, j]
            # Normalize likelihood over land (so per-isotope map sums to 1 over land)
            lik_land = lik.copy()
            lik_land[~land_mask] = 0.0
            s = lik_land.sum()
            if s > 0:
                lik_land_norm = lik_land / s
            else:
                lik_land_norm = lik_land  # all zero - leave as is

            # Save raw likelihood and normalized land-only map plots
            #out_raw = f"output/gpr/{sample_name}_{iso}_lik_raw.png"
            out_landnorm = f"output/gpr/individual_maps/{sample_name}_{iso}_lik_landnorm.png"
            #save_tricontour_on_land(A_coords, lik, land_mask, extent, out_raw, cmap='magma', title=f"{sample_name} - {iso} raw likelihood")
            save_tricontour_on_land(A_coords, lik_land_norm, land_mask, extent, out_landnorm, cmap='magma', title=f"{sample_name} - {iso} likelihood (norm over land)")

            iso_outputs[iso] = {
                'lik_raw': lik,
                'lik_land_norm': lik_land_norm,
                #'plot_raw': out_raw,
                'plot_landnorm': out_landnorm
            }

        # Combined posterior: product of per-isotope likelihoods * prior, then zero-out ocean and renormalize over land
        pix = np.prod(lik_maps, axis=1)  # (n_grid,)
        numer = pix * prior
        numer[~np.isfinite(numer)] = 0.0
        # zero ocean
        numer[~land_mask] = 0.0
        s = numer.sum()
        if s > 0:
            posterior = numer / s
        else:
            # fallback tiny epsilon on land
            posterior = np.zeros_like(numer)
            posterior[land_mask] = pix[land_mask] + 1e-300
            posterior /= posterior.sum()

        # Save posterior map (land-only contour)
        out_post = f"output/gpr/{sample_name}_combined_posterior.png"
        save_tricontour_on_land(A_coords, posterior, land_mask, extent, out_post, cmap='viridis', title=f"{sample_name} - Combined posterior (sea excluded)")

        # Find MAP and distance to true location
        mode_idx = np.nanargmax(posterior)
        mode_coord = A_coords[mode_idx]
        true_coord = np.array([test_lat, test_lon])
        distance_deg = np.sqrt(np.sum((true_coord - mode_coord)**2))

        # Also save HPD masks (50%, 90%, 95%) as overlay plots (simple plotting of points inside mask)
        masks = hpd_mask_from_posterior(posterior, levels=[0.5, 0.9, 0.95])
        # Save a combined figure that shows posterior and markers
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        add_coast_and_borders(ax)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Plot posterior contours (land-only)
        land_coords = A_coords[land_mask]
        land_posterior = posterior[land_mask]
        triang = mtri.Triangulation(land_coords[:,1], land_coords[:,0])
        tcf = ax.tricontourf(triang, land_posterior, levels=50, cmap='viridis', transform=ccrs.PlateCarree())
        plt.colorbar(tcf, ax=ax, label='Posterior')
        colors = {0.5: 'yellow', 0.9: 'orange', 0.95: 'red'}
        
        for lvl, mask in reversed(list(masks.items())):
            # scatter land points within mask, slightly opaque
            points_mask = mask & land_mask
            ax.scatter(
            A_coords[points_mask,1], 
            A_coords[points_mask,0], 
            s=2, 
            c=colors.get(lvl,'white'), 
            alpha=0.5,  # slightly opaque
            transform=ccrs.PlateCarree(), 
            label=f'HPD {int(lvl*100)}%'
            )

        
        # Markers
        #ax.scatter(test_lon, test_lat, c='cyan', s=100, marker='*', edgecolors='black', label='Test sample', transform=ccrs.PlateCarree())
        ax.scatter(mode_coord[1], mode_coord[0], c='red', s=100, marker='x', label='MAP', transform=ccrs.PlateCarree())
        ax.scatter(true_coord[1], true_coord[0], c='green', s=80, marker='s', edgecolors='black', label='True', transform=ccrs.PlateCarree())
        ax.legend(loc='upper right')
        ax.set_title(f"{sample_name} - Combined posterior and HPD\nMAP dist (deg) = {distance_deg:.4f}")
        # Plot HPD contours as scatter overlays (colored)
        
        plt.tight_layout()
        out_hpd = f"output/gpr/{sample_name}_posterior_hpd.png"
        plt.savefig(out_hpd, bbox_inches='tight', dpi=300)
        plt.close()

        # assemble results
        results[sample_name] = {
            'y_obs': y_obs,
            'mode_coord': mode_coord,
            'distance_deg': distance_deg,
            'posterior': posterior,
            'posterior_plot': out_post,
            'posterior_hpd_plot': out_hpd,
            'per_isotope': iso_outputs
        }

    # Optionally: also save the full predicted means/vars arrays for later analysis
    np.save("output/gpr/pred_means.npy", means)
    np.save("output/gpr/pred_vars.npy", vars_)
    np.save("output/gpr/land_mask.npy", land_mask)

    return results

# ---------------------------
# Simple HPD helper (already used above)
# ---------------------------
def hpd_mask_from_posterior(posterior, levels=[0.5, 0.75, 0.9, 0.95]):
    idx_sorted = np.argsort(-posterior)
    cum = np.cumsum(posterior[idx_sorted])
    masks = {}
    for lvl in levels:
        cutoff_index = np.searchsorted(cum, lvl, side='right')
        mask = np.zeros_like(posterior, dtype=bool)
        if cutoff_index > 0:
            mask[idx_sorted[:cutoff_index]] = True
        masks[lvl] = mask
    return masks

# ---------------------------
# If invoked as script: example usage
# ---------------------------
if __name__ == "__main__":
    samples = pd.read_csv("data/presave.csv")

    #cleaning
    samples.columns = samples.columns.str.strip()
    samples = samples.replace("NA", np.nan)
    for col in samples.columns:
        samples[col] = samples[col].astype(str).str.strip()
        samples[col] = pd.to_numeric(samples[col], errors='coerce')
    
    #environment vars 

    env_df = pd.read_csv('data/congo_basin_grid_parallel.csv')
    env_df.columns = env_df.columns.str.strip()

    env_vars = [
    "Elevation", "sea_dist_km",
    "bio1", "bio2", "bio3", "bio4", "bio5", "bio6", "bio7",
    "bio8", "bio9", "bio10", "bio11", "bio12", "bio13", "bio14",
    "bio15", "bio16", "bio17", "bio18", "bio19"
    ]

    # Find starting index of env vars 
    env_var_start_col = samples.columns.get_loc(env_vars[0])
   
    # isotope columns 
    isotopes = ['X18O', 'X13C' , 'D2H', 'dS']
    isotopes = ["Li", "Mg", "Al", "Si", "P", "K", "Ca", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Rb", "Sr",
        "Y", "Zr", "Mo", "Cd", "Sn", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Er", "Yb", "W", "Pb", "Bi"
    ]
    # Normalize Latitude and Longitude only for each isotope's non-NaN rows
    # Normalize isotope columns only for each isotope's non-NaN rows
    for iso in isotopes:
        mask = ~samples[iso].isna()
        min_val = samples.loc[mask, iso].min()
        max_val = samples.loc[mask, iso].max()
        if max_val > min_val:
            samples.loc[mask, iso] = (samples.loc[mask, iso] - min_val) / (max_val - min_val)
        else:
            samples.loc[mask, iso] = 0.0  # If all values are the same, set to 0

    #Testing: must contain value for each isotope
    test_candidates = samples.dropna(subset=isotopes).reset_index(drop=True)
    test_samples_df = test_candidates.iloc[[1, 10, 20, 30, 40]][['Latitude','Longitude'] + isotopes].reset_index(drop=True)
    
    # Run the pipeline
    results = run_provenance_pipeline(samples, isotopes, 
                                      env_df,
                                      test_samples_df,
                                      use_environmental_vars=False,
                                      env_var_start_col = env_var_start_col,
                                      grid_resolution=GRID_RESOLUTION,   # 200x200 grid -> 40k points
                                      n_iters=3000,
                                      lr=0.01,
                                      plot_isoscapes_flag=True,
                                      bbox=BBOX,
                                      prior_type='flat',
                                      verbose=True)

    print("Done. Results summary:")
    for k,v in results.items():
        print(k, "MAP:", v['mode_coord'], "dist_deg:", v['distance_deg'])
    print("Plots & arrays in output/gpr/")
    