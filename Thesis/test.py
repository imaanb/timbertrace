import os
import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.model_selection import KFold
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.geometry import Point, shape
from shapely.prepared import prep
import matplotlib.tri as mtri
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore")
from config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, GRID_RESOLUTION


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output/gpr", exist_ok=True)

# --- Constants ---
BBOX = (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)  # Define these in config.py
GRID_RESOLUTION = 200  # Define in config.py

# --- Per-feature atmospheric kernel ---
class PerFeatureLinearKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False
    def __init__(self, num_dimensions, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.num_dimensions = num_dimensions
        self.register_parameter(
            name="raw_weights",
            parameter=torch.nn.Parameter(torch.zeros(num_dimensions))
        )
        self.register_constraint("raw_weights", gpytorch.constraints.Positive())
    @property
    def weights(self):
        return self.raw_weights_constraint.transform(self.raw_weights)
    def forward(self, x1, x2, diag=False, **params):
        if self.active_dims is not None:
            active = torch.tensor(self.active_dims, dtype=torch.long, device=x1.device)
            max_idx = int(active.max().item())
            if max_idx >= x1.shape[-1]:
                raise IndexError(f"active_dims contains index {max_idx} but input has {x1.shape[-1]} features.")
            x1 = x1[..., active]
            x2 = x2[..., active]
        w2 = self.weights ** 2
        if diag:
            return torch.sum(w2 * x1 * x2, dim=-1)
        return (x1 * w2).matmul(x2.transpose(-1, -2))

# --- GP model (Single isotope) ---
class SingleIsoGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_coords=2, p_atm=0, mean='constant'):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() if mean == "constant" else gpytorch.means.LinearMean(input_size=2)
        matern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 1]))
        if p_atm > 0:
            atm_active = list(range(n_coords, n_coords + p_atm))
            atm_kernel = PerFeatureLinearKernel(num_dimensions=p_atm, active_dims=atm_active)
            self.covar_module = matern + atm_kernel
        else:
            self.covar_module = matern
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- Parallel training wrapper ---
def train_single_gp_wrapper(args):
    train_x, train_y, mean, n_iters, lr, n_coords, p_atm, verbose = args
    return train_single_gp(train_x, train_y, mean, n_iters, lr, n_coords, p_atm, verbose)

def train_single_gp(train_x, train_y, mean, n_iters=300, lr=0.05, n_coords=2, p_atm=0, verbose=False):
    if train_x.shape[0] < 3:
        if verbose:
            print("Not enough training points for GP (need >=3). Skipping.")
        return None, None, []
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SingleIsoGPModel(train_x, train_y, likelihood, n_coords=n_coords, p_atm=p_atm, mean=mean).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss_history = []
    for i in range(n_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    model.eval()
    likelihood.eval()
    return model, likelihood, loss_history

# --- Parallel training of per-isotope GPs ---
def train_per_isotope_gps_parallel(samples_df, isotopes, outfile, mean, use_environmental_vars=False, env_var_start_col=301, n_iters=400, lr=0.05, verbose=False, n_workers=4):
    trained = {}
    args_list = []
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
        else:
            U = None
            X_full = X_coords
            p_atm = 0
        X_full_t = torch.from_numpy(X_full.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(device)
        args_list.append((X_full_t, y_t, mean, n_iters, lr, 2, p_atm, verbose))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(train_single_gp_wrapper, args_list))

    for i, iso in enumerate(isotopes):
        if df_iso.shape[0] >= 3:
            model, lik, loss_hist = results[i]
            trained[iso] = {
                'model': model,
                'lik': lik,
                'Xcoords': X_coords,
                'U': U,
                'loss_history': loss_hist
            }
    return trained

# --- Batch predictions ---
@torch.no_grad()
def predict_gp_on_grid(model_lik_tuple, A_coords, U_grid=None, batch_size=1000):
    model, lik = model_lik_tuple
    n_grid = A_coords.shape[0]
    if model is None or lik is None:
        return np.full(n_grid, np.nan), np.full(n_grid, np.nan)
    mu = np.zeros(n_grid, dtype=np.float32)
    var = np.zeros(n_grid, dtype=np.float32)
    for i in range(0, n_grid, batch_size):
        batch_coords = A_coords[i:i + batch_size]
        if U_grid is not None:
            batch_U = U_grid[i:i + batch_size]
            Xq = torch.from_numpy(np.hstack([batch_coords, batch_U]).astype(np.float32)).to(device)
        else:
            Xq = torch.from_numpy(batch_coords.astype(np.float32)).to(device)
        with gpytorch.settings.fast_pred_var():
            pred = lik(model(Xq))
            mu[i:i + batch_size] = pred.mean.cpu().numpy()
            var[i:i + batch_size] = pred.variance.cpu().numpy()
    return mu, var

# --- Predict all isotopes on grid (batched) ---
def predict_all_isotopes_on_grid(trained_gps, A_coords, U_grid=None, batch_size=1000):
    iso_list = list(trained_gps.keys())
    n_grid = A_coords.shape[0]
    m = len(iso_list)
    means = np.zeros((n_grid, m), dtype=np.float32)
    vars_ = np.zeros((n_grid, m), dtype=np.float32)
    for j, iso in enumerate(iso_list):
        model, lik = trained_gps[iso]['model'], trained_gps[iso]['lik']
        mu, var = predict_gp_on_grid((model, lik), A_coords, U_grid, batch_size)
        means[:, j] = mu
        vars_[:, j] = var
    return means, vars_, iso_list

# --- Natural Earth helpers ---
def get_natural_earth_geoms(name, category='physical', resolution='10m'):
    shp = shapereader.natural_earth(resolution=resolution, category=category, name=name)
    reader = shapereader.Reader(shp)
    return [rec.geometry for rec in reader.records()]

def build_land_mask_natural_earth(A_coords, resolution='10m', bbox=None):
    land_geoms = get_natural_earth_geoms('land', category='physical', resolution=resolution)
    prepared = [prep(shape(g)) if not hasattr(g, 'prepared') else g for g in land_geoms]
    pts = [Point(lon, lat) for lat, lon in A_coords]
    mask = np.zeros(len(A_coords), dtype=bool)
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        idxs = np.where((A_coords[:,0] >= lat_min) & (A_coords[:,0] <= lat_max) & (A_coords[:,1] >= lon_min) & (A_coords[:,1] <= lon_max))[0]
    else:
        idxs = np.arange(len(A_coords))
    for prep_geom in prepared:
        idxs_test = idxs[~mask[idxs]]
        for idx in idxs_test:
            p = pts[idx]
            try:
                if prep_geom.contains(p) or prep_geom.touches(p):
                    mask[idx] = True
            except Exception:
                try:
                    geom = prep_geom.context if hasattr(prep_geom, 'context') else prep_geom
                    if geom.contains(p) or geom.touches(p):
                        mask[idx] = True
                except Exception:
                    pass
    return mask

# --- Plot helpers ---
def add_coast_and_borders(ax):
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')
    ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.6)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.3)

def save_tricontour_on_land(A_coords, values, land_mask, extent, outpath, cmap='viridis', levels=50, title=None):
    land_coords = A_coords[land_mask]
    land_vals = values[land_mask]
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
    triang = mtri.Triangulation(land_coords[:,1], land_coords[:,0])
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    add_coast_and_borders(ax)
    if extent: ax.set_extent(extent, crs=ccrs.PlateCarree())
    tcf = ax.tricontourf(triang, land_vals, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    plt.colorbar(tcf, ax=ax, label='Value')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close()
def compute_likelihood_maps_for_sample(means, vars_, iso_list, y_obs):
    """
    Compute per-isotope likelihood (pdf) maps for a single test sample y_obs (length m)
    Returns likelihoods: (n_grid, m)
    """
    n_grid, m = means.shape
    liks = np.ones((n_grid, m), dtype=np.float64)
    for j in range(m):
        if np.isnan(y_obs[j]):
            continue
        mu = means[:, j]
        var = vars_[:, j]
        var = np.maximum(var, 1e-10)
        lik = norm.pdf(y_obs[j], loc=mu, scale=np.sqrt(var))
        liks[:, j] = lik
    return liks

# --- Core pipeline ---
def run_provenance_pipeline(samples_df, isotopes, env_df, test_samples_df, outfile,
                            use_environmental_vars=False, env_var_start_col=291,
                            grid_resolution=200, n_iters=1000, completeness_required=100,
                            lr=0.05, plot_isoscapes_flag=True, bbox=BBOX, prior_type='flat',
                            verbose=False, mean="Constant", plot_hpd=False, n_workers=4):
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_lin = np.round(np.linspace(lat_min, lat_max, grid_resolution), 9)
    lon_lin = np.round(np.linspace(lon_min, lon_max, grid_resolution), 9)
    Lon_grid, Lat_grid = np.meshgrid(lon_lin, lat_lin)
    A_coords = np.column_stack([Lat_grid.ravel(), Lon_grid.ravel()])
    n_grid = A_coords.shape[0]
    extent = [lon_min, lon_max, lat_min, lat_max]
    if use_environmental_vars:
        coords_env = env_df[['latitude', 'longitude']].values
        tree = KDTree(coords_env)
        dist, idx = tree.query(A_coords, k=1)
        env_vars = env_df.iloc[idx[:, 0]].reset_index(drop=True)
        U_grid = env_vars.iloc[:, env_var_start_col:].values.astype(np.float32)
    else:
        U_grid = None
    if verbose: print("Building land mask (Natural Earth)...")
    land_mask = build_land_mask_natural_earth(A_coords, resolution='10m', bbox=bbox)
    if verbose:
        print(f"Grid pts: {n_grid}, land pts: {land_mask.sum()}, ocean pts: {n_grid - land_mask.sum()}")
    if verbose: print("Training per-isotope GPs (dropping NaNs per isotope)...")
    trained_gps = train_per_isotope_gps_parallel(samples_df, isotopes, outfile, mean, use_environmental_vars, env_var_start_col, n_iters, lr, verbose, n_workers)
    if verbose: print("Predicting per-isotope isoscapes on grid...")
    means, vars_, iso_list = predict_all_isotopes_on_grid(trained_gps, A_coords, U_grid)
    results = {}
    if prior_type == 'flat':
        prior = np.ones(n_grid, dtype=np.float32)
    else:
        prior = np.ones(n_grid, dtype=np.float32)
    prior = prior / prior.sum()
    for idx, row in test_samples_df.reset_index(drop=True).iterrows():
        sample_name = f"sample_{idx}"
        y_obs = np.array([row.get(iso, np.nan) for iso in iso_list], dtype=np.float64)
        test_lat = row['Latitude']; test_lon = row['Longitude']
        if verbose:
            print(f"\nProcessing test sample {idx}: lat={test_lat}, lon={test_lon}, isotopes={y_obs}")
        lik_maps = compute_likelihood_maps_for_sample(means, vars_, iso_list, y_obs)
        iso_outputs = {}
        for j, iso in enumerate(iso_list):
            lik = lik_maps[:, j]
            lik_land = lik.copy()
            lik_land[~land_mask] = 0.0
            s = lik_land.sum()
            if s > 0:
                lik_land_norm = lik_land / s
            else:
                lik_land_norm = lik_land
            out_landnorm = f"{outfile}/individual_maps/{sample_name}_{iso}_lik_landnorm.png"
            if plot_isoscapes_flag:
                save_tricontour_on_land(A_coords, lik_land_norm, land_mask, extent, out_landnorm, cmap='magma', title=f"{sample_name} - {iso} likelihood (norm over land)")
            iso_outputs[iso] = {
                'lik_raw': lik,
                'lik_land_norm': lik_land_norm,
                'plot_landnorm': out_landnorm
            }
        pix = np.prod(lik_maps, axis=1)
        numer = pix * prior
        numer[~np.isfinite(numer)] = 0.0
        numer[~land_mask] = 0.0
        s = numer.sum()
        if s > 0:
            posterior = numer / s
        else:
            posterior = np.zeros_like(numer)
            posterior[land_mask] = pix[land_mask] + 1e-300
            posterior /= posterior.sum()
        out_post = f"{outfile}/{sample_name}_combined_posterior.png"
        if plot_isoscapes_flag:
            save_tricontour_on_land(A_coords, posterior, land_mask, extent, out_post, cmap='viridis', title=f"{sample_name} - Combined posterior (sea excluded)")
        mode_idx = np.nanargmax(posterior)
        mode_coord = A_coords[mode_idx]
        true_coord = np.array([test_lat, test_lon])
        distance_deg = np.sqrt(np.sum((true_coord - mode_coord)**2))
        if plot_hpd:
            masks = hpd_mask_from_posterior(posterior, levels=[0.5, 0.9, 0.95])
            fig = plt.figure(figsize=(10,8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            add_coast_and_borders(ax)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            land_coords = A_coords[land_mask]
            land_posterior = posterior[land_mask]
            triang = mtri.Triangulation(land_coords[:,1], land_coords[:,0])
            tcf = ax.tricontourf(triang, land_posterior, levels=50, cmap='viridis', transform=ccrs.PlateCarree())
            plt.colorbar(tcf, ax=ax, label='Posterior')
            colors = {0.5: 'yellow', 0.9: 'orange', 0.95: 'red'}
            for lvl, mask in reversed(list(masks.items())):
                points_mask = mask & land_mask
                ax.scatter(A_coords[points_mask,1], A_coords[points_mask,0], s=2, c=colors.get(lvl,'white'), alpha=0.5, transform=ccrs.PlateCarree(), label=f'HPD {int(lvl*100)}%')
            ax.scatter(true_coord[1], true_coord[0], c='green', s=80, marker='s', edgecolors='black', label='True', transform=ccrs.PlateCarree())
            ax.scatter(mode_coord[1], mode_coord[0], c='red', s=100, marker='x', label='MAP', transform=ccrs.PlateCarree())
            ax.legend(loc='upper right')
            ax.set_title(f"{sample_name} - Combined posterior and HPD\nMAP dist (deg) = {distance_deg:.4f}")
            out_hpd = f"{outfile}/{sample_name}_posterior_hpd.png"
            plt.savefig(out_hpd, bbox_inches='tight', dpi=300)
            plt.close()
        results[sample_name] = {
            'y_obs': y_obs,
            'mode_coord': mode_coord,
            'distance_deg': distance_deg,
            'posterior': posterior,
            'posterior_plot': out_post,
            'per_isotope': iso_outputs
        }
    return results

# --- HPD helper ---
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

# --- Main ---
if __name__ == "__main__":
    samples = pd.read_csv("data/transformed_genetic_data.csv")
    samples.columns = samples.columns.str.strip()
    samples = samples.replace("NA", np.nan)
    for col in samples.columns:
        samples[col] = samples[col].astype(str).str.strip()
        samples[col] = pd.to_numeric(samples[col], errors='coerce')
    env_df = pd.read_csv('data/congo_basin_grid_parallel.csv')
    env_df.columns = env_df.columns.str.strip()
    env_df = env_df.rename(columns={'elevation': 'Elevation'})
    env_vars = ["Elevation", "sea_dist_km", "bio1", "bio2", "bio3", "bio4", "bio5", "bio6", "bio7", "bio8", "bio9", "bio10", "bio11", "bio12", "bio13", "bio14", "bio15", "bio16", "bio17", "bio18", "bio19"]
    env_var_start_col = samples.columns.get_loc(env_vars[0])
    for var in env_vars:
        min_val = env_df[var].min()
        max_val = env_df[var].max()
        if max_val > min_val:
            env_df[var] = (env_df[var] - min_val) / (max_val - min_val)
        else:
            env_df[var] = 0.0
    isotopes = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40', 'P41', 'P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49', 'P50', 'P51', 'P52', 'P53', 'P54', 'P55', 'P56', 'P57', 'P58', 'P59', 'P60', 'P61', 'P62', 'P63', 'P64', 'P65', 'P66', 'P67', 'P68', 'P69', 'P70', 'P71', 'P72', 'P73', 'P74', 'P75', 'P76', 'P77', 'P78', 'P79', 'P80', 'P81', 'P82', 'P83', 'P84', 'P85', 'P86', 'P87', 'P88', 'P89', 'P90', 'P91', 'P92', 'P93', 'P94', 'P95', 'P96', 'P97', 'P98', 'P99', 'P100', 'P101', 'P102', 'P103', 'P104', 'P105', 'P106', 'P107', 'P108', 'P109', 'P110', 'P111', 'P112', 'P113', 'P114', 'P115', 'P116', 'P117', 'P118', 'P119', 'P120', 'P121', 'P122', 'P123', 'P124', 'P125', 'P126', 'P127', 'P128', 'P129', 'P130', 'P131', 'P132', 'P133', 'P134', 'P135', 'P136', 'P137', 'P138', 'P139', 'P140', 'P141', 'P142', 'P143', 'P144', 'P145', 'P146', 'P147', 'P148', 'P149', 'P150', 'P151', 'P152', 'P153', 'P154', 'P155', 'P156', 'P157', 'P158', 'P159', 'P160', 'P161', 'P162', 'P163', 'P164', 'P165', 'P166', 'P167', 'P168', 'P169', 'P170', 'P171', 'P172', 'P173', 'P174', 'P175', 'P176', 'P177', 'P178', 'P179', 'P180', 'P181', 'P182', 'P183', 'P184', 'P185', 'P186', 'P187', 'P188', 'P189', 'P190', 'P191', 'P192', 'P193', 'P194', 'P195', 'P196', 'P197', 'P198', 'P199', 'P200', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'P207', 'P208', 'P209', 'P210', 'P211', 'P212', 'P213', 'P214', 'P215', 'P216', 'P217', 'P218', 'P219', 'P220', 'P221', 'P222', 'P223', 'P224', 'P225', 'P226', 'P227', 'P228', 'P229', 'P230', 'P231', 'P232', 'P233', 'P234', 'P235', 'P236', 'P237', 'P238', 'X18O', 'X13C', 'D2H', 'dS', "Li", "Mg", "Al", "Si", "P", "K", "Ca", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Rb", "Sr", "Y", "Zr", "Mo", "Cd", "Sn", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Er", "Yb", "W", "Pb", "Bi"]
    for iso in isotopes:
        mask = ~samples[iso].isna()
        min_val = samples.loc[mask, iso].min()
        max_val = samples.loc[mask, iso].max()
        if max_val > min_val:
            samples.loc[mask, iso] = (samples.loc[mask, iso] - min_val) / (max_val - min_val)
        else:
            samples.loc[mask, iso] = 0.0
    test_samples_df = samples.iloc[np.random.choice(400, 5, replace=False)].reset_index(drop=True)
    #samples = samples.drop(test_samples_df['index']).reset_index(drop=True)
    train_samples = samples[~samples["matchname"].isin(test_samples_df['matchname'])
            ]
    
    
    
    results = run_provenance_pipeline(samples, isotopes, env_df, test_samples_df, "output/gpr",
                                      use_environmental_vars=False, env_var_start_col=env_var_start_col,
                                      grid_resolution=GRID_RESOLUTION, n_iters=100, lr=0.05,
                                      plot_isoscapes_flag=True, bbox=BBOX, prior_type='flat',
                                      verbose=True, plot_hpd=True, n_workers=4)
    print("Done. Results summary:")
    for k, v in results.items():
        print(k, "MAP:", v['mode_coord'], "dist_deg:", v['distance_deg'])
    avg_dist = np.mean([v['distance_deg'] for v in results.values()])
    print(f"Average MAP distance (deg): {avg_dist:.4f}")
