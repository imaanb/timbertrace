"""
kriging_pipeline.py

Object-oriented pipeline:
 - VariogramSelector: tries variogram models via LOO CV (OK or UK)
 - KrigingModel: run OK or UK on training data with chosen variogram
 - RegressionKriging: regression + kriging residuals
 - Validator: LOO CV or hold-out test evaluation metrics
 - Map plotting + Natural Earth land masking (as before)

Notes:
 - Maps are always constructed from TRAINING data only (user-supplied test set is never used to build maps).
 - LOO CV is used to choose variogram automatically when no dedicated validation set is supplied.
"""

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
from shapely.geometry import Point
from shapely.prepared import prep

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------
# Utilities
# ---------------------------
def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


# ---------------------------
# Land Masker (Natural Earth)
# ---------------------------
class LandMasker:
    def __init__(self, resolution="10m"):
        self.resolution = resolution
        self.land_geoms = self._load_land_geometries()
        self.prepared_geoms = [prep(g) for g in self.land_geoms]

    def _load_land_geometries(self):
        shp = shapereader.natural_earth(
            resolution=self.resolution, category="physical", name="land"
        )
        reader = shapereader.Reader(shp)
        return [r.geometry for r in reader.records()]

    def build_mask(self, coords, bbox=None):
        """
        coords: (n,2) array of [lat, lon]
        bbox: (lat_min, lat_max, lon_min, lon_max)
        """
        pts = [Point(lon, lat) for lat, lon in coords]
        mask = np.zeros(len(coords), dtype=bool)

        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            idxs = np.where(
                (coords[:, 0] >= lat_min)
                & (coords[:, 0] <= lat_max)
                & (coords[:, 1] >= lon_min)
                & (coords[:, 1] <= lon_max)
            )[0]
        else:
            idxs = np.arange(len(coords))

        for geom in self.prepared_geoms:
            idxs_test = idxs[~mask[idxs]]
            for idx in idxs_test:
                if geom.contains(pts[idx]) or geom.touches(pts[idx]):
                    mask[idx] = True

        return mask


# ---------------------------
# Map plotting helper
# ---------------------------
class MapPlotter:
    def __init__(self, extent=None):
        """
        extent = [lon_min, lon_max, lat_min, lat_max]
        """
        self.extent = extent

    @staticmethod
    def add_basemap(ax):
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor="lightblue")
        ax.add_feature(cfeature.LAND, zorder=0, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6)
        ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.5)
        ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.3)

    def save_tricontour_on_land(self, coords, values, land_mask, outpath, cmap="viridis", levels=50, title=None):
        land_coords = coords[land_mask]
        land_vals = values[land_mask]

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        self.add_basemap(ax)
        if self.extent:
            ax.set_extent(self.extent, crs=ccrs.PlateCarree())

        if len(land_coords) == 0 or np.all(np.isnan(land_vals)):
            ax.set_title(title or "No data")
        else:
            triang = mtri.Triangulation(land_coords[:, 1], land_coords[:, 0])
            tcf = ax.tricontourf(triang, land_vals, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
            plt.colorbar(tcf, ax=ax, label="Value")

        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------
# Variogram selector (tries models, chooses best by CV RMSE)
# ---------------------------
class VariogramSelector:
    def __init__(self, models=("spherical", "exponential", "gaussian", "linear")):
        self.models = models

    def evaluate_ok_by_loo(self, lons, lats, vals):
        """
        Leave-one-out cross-validation for Ordinary Kriging for each variogram model.
        Returns dict {model: rmse}.
        """
        results = {}
        n = len(vals)
        # quick checks
        if n < 5:
            # small n: still proceed but warn
            print("Warning: small number of samples (<5). CV may be unstable.")

        for model in self.models:
            preds = np.empty(n)
            for i in range(n):
                # leave i out
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                try:
                    ok = OrdinaryKriging(
                        lons[mask], lats[mask], vals[mask], variogram_model=model,
                        verbose=False, enable_plotting=False
                    )
                    # predict at the left-out point
                    zi, ss = ok.execute("points", np.array([lons[i]]), np.array([lats[i]]))
                    preds[i] = zi[0]
                except Exception as e:
                    # if model fitting fails, set predictions to nan and break
                    preds[i] = np.nan
            # filter nans
            valid = ~np.isnan(preds)
            if valid.sum() == 0:
                results[model] = np.inf
            else:
                results[model] = rmse(vals[valid], preds[valid])
            print(f"Model {model}: LOO RMSE = {results[model]:.4f}")
        return results

    def select_best(self, lons, lats, vals):
        scores = self.evaluate_ok_by_loo(lons, lats, vals)
        best = min(scores, key=scores.get)
        return best, scores


# ---------------------------
# Kriging wrapper (OK & UK)
# ---------------------------
class KrigingModel:
    def __init__(self, variogram_model="spherical", method="ok"):
        """
        method: 'ok' (ordinary) or 'uk' (universal with external drift)
        variogram_model: string accepted by pykrige
        """
        self.variogram_model = variogram_model
        self.method = method

    def fit_predict_grid_ok(self, lons, lats, vals, grid_lon, grid_lat):
        OK = OrdinaryKriging(
            lons, lats, vals,
            variogram_model=self.variogram_model,
            verbose=False, enable_plotting=False
        )
        z, ss = OK.execute("grid", grid_lon, grid_lat)
        return z, ss

    def fit_predict_grid_uk(self, lons, lats, vals, external, grid_lon, grid_lat, grid_external):
        """
        external: array same length as lons/lats (covariate)
        grid_external: 2D array matching meshgrid(grid_lon, grid_lat)
        """
        UK = UniversalKriging(
            lons, lats, vals,
            variogram_model=self.variogram_model,
            drift_terms=["external_Z"],
            external_drift=external
        )
        z, ss = UK.execute("grid", grid_lon, grid_lat, external_drift_arrays=grid_external)
        return z, ss

    def fit_predict(self, lons, lats, vals, grid_lon, grid_lat, external=None, grid_external=None):
        if self.method == "ok":
            return self.fit_predict_grid_ok(lons, lats, vals, grid_lon, grid_lat)
        elif self.method == "uk":
            if external is None or grid_external is None:
                raise ValueError("External drift arrays required for universal kriging")
            return self.fit_predict_grid_uk(lons, lats, vals, external, grid_lon, grid_lat, grid_external)
        else:
            raise ValueError("Unknown method")


# ---------------------------
# Regression-Kriging class
# ---------------------------
class RegressionKriging:
    def __init__(self, base_model=None, variogram_model="spherical"):
        self.reg = base_model or RandomForestRegressor(n_estimators=200, random_state=0)
        self.variogram = variogram_model

    def fit(self, X_train, y_train):
        self.reg.fit(X_train, y_train)
        self.yhat_train = self.reg.predict(X_train)
        self.resid_train = y_train - self.yhat_train
        # keep training coords out of X_train assumption: X_train must include lat/lon or they are passed separately

    def predict_grid(self, grid_df, train_lons, train_lats, grid_lon, grid_lat):
        """
        grid_df: DataFrame of covariates for grid points (same columns as X_train)
        train_lons, train_lats: coords for residual kriging (training points)
        """
        # regression prediction on grid
        grid_reg = self.reg.predict(grid_df)

        # krige residuals from training points
        krig = KrigingModel(variogram_model=self.variogram, method="ok")
        z_resid, ss = krig.fit_predict_grid_ok(train_lons, train_lats, self.resid_train, grid_lon, grid_lat)

        # combine (note shapes: z_resid is grid-shaped same orientation as meshgrid)
        final = grid_reg.reshape(z_resid.shape) + z_resid
        return final, ss


# ---------------------------
# Validator: LOO CV (OK/UK) and test evaluation
# ---------------------------
class Validator:
    @staticmethod
    def loo_cv_ok(lons, lats, vals, variogram_model="spherical"):
        n = len(vals)
        preds = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            ok = OrdinaryKriging(lons[mask], lats[mask], vals[mask],
                                 variogram_model=variogram_model,
                                 verbose=False, enable_plotting=False)
            zi, ss = ok.execute("points", np.array([lons[i]]), np.array([lats[i]]))
            preds[i] = zi[0]
        return preds

    @staticmethod
    def evaluate(y_true, y_pred):
        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
        return {
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "mae": mean_absolute_error(y_true[mask], y_pred[mask]),
            "r2": r2_score(y_true[mask], y_pred[mask])
        }

    @staticmethod
    def test_evaluate_on_holdout(trainer, grid_predictor=None, X_test=None, coords_test=None, y_test=None, method="ok", variogram_model="spherical", external_test=None, grid_external=None):
        """
        trainer: for OK/UK: (lons_train, lats_train, y_train) or for regression-kriging an object with predict_grid
        The function will only evaluate predictions on test coords. It WILL NOT use test points to build maps.
        """
        if method == "ok":
            lons_train, lats_train, y_train = trainer
            ok = OrdinaryKriging(lons_train, lats_train, y_train, variogram_model=variogram_model,
                                 verbose=False, enable_plotting=False)
            zi, ss = ok.execute("points", coords_test[:,1], coords_test[:,0])
            preds = np.array(zi).ravel()
            return Validator.evaluate(y_test, preds)
        elif method == "uk":
            lons_train, lats_train, y_train, external_train = trainer
            UK = UniversalKriging(lons_train, lats_train, y_train, variogram_model=variogram_model,
                                  drift_terms=["external_Z"], external_drift=external_train)
            zi, ss = UK.execute("points", coords_test[:,1], coords_test[:,0], external_drift_arrays=None)
            preds = np.array(zi).ravel()
            return Validator.evaluate(y_test, preds)
        elif method == "regression-kriging":
            # trainer is regression-kriging object which has predict_grid
            rk = trainer
            # produce regression prediction at test points by calling reg.predict
            Xtest = X_test
            reg_preds = rk.reg.predict(Xtest)
            # krige residuals using training residuals to test coords
            krig = KrigingModel(variogram_model=rk.variogram, method="ok")
            z_resid, ss = krig.fit_predict_grid_ok(rk.train_lons, rk.train_lats, rk.resid_train, np.array([coords_test[:,1]]), np.array([coords_test[:,0]]))
            # fit_predict_grid_ok expects grid vectors; here we hack predict at points by predicting grid of single row/cols
            preds = reg_preds + z_resid.ravel()
            return Validator.evaluate(y_test, preds)
        else:
            raise ValueError("Unknown method")


# ---------------------------
# Full pipeline class
# ---------------------------
class wKrigingPipeline:
    def __init__(self, train_df, lon_col="Longitude", lat_col="Latitude", value_col="value",
                 grid_res=200, bbox=None, extent=None):
        """
        train_df: DataFrame with training samples (must contain lon_col, lat_col, value_col and optionally covariates)
        bbox: (lat_min, lat_max, lon_min, lon_max)
        extent: [lon_min, lon_max, lat_min, lat_max] for plotting
        """
        self.train_df = train_df.copy()
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.value_col = value_col
        self.grid_res = grid_res
        self.bbox = bbox
        self.extent = extent

        # prepare grid vectors later
        if bbox is not None:
            lat_min, lat_max, lon_min, lon_max = bbox
            self.grid_lon = np.linspace(lon_min, lon_max, grid_res)
            self.grid_lat = np.linspace(lat_min, lat_max, grid_res)
        else:
            # auto from training data
            lon_min, lon_max = train_df[lon_col].min(), train_df[lon_col].max()
            lat_min, lat_max = train_df[lat_col].min(), train_df[lat_col].max()
            self.grid_lon = np.linspace(lon_min, lon_max, grid_res)
            self.grid_lat = np.linspace(lat_min, lat_max, grid_res)

    def clean(self):
        # basic cleaning: strip/convert similar to your previous flow
        df = self.train_df.copy()
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace("NA", np.nan)
        df = df.dropna(subset=[self.lon_col, self.lat_col, self.value_col])
        self.train_df = df.reset_index(drop=True)

    def auto_select_variogram(self):
        lons = self.train_df[self.lon_col].values
        lats = self.train_df[self.lat_col].values
        vals = self.train_df[self.value_col].values
        selector = VariogramSelector()
        best, scores = selector.select_best(lons, lats, vals)
        print("Variogram scores:", scores)
        print("Selected variogram:", best)
        self.variogram = best

    def run_ok_map(self, outpath, variogram=None):
        # build OK from training data (variogram chosen or provided)
        if variogram is None:
            variogram = getattr(self, "variogram", "spherical")
        lons = self.train_df[self.lon_col].values
        lats = self.train_df[self.lat_col].values
        vals = self.train_df[self.value_col].values

        krig = KrigingModel(variogram_model=variogram, method="ok")
        z, ss = krig.fit_predict(lons, lats, vals, self.grid_lon, self.grid_lat)

        # prepare coords, mask, plot
        lon_grid, lat_grid = np.meshgrid(self.grid_lon, self.grid_lat)
        coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        values = z.ravel()

        masker = LandMasker()
        mask = masker.build_mask(coords, bbox=self.bbox)
        plotter = MapPlotter(extent=self.extent)
        plotter.save_tricontour_on_land(coords, values, mask, outpath,
                                       cmap="viridis", levels=50, title=f"OK {variogram}")
        print("Map saved to", outpath)
        return z, ss

    def run_uk_map(self, covariate_col, grid_covariate_array, outpath, variogram=None):
        """
        covariate_col: column name in train_df used as external drift
        grid_covariate_array: 2D array matching meshgrid(self.grid_lon, self.grid_lat)
        """
        if variogram is None:
            variogram = getattr(self, "variogram", "spherical")

        df = self.train_df.dropna(subset=[covariate_col])
        lons = df[self.lon_col].values
        lats = df[self.lat_col].values
        vals = df[self.value_col].values
        external = df[covariate_col].values

        krig = KrigingModel(variogram_model=variogram, method="uk")
        z, ss = krig.fit_predict(lons, lats, vals, self.grid_lon, self.grid_lat,
                                 external=external, grid_external=grid_covariate_array)

        lon_grid, lat_grid = np.meshgrid(self.grid_lon, self.grid_lat)
        coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        values = z.ravel()
        masker = LandMasker()
        mask = masker.build_mask(coords, bbox=self.bbox)
        plotter = MapPlotter(extent=self.extent)
        plotter.save_tricontour_on_land(coords, values, mask, outpath,
                                       cmap="viridis", levels=50, title=f"UK {variogram} (external drift: {covariate_col})")
        print("Map saved to", outpath)
        return z, ss

    def run_regression_kriging_map(self, covariate_cols, outpath, variogram=None, base_model=None):
        """
        covariate_cols: list of column names to use in regression (must exist in train_df)
        base_model: sklearn regressor (default RandomForest)
        """
        if variogram is None:
            variogram = getattr(self, "variogram", "spherical")

        df = self.train_df.dropna(subset=covariate_cols + [self.value_col])
        X = df[covariate_cols].values
        y = df[self.value_col].values
        lons = df[self.lon_col].values
        lats = df[self.lat_col].values

        rk = RegressionKriging(base_model=base_model, variogram_model=variogram)
        rk.reg = base_model or RandomForestRegressor(n_estimators=200, random_state=0)
        rk.fit(X, y)

        # store train coords/residuals for later validation if needed
        rk.train_lons = lons
        rk.train_lats = lats
        rk.resid_train = y - rk.reg.predict(X)
        rk.resid_train = rk.resid_train

        # build grid covariates DataFrame: user must prepare grid covariates matching covariate_cols
        # For convenience, here we try to build a grid dataframe if the train_df contains lat/lon based covariates
        # But better: user should pass grid_df matching covariate_cols. We'll auto-create if possible and raise otherwise.
        # Try auto-grid: if covariate_cols include 'Latitude'/'Longitude' skip; otherwise attempt to interpolate covariates onto grid
        # For now we require user to pass covariate arrays (more robust). To keep API simple, we try to compute grid covariates with nearest neighbor from train points.

        # Make grid_df by nearest-neighbor assignment from training covariates (fast, simple)
        lon_grid, lat_grid = np.meshgrid(self.grid_lon, self.grid_lat)
        coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        grid_df = pd.DataFrame(columns=covariate_cols)
        # fill by NN
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(np.column_stack([df[self.lon_col].values, df[self.lat_col].values]))
        dists, idxs = nbrs.kneighbors(np.column_stack([coords[:,1], coords[:,0]]))
        for i, col in enumerate(covariate_cols):
            grid_df[col] = df.iloc[idxs.ravel()][col].values

        # now predict grid
        grid_vals = rk.reg.predict(grid_df.values)  # flattened
        # now krige residuals from training points to grid
        z_resid, ss = KrigingModel(variogram_model=variogram, method="ok").fit_predict_grid_ok(lons, lats, rk.resid_train, self.grid_lon, self.grid_lat)
        final = grid_vals.reshape(z_resid.shape) + z_resid

        # plotting
        coords_grid = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        masker = LandMasker()
        mask = masker.build_mask(coords_grid, bbox=self.bbox)
        plotter = MapPlotter(extent=self.extent)
        plotter.save_tricontour_on_land(coords_grid, final.ravel(), mask, outpath,
                                       cmap="viridis", levels=50, title=f"Regression-Kriging ({variogram})")
        print("Map saved to", outpath)
        # return final map and residual variance surface
        return final, ss, rk

    def cross_validate_variogram_and_report(self):
        self.clean()
        self.auto_select_variogram()
        # do LOO CV with selected variogram and report diagnostics
        lons = self.train_df[self.lon_col].values
        lats = self.train_df[self.lat_col].values
        vals = self.train_df[self.value_col].values
        preds = Validator.loo_cv_ok(lons, lats, vals, variogram_model=self.variogram)
        metrics = Validator.evaluate(vals, preds)
        print("LOO CV diagnostics:", metrics)
        return metrics

# multi_isotope_kriging.py
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# depends on previous pipeline classes: KrigingModel, KrigingPipeline, LandMasker, MapPlotter, RegressionKriging
# assume they are in the same module or importable:
# from kriging_pipeline import KrigingModel, KrigingPipeline, LandMasker, MapPlotter, RegressionKriging

# For this snippet I'll re-use the simple KrigingModel, LandMasker, MapPlotter from earlier.
# If you have them in another module, import them instead.

from pykrige.ok import OrdinaryKriging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
from shapely.geometry import Point
from shapely.prepared import prep

# ---------------------------
# Minimal KrigingModel, LandMasker, MapPlotter (reuse / adapt from earlier)
# ---------------------------
class KrigingModel:
    def __init__(self, variogram_model="spherical", method="ok"):
        self.variogram_model = variogram_model
        self.method = method

    def fit_predict_grid_ok(self, lons, lats, vals, grid_lon, grid_lat):
        OK = OrdinaryKriging(
            lons, lats, vals,
            variogram_model=self.variogram_model,
            verbose=False, enable_plotting=False
        )
        z, ss = OK.execute("grid", grid_lon, grid_lat)
        return np.array(z), np.array(ss)

class LandMasker:
    def __init__(self, resolution='10m'):
        shp = shapereader.natural_earth(resolution=resolution, category='physical', name='land')
        reader = shapereader.Reader(shp)
        self.land_geoms = [r.geometry for r in reader.records()]
        self.prepared = [prep(g) for g in self.land_geoms]

    def build_mask(self, coords, bbox=None):
        pts = [Point(lon, lat) for lat, lon in coords]
        mask = np.zeros(len(coords), dtype=bool)
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            idxs = np.where(
                (coords[:,0] >= lat_min) & (coords[:,0] <= lat_max) &
                (coords[:,1] >= lon_min) & (coords[:,1] <= lon_max)
            )[0]
        else:
            idxs = np.arange(len(coords))
        for geom in self.prepared:
            idxs_test = idxs[~mask[idxs]]
            for idx in idxs_test:
                if geom.contains(pts[idx]) or geom.touches(pts[idx]):
                    mask[idx] = True
        return mask

class MapPlotter:
    def __init__(self, extent=None):
        self.extent = extent

    @staticmethod
    def add_basemap(ax):
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')
        ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.6)
        ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
        ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.3)

    def save_tricontour_on_land(self, coords, values, land_mask, outpath, cmap='viridis', levels=50, title=None):
        land_coords = coords[land_mask]
        land_vals = values[land_mask]
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        self.add_basemap(ax)
        if self.extent:
            ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        if len(land_coords) > 0 and not np.all(np.isnan(land_vals)):
            triang = mtri.Triangulation(land_coords[:,1], land_coords[:,0])
            tcf = ax.tricontourf(triang, land_vals, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
            plt.colorbar(tcf, ax=ax, label='Value')
        else:
            ax.set_title("No data")
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()

# ---------------------------
# MultiIsotopeAnalyzer
# ---------------------------
class MultiIsotopeAnalyzer:
    def __init__(self, train_df, lon_col="Longitude", lat_col="Latitude",
                 grid_lon=None, grid_lat=None, bbox=None, extent=None,
                 variogram="spherical", var_floor=1e-6):
        """
        train_df: DataFrame with training samples (contains lon_col, lat_col, and isotope columns)
        grid_lon, grid_lat: 1D arrays defining interpolation grid (if None, auto from training bbox)
        bbox: (lat_min, lat_max, lon_min, lon_max) used for land mask preselection
        extent: plotting extent [lon_min, lon_max, lat_min, lat_max]
        variogram: default variogram model for OK
        var_floor: minimum variance added to kriging var to avoid zeros
        """
        self.df = train_df.copy()
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.grid_lon = grid_lon
        self.grid_lat = grid_lat
        self.bbox = bbox
        self.extent = extent
        self.variogram = variogram
        self.var_floor = var_floor
        # auto grid if missing
        if self.grid_lon is None or self.grid_lat is None:
            lon_min, lon_max = self.df[lon_col].min(), self.df[lon_col].max()
            lat_min, lat_max = self.df[lat_col].min(), self.df[lat_col].max()
            # default resolution 150
            res = 150
            self.grid_lon = np.linspace(lon_min, lon_max, res)
            self.grid_lat = np.linspace(lat_min, lat_max, res)
        # prepare meshgrid coords
        lon_grid, lat_grid = np.meshgrid(self.grid_lon, self.grid_lat)
        self.grid_mesh_lon = lon_grid
        self.grid_mesh_lat = lat_grid
        self.grid_coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        self.landmasker = LandMasker()
        self.plotter = MapPlotter(extent=self.extent)

    def clean(self):
        df = self.df.copy()
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.replace("NA", np.nan)
        self.df = df

    def build_isoscape_ok(self, isotope_col, dropna=True):
        """
        Ordinary kriging isoscape (mean + variance) from training points only.
        Returns (mean_2D, var_2D)
        """
        df = self.df.copy()
        if dropna:
            df = df.dropna(subset=[self.lon_col, self.lat_col, isotope_col])
        lons = df[self.lon_col].values
        lats = df[self.lat_col].values
        vals = df[isotope_col].values
        if len(vals) < 3:
            raise ValueError("Need at least 3 training points for kriging")
        km = KrigingModel(variogram_model=self.variogram, method="ok")
        z, ss = km.fit_predict_grid_ok(lons, lats, vals, self.grid_lon, self.grid_lat)
        # ensure non-negative variance & add floor
        ss = np.maximum(ss, 0.0) + self.var_floor
        return np.array(z), np.array(ss)

    def build_isoscapes_for_isotopes(self, isotope_cols):
        """
        Build isoscape mean & var for each isotope in isotope_cols.
        Returns dict: {isotope: {"mean": 2D array, "var": 2D array}}
        """
        iso_maps = {}
        for iso in isotope_cols:
            print(f"Building isoscape for {iso} ...")
            mean2d, var2d = self.build_isoscape_ok(iso)
            iso_maps[iso] = {"mean": mean2d, "var": var2d}
        return iso_maps

    def sample_probability_map(self, sample_row, iso_maps=None, obs_sigma=None, normalize=True):
        """
        Compute per-isotope probability maps and joint probability map for a sample row (pd.Series or DataFrame row).
        sample_row: pd.Series or DataFrame row with isotope columns
        iso_maps: precomputed dict from build_isoscapes_for_isotopes; if None, will compute for columns in sample_row
        obs_sigma: dict of observation SDs per isotope (optional). If None assume 0.
        normalize: normalize probabilities to sum=1
        Returns:
          per_iso_probs: {iso: prob_flat_array (same length as grid_coords)}
          joint_prob: prob_flat_array (product of per-iso likelihoods normalized)
          per_iso_likelihood_grids: {iso: 2D lik grid}
        """
        # Convert sample_row to dict for isotope values
        if isinstance(sample_row, pd.Series):
            sample_dict = sample_row.to_dict()
        elif isinstance(sample_row, pd.DataFrame):
            sample_dict = sample_row.iloc[0].to_dict()
        else:
            raise ValueError("sample_row must be a pandas Series or DataFrame row")

        if iso_maps is None:
            iso_maps = self.build_isoscapes_for_isotopes(list(sample_dict.keys()))

        n_cells = self.grid_coords.shape[0]
        per_iso_probs = {}
        per_iso_likelihoods = {}
        for iso in sample_dict:
            val = sample_dict[iso]
            if iso not in iso_maps:
                raise ValueError(f"isotope {iso} not in iso_maps")
            mean = iso_maps[iso]["mean"].ravel()
            var = iso_maps[iso]["var"].ravel()
            obs_var = 0.0
            if obs_sigma and iso in obs_sigma and obs_sigma[iso] is not None:
                obs_var = obs_sigma[iso] ** 2
            total_var = var + obs_var + self.var_floor
            total_std = np.sqrt(total_var)
            lik = norm.pdf(val, loc=mean, scale=total_std)
            lik = np.where(np.isnan(mean), 0.0, lik)
            per_iso_likelihoods[iso] = lik
            prob = lik / np.sum(lik) if lik.sum() != 0 else np.zeros_like(lik)
            per_iso_probs[iso] = prob
        joint_lik = np.ones(n_cells)
        for iso in per_iso_likelihoods:
            joint_lik *= per_iso_likelihoods[iso]
        joint_lik = np.where(np.isnan(joint_lik), 0.0, joint_lik)
        joint_prob = joint_lik / np.sum(joint_lik) if normalize and joint_lik.sum() != 0 else joint_lik
        per_iso_likelihood_grids = {iso: per_iso_likelihoods[iso].reshape(self.grid_mesh_lat.shape) for iso in per_iso_likelihoods}
        per_iso_prob_grids = {iso: per_iso_probs[iso].reshape(self.grid_mesh_lat.shape) for iso in per_iso_probs}
        joint_prob_grid = joint_prob.reshape(self.grid_mesh_lat.shape)
        return per_iso_prob_grids, joint_prob_grid, iso_maps

    def save_probability_maps(self, per_iso_prob_grids, joint_prob_grid, outdir="output", prefix="prob", cmap='viridis'):
        """
        Save per-isotope and joint probability maps (tri-contour on land).
        """
        coords = self.grid_coords  # flattened [lat,lon]
        mask = self.landmasker.build_mask(coords, bbox=self.bbox)
        # per isotope
        for iso, grid in per_iso_prob_grids.items():
            self.plotter.save_tricontour_on_land(coords, grid.ravel(), mask,
                                                f"{outdir}/{prefix}_{iso}.png", cmap=cmap,
                                                levels=50, title=f"Probability map: {iso}")
            print("Saved", f"{outdir}/{prefix}_{iso}.png")
        # joint
        self.plotter.save_tricontour_on_land(coords, joint_prob_grid.ravel(), mask,
                                            f"{outdir}/{prefix}_joint.png", cmap=cmap,
                                            levels=50, title="Joint probability map")
        print("Saved", f"{outdir}/{prefix}_joint.png")

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # example configuration (adapt to your config)
    from config import path, GRID_RESOLUTION, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    # read training CSV (must NOT include test points)
    df_train = pd.read_csv(path)


    # specify isotopes to build isoscapes for
    isotopes = ["P77", "Li", "P52"]  # example columns present in df_train
    test_samples_df = df_train.iloc[np.random.choice(400, 5, replace=False)][['Latitude','Longitude'] + isotopes].reset_index(drop=True)

    test_candidates = df_train.dropna(subset=isotopes).reset_index(drop=True)
    test_samples_df = df_train.iloc[np.random.choice(400, 5, replace=False)][['Latitude','Longitude'] + isotopes].reset_index(drop=True)
    df_train = df_train.drop(test_samples_df.index).reset_index(drop=True)

    # build grid
    grid_lon = np.linspace(LON_MIN, LON_MAX, GRID_RESOLUTION)
    grid_lat = np.linspace(LAT_MIN, LAT_MAX, GRID_RESOLUTION)
    bbox = (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    analyzer = MultiIsotopeAnalyzer(df_train, lon_col="Longitude", lat_col="Latitude",
                                    grid_lon=grid_lon, grid_lat=grid_lat,
                                    bbox=bbox, extent=extent, variogram="spherical")
    analyzer.clean()

    # build isoscapes (means + variances) for all isotopes
    iso_maps = analyzer.build_isoscapes_for_isotopes(isotopes)

    # Suppose we have a test/unknown sample with measurements:
    
    # optional measurement uncertainties:
    test =  (test_samples_df.iloc[0]).drop(['Latitude', 'Longitude'])
    print("hoooooiiiiiiiiiiiii", test)
    per_iso_probs, joint_prob_grid, iso_maps = analyzer.sample_probability_map(test_samples_df.iloc[0], iso_maps=iso_maps, obs_sigma=None, normalize=True)

    # save maps
    analyzer.save_probability_maps(per_iso_probs, joint_prob_grid, outdir="output/kriging/OK/kriging_probs", prefix="sample1")

    # You now have:
    # - 'output/kriging_probs/sample1_P77.png', etc.
    # - 'output/kriging_probs/sample1_joint.png'
    # and the arrays per_iso_probs (2D grids) and joint_prob_grid in-memory for further analysis.



"""
# ---------------------------
# Example usage (script mode)
# ---------------------------
if __name__ == "__main__":
    # Example: adapt to your config and CSV layout
    from config import path, GRID_RESOLUTION, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

    # read full dataset
    df = pd.read_csv(path)
    # choose training set (make sure not to include test points)
    # e.g. if your full file contains separate 'split' column, use it; here we assume df is training set
    isotope = "P77"
    bbox = (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    pipeline = KrigingPipeline(df, lon_col="Longitude", lat_col="Latitude", value_col=isotope,
                               grid_res=GRID_RESOLUTION, bbox=bbox, extent=extent)

    # 1) automatic variogram select (LOO CV) and report metrics
    pipeline.clean()
    pipeline.auto_select_variogram()
    pipeline.cross_validate_variogram_and_report()

    # 2) produce ordinary kriging map (uses training data only)
    ok_out = "output/kriging/OK/map_ok.png"
    pipeline.run_ok_map(ok_out, variogram=pipeline.variogram)

    # 3) if you have a covariate raster (2D array grid matching the grid), you can run UK:
    # Example: create grid_covariate_array with shape (grid_lat_len, grid_lon_len)
    # grid_covariate_array = ... (user supplies)
    # uk_out = "output/kriging_uk.png"
    # pipeline.run_uk_map(covariate_col="altitude", grid_covariate_array=grid_covariate_array, outpath=uk_out, variogram=pipeline.variogram)

    # 4) regression-kriging (requires covariate columns in training data)
    # covariate_cols = ["altitude", "temp", "precip"]
    # rk_out = "output/kriging_regkrig.png"
    # final_map, ss, rk_model = pipeline.run_regression_kriging_map(covariate_cols, rk_out, variogram=pipeline.variogram)
"""