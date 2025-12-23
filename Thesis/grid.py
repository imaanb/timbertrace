import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
import requests
import tempfile
import os
import ee
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
from config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, GRID_RESOLUTION

# Define Congo Basin bounding box (approximate)
CONGO_BASIN_BOUNDS = {
    'min_lon': LON_MIN,
    'max_lon': LON_MAX,
    'min_lat': LAT_MIN,
    'max_lat': LAT_MAX
}
def authenticate_and_initialize(project_name):
    ee.Authenticate()
    ee.Initialize(project=project_name)

def create_congo_basin_grid(resolution):
    """
    Create a grid over the Congo Basin with specified resolution (number of cells in lat/lon)
    
    Parameters:
    resolution: tuple (n_lat, n_lon) specifying grid shape
    """

    n_lat = resolution
    n_lon = resolution
    lons = np.linspace(CONGO_BASIN_BOUNDS['min_lon'], CONGO_BASIN_BOUNDS['max_lon'], n_lon)
    lats = np.linspace(CONGO_BASIN_BOUNDS['min_lat'], CONGO_BASIN_BOUNDS['max_lat'], n_lat)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    grid_data = []
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            grid_data.append({
                'longitude': lon_grid[i, j],
                'latitude': lat_grid[i, j],
                'cell_id': f"{i}_{j}"
            })
    
    return pd.DataFrame(grid_data)

def add_elevation_data(df):
    """
    Add elevation data using Google Earth Engine
    Requires authentication with GEE
    """  
    # Load GMTED2010 dataset
    dataset = ee.Image('USGS/GMTED2010_FULL')
    elevation = dataset.select('be75')  # Mean elevation
    
    # Create points from grid coordinates
    points = []
    for idx, row in df.iterrows():
        point = ee.Geometry.Point([row['longitude'], row['latitude']])
        points.append(point)
    
    # Sample elevation at each point
    elevations = []
    for point in tqdm(points, desc="Processing elevation data"):
        try:  
            sample = elevation.sample(point, 30).first()  # 30m resolution
            elev_value = sample.get('be75').getInfo()
            elevations.append(elev_value if elev_value is not None else 0)
        except:
            elevations.append(0)  # Default if sampling fails
            
    
    df['elevation'] = elevations
  
    return df



def add_elevation_data_parallel(df, max_workers=None):
    """
    Add elevation data using Google Earth Engine in parallel
    """
    dataset = ee.Image('USGS/GMTED2010_FULL')
    elevation = dataset.select('be75')
    
    def get_elevation(row):
        try:
            point = ee.Geometry.Point([row['longitude'], row['latitude']])
            sample = elevation.sample(point, 30).first()
            elev_value = sample.get('be75').getInfo()
            return elev_value if elev_value is not None else 0
        except:
            return 0
    
    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=max_workers or min(32, len(df))) as executor:
        elevations = list(tqdm(
            executor.map(get_elevation, [row for _, row in df.iterrows()]),
            total=len(df),
            desc="Processing elevation data (parallel)"
        ))
    
    df['elevation'] = elevations
    return df

def add_worldclim_data_parallel(df, max_workers=None):
    """
    Add WorldClim data in parallel
    """
    dataset = ee.Image('WORLDCLIM/V1/BIO')
    
    bio_columns = {
        'bio01': 'annual_mean_temperature',
        'bio02': 'mean_diurnal_range',
        'bio03': 'isothermality',
        'bio04': 'temperature_seasonality',
        'bio05': 'max_temp_warmest_month',
        'bio06': 'min_temp_coldest_month',
        'bio07': 'temperature_annual_range',
        'bio08': 'mean_temp_wettest_quarter',
        'bio09': 'mean_temp_driest_quarter',
        'bio10': 'mean_temp_warmest_quarter',
        'bio11': 'mean_temp_coldest_quarter',
        'bio12': 'annual_precipitation',
        'bio13': 'precip_wettest_month',
        'bio14': 'precip_driest_month',
        'bio15': 'precipitation_seasonality',
        'bio16': 'precip_wettest_quarter',
        'bio17': 'precip_driest_quarter',
        'bio18': 'precip_warmest_quarter',
        'bio19': 'precip_coldest_quarter'
    }
    
    def get_climate_data(row):
        point = ee.Geometry.Point([row['longitude'], row['latitude']])
        try:
            sample = dataset.sample(point, 1000).first()
            return {bio_code: sample.get(bio_code).getInfo() for bio_code in bio_columns.keys()}
        except:
            return {bio_code: 0 for bio_code in bio_columns.keys()}
    
    with ThreadPoolExecutor(max_workers=max_workers or min(32, len(df))) as executor:
        variables = list(tqdm(
            executor.map(get_climate_data, [row for _, row in df.iterrows()]),
            total=len(df),
            desc="Processing climate data (parallel)"
        ))
    
    # Add columns to dataframe
    for bio_code, column_name in bio_columns.items():
        df[column_name] = [var[bio_code] for var in variables]
    
    return df

def calculate_distance_from_sea(df, coastline_file=None):
    """
    Calculate approximate distance from sea
    """
    # Simplified calculation - distance to Atlantic coast (approx at 6°S)
    atlantic_coast_lon = 12.0  # Approximate longitude of Atlantic coast
    
    def dist_to_sea(lon, lat):
        # Simple Euclidean distance to coast point
        coast_point = (atlantic_coast_lon, -6.0)
        return np.sqrt((lon - coast_point[0])**2 + (lat - coast_point[1])**2)
    
    df['distance_from_sea_km'] = df.apply(
        lambda row: dist_to_sea(row['longitude'], row['latitude']) * 111,  # Convert degrees to km
        axis=1
    )
    return df


# Parallel execution example
print("Creating Congo Basin grid...")
grid_df = create_congo_basin_grid(GRID_RESOLUTION)

authenticate_and_initialize(project_name='917336656805')

print("Adding elevation data (parallel)...")
grid_df = add_elevation_data_parallel(grid_df, max_workers=16)

print("Adding temperature data (parallel)...")
grid_df = add_worldclim_data_parallel(grid_df, max_workers=16)

print("Calculating distance from sea...")
grid_df = calculate_distance_from_sea(grid_df)

print("\nSample of grid data:")
print(grid_df.head())

grid_df.to_csv('congo_basin_grid_parallel.csv', index=False)
print(f"\nGrid saved with parallel processing: {len(grid_df)} cells")