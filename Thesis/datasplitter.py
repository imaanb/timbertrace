import matplotlib.pyplot as plt
import pandas as pd
from config import geo_dict
def separator(data_format, mode = "Kfold"): 
    if mode == "Kfold": 
        pass

def site_id(path):
    samples = pd.read_csv(path)
    samples.columns = samples.columns.str.strip()
    ids = samples['matchname']
    groups = ids.str[:2]
    grouped_samples = samples.groupby(groups)
    group_dict = {group: list(group_df['matchname']) for group, group_df in grouped_samples}
    #print(group_dict)
    #print(grouped_samples.indices)
    return samples, grouped_samples, group_dict



def site_plotter(site_groups, group_dict):
    plt.figure(figsize=(10, 8))
    for site, group in site_groups:
        plt.scatter(group['Longitude'], group['Latitude'], label=site, alpha=0.7)
        # Plot group name at the mean location of the group
        mean_long = group['Longitude'].mean()
        mean_lat = group['Latitude'].mean()
        plt.text(mean_long, mean_lat, str(site), fontsize=10, weight='bold', ha='center', va='center')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Sample Sites Map')
    plt.legend()
    plt.show()

def site_plotter_with_index(site_groups):
    plt.figure(figsize=(10, 8))
    for site, group in site_groups:
        plt.scatter(group['Longitude'], group['Latitude'], label=site, alpha=0.7)
        for idx, row in group.iterrows():
            plt.text(row['Longitude'], row['Latitude'], str(idx), fontsize=8, ha='right', va='bottom')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Sample Sites Map with Indices')
    plt.legend()
    plt.grid(True)

    plt.show()



def get_extreme_sites(group_dict, samples, k=1):
    # Merge group_dict into a DataFrame for easy lookup
    site_info = []
    for site, names in group_dict.items():
        group_samples = samples[samples['matchname'].isin(names)]
        # Use mean coordinates for each site
        mean_lat = group_samples['Latitude'].mean()
        mean_long = group_samples['Longitude'].mean()
        site_info.append({'site': site, 'mean_lat': mean_lat, 'mean_long': mean_long})
    site_df = pd.DataFrame(site_info)

    northern = site_df.nlargest(k, 'mean_lat')
    southern = site_df.nsmallest(k, 'mean_lat')
    eastern = site_df.nlargest(k, 'mean_long')
    western = site_df.nsmallest(k, 'mean_long')

    return {
        'northern': northern['site'].tolist(),
        'southern': southern['site'].tolist(),
        'eastern': eastern['site'].tolist(),
        'western': western['site'].tolist()
    }

def get_central_and_peripheral_sites(group_dict, samples, k=1):
    # Calculate mean coordinates for each site
    site_info = []
    for site, names in group_dict.items():
        group_samples = samples[samples['matchname'].isin(names)]
        mean_lat = group_samples['Latitude'].mean()
        mean_long = group_samples['Longitude'].mean()
        site_info.append({'site': site, 'mean_lat': mean_lat, 'mean_long': mean_long})
    site_df = pd.DataFrame(site_info)

    # Calculate centroid of all sites
    centroid_lat = site_df['mean_lat'].mean()
    centroid_long = site_df['mean_long'].mean()

    # Calculate Euclidean distance from centroid for each site
    site_df['distance_to_centroid'] = ((site_df['mean_lat'] - centroid_lat)**2 + (site_df['mean_long'] - centroid_long)**2)**0.5

    # Find k most central (smallest distance) and k most peripheral (largest distance) sites
    central_sites = site_df.nsmallest(k, 'distance_to_centroid')['site'].tolist()
    peripheral_sites = site_df.nlargest(k, 'distance_to_centroid')['site'].tolist()

    return {'central': central_sites, 'peripheral': peripheral_sites}




mode = "Kfold" # GCentral-Fold, GWest-Fold, GEastfold, GNorth-Fold, G 
data_format = ['iso'] # 'gen', 'chem' 
path =   'data/presave.csv'




samples, groups, group_dict = site_id(path)
site_plotter(groups, group_dict)
print(get_extreme_sites(group_dict, samples, 6))
print(get_central_and_peripheral_sites(group_dict, samples, 6))

