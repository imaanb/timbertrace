import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd 


###############################################################
# 1. GENETIC DISTANCE (your formula)
###############################################################

def genic_distance_matrix(test_gen, ref_gen, loci_counts):
    T = test_gen.shape[0]
    R = ref_gen.shape[0]
    M = len(loci_counts)

    D = np.zeros((T, R))
    col = 0

    for N_m in loci_counts:
        cols = slice(col, col + N_m)
        p_test = test_gen[:, cols]
        p_ref = ref_gen[:, cols]

        diff = np.abs(p_test[:, None, :] - p_ref[None, :, :]).sum(axis=2)
        D += 0.5 * diff
        col += N_m

    return D / M


###############################################################
# 2. STANDARDIZED (Z-SCORE) EUCLIDEAN DISTANCES
###############################################################

def standardized_distance(test_feat, ref_feat, scaler=None):
    if test_feat is None or ref_feat is None:
        return None, None

    if scaler is None:
        scaler = StandardScaler().fit(np.vstack([test_feat, ref_feat]))

    test_s = scaler.transform(test_feat)
    ref_s = scaler.transform(ref_feat)

    diff = test_s[:, None, :] - ref_s[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    return dist, scaler


def minmax_scale(D):
    return (D - D.min()) / (D.max() - D.min() + 1e-12)


###############################################################
# 3. COMBINE DISTANCES
###############################################################

def combine_distances(D_gen=None, D_iso=None, D_chem=None, weights=None):
    if weights is None:
        weights = {}

    D_list = []
    W_list = []

    if D_gen is not None:
        D_list.append(D_gen)
        W_list.append(weights.get("gen", 1.0))

    if D_iso is not None:
        D_list.append(minmax_scale(D_iso))
        W_list.append(weights.get("iso", 1.0))

    if D_chem is not None:
        D_list.append(minmax_scale(D_chem))
        W_list.append(weights.get("chem", 1.0))

    W_sum = sum(W_list)

    D_final = sum(w * D for w, D in zip(W_list, D_list)) / W_sum
    return D_final


###############################################################
# 4. KNN REGRESSION FOR LAT/LON
###############################################################

def knn_regress(D, y_ref, k, weighted=True):
    T = D.shape[0]
    preds = np.zeros((T, y_ref.shape[1]))

    for i in range(T):
        idx = np.argsort(D[i])[:k]
        dist = D[i, idx]
        Y = y_ref[idx]

        if weighted:
            w = 1 / (dist + 1e-6)
            preds[i] = (w[:, None] * Y).sum(axis=0) / w.sum()

        else:
            preds[i] = Y.mean(axis=0)

    return preds


###############################################################
# 5. CROSS-VALIDATION TO CHOOSE BEST k
###############################################################

def choose_best_k(D, y_ref, k_values, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    results = {}

    for k in k_values:
        mse_lat = []
        mse_lon = []

        for train_idx, val_idx in kf.split(D):

            D_val = D[val_idx][:, train_idx]
            D_train = D[train_idx][:, train_idx]  # unused but kept for clarity

            y_train = y_ref[train_idx]
            y_val = y_ref[val_idx]

            preds = knn_regress(D_val, y_train, k=k)

            mse_lat.append(mean_squared_error(y_val[:,0], preds[:,0]))
            mse_lon.append(mean_squared_error(y_val[:,1], preds[:,1]))

        results[k] = (np.mean(mse_lat), np.mean(mse_lon))

    best_k = min(results, key=lambda k: results[k][0] + results[k][1])
    return best_k, results


###############################################################
# 6. FULL PIPELINE
###############################################################

def geo_knn_pipeline(
    test_gen=None, ref_gen=None, loci_counts=None,
    test_iso=None, ref_iso=None,
    test_chem=None, ref_chem=None,
    lat_ref=None, lon_ref=None,
    k_values=range(3, 51, 2),
    weights={"gen":1.0, "iso":1.0, "chem":1.0}
):
    y_ref = np.vstack([lat_ref, lon_ref]).T   # shape (R,2)

    # Compute distances (optional)
    D_gen = genic_distance_matrix(test_gen, ref_gen, loci_counts) if test_gen is not None else None
    D_iso, sc_iso = standardized_distance(test_iso, ref_iso) if test_iso is not None else (None, None)
    D_chem, sc_chem = standardized_distance(test_chem, ref_chem) if test_chem is not None else (None, None)

    # Combine distances
    D_all = combine_distances(D_gen, D_iso, D_chem, weights)

    # Cross-validate
    best_k, cv_results = choose_best_k(D_all, y_ref, k_values)

    # Final predictions
    preds = knn_regress(D_all, y_ref, k=best_k)

    return preds, best_k, cv_results, {
        "D_gen": D_gen,
        "D_iso": D_iso,
        "D_chem": D_chem,
        "D_all": D_all,
        "scalers": {"iso": sc_iso, "chem": sc_chem}
    }


preds, best_k, cv_results, debug = geo_knn_pipeline(
    test_gen=my_test_gen,
    ref_gen=my_ref_gen,
    loci_counts=my_loci_counts,

    test_iso=my_test_iso,
    ref_iso=my_ref_iso,

    test_chem=my_test_chem,
    ref_chem=my_ref_chem,

    lat_ref=ref_latitude,
    lon_ref=ref_longitude,

    k_values=range(3, 51, 2),  # try odd k from 3 to 49

    weights={"gen":0.7, "iso":0.2, "chem":0.1}  # customize any combination
)


gen_df = pd.read_csv("data/presave.csv")