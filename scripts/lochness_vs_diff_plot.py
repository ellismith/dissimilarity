import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

fpath = '/data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad'
region = "HIP"
cluster_to_plot = "14"  # example
min_age = 1

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

print("Loading data...")
with h5py.File(fpath, 'r') as f:
    regions = decode_categorical(f['obs']['region'])
    louvain = decode_categorical(f['obs']['louvain'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages = f['obs']['age'][:]
    data = f['obsp']['distances']['data'][:]
    indices = f['obsp']['distances']['indices'][:]
    indptr = f['obsp']['distances']['indptr'][:]

knn = sp.csr_matrix((data, indices, indptr), shape=(len(indptr)-1, len(indptr)-1))

region_mask = (regions == region) & (ages >= min_age)
cluster_mask = region_mask & (louvain == cluster_to_plot)
cell_indices = np.where(cluster_mask)[0]

animals_in_cluster = animal_ids[cell_indices]
unique_animals = np.unique(animals_in_cluster)

# donor proportions
donor_props = {a: (animals_in_cluster == a).mean() for a in unique_animals}

lochness_ratio = []
lochness_diff = []
cell_donor_counts = []

for idx in cell_indices:
    row = knn.getrow(idx)
    neighbors = row.indices
    neighbors_in_cluster = neighbors[cluster_mask[neighbors]]
    if len(neighbors_in_cluster) == 0:
        continue
    donor = animal_ids[idx]
    obs_frac = (animal_ids[neighbors_in_cluster] == donor).mean()
    exp_frac = donor_props[donor]
    # original ratio metric
    score_ratio = obs_frac / exp_frac if exp_frac > 0 else np.nan
    # difference metric
    score_diff = obs_frac - exp_frac
    lochness_ratio.append(score_ratio)
    lochness_diff.append(score_diff)
    cell_donor_counts.append((animals_in_cluster == donor).sum())

# plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(cell_donor_counts, lochness_ratio, alpha=0.5)
plt.xlabel("Donor cells in cluster")
plt.ylabel("Lochness (obs/exp)")
plt.title("Original ratio metric")
plt.yscale('log')  # log to emphasize explosion

plt.subplot(1,2,2)
plt.scatter(cell_donor_counts, lochness_diff, alpha=0.5, color='orange')
plt.xlabel("Donor cells in cluster")
plt.ylabel("DiffLochness (obs - exp)")
plt.title("Difference metric")

plt.tight_layout()
plt.show()
