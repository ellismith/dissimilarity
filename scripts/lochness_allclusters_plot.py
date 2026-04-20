import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fpath = '/data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad'
region = "HIP"
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
cluster_labels = np.unique(louvain[region_mask])

# assign colors to clusters
cmap = cm.get_cmap('tab20', len(cluster_labels))
cluster_colors = {cl: cmap(i) for i, cl in enumerate(cluster_labels)}

plt.figure(figsize=(12,6))

for cl in cluster_labels:
    cluster_mask = region_mask & (louvain == cl)
    cell_indices = np.where(cluster_mask)[0]
    if len(cell_indices) == 0:
        continue

    animals_in_cluster = animal_ids[cell_indices]
    unique_animals = np.unique(animals_in_cluster)
    donor_props = {a: (animals_in_cluster == a).mean() for a in unique_animals}

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
        score_diff = obs_frac - exp_frac
        lochness_diff.append(score_diff)
        cell_donor_counts.append((animals_in_cluster == donor).sum())

    plt.scatter(cell_donor_counts, lochness_diff, alpha=0.5, color=cluster_colors[cl], label=f'cluster {cl}')

plt.xlabel("Donor cells in cluster")
plt.ylabel("DiffLochness (obs - exp)")
plt.title(f"DiffLochness across {region} clusters")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()

out_file = "/scratch/easmit31/factor_analysis/lochness_diff_allclusters.png"
plt.savefig(out_file, dpi=150)
print(f"Saved figure to {out_file}")
