import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

fpath = '/data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad'
region = "HIP"
louvain_cluster = "19"  # change this

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
    X_umap = f['obsm']['X_umap'][:]
    data = f['obsp']['distances']['data'][:]
    indices = f['obsp']['distances']['indices'][:]
    indptr = f['obsp']['distances']['indptr'][:]
    n_cells = len(indptr) - 1

knn = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))

mask = (regions == region) & (louvain == louvain_cluster) & (ages >= 1)
cell_indices = np.where(mask)[0]
N = len(cell_indices)
print(f"Cells in louvain {louvain_cluster}: {N}")

animals_in_cluster = animal_ids[cell_indices]
donor_props = {a: (animals_in_cluster == a).mean() for a in np.unique(animals_in_cluster)}

lochness_scores = np.full(n_cells, np.nan)
for idx in cell_indices:
    row = knn.getrow(idx)
    neighbor_idx = row.indices
    neighbor_in_cluster = neighbor_idx[mask[neighbor_idx]]
    if len(neighbor_in_cluster) == 0:
        continue
    donor = animal_ids[idx]
    obs_frac = (animal_ids[neighbor_in_cluster] == donor).mean()
    exp_frac = donor_props[donor]
    lochness_scores[idx] = obs_frac / exp_frac if exp_frac > 0 else np.nan

umap_subset = X_umap[cell_indices]
scores_subset = lochness_scores[cell_indices]
ages_subset = ages[cell_indices]
vmax = np.nanpercentile(scores_subset, 95)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc = axes[0].scatter(umap_subset[:, 0], umap_subset[:, 1],
                     c=scores_subset, cmap='RdBu_r',
                     vmin=0, vmax=vmax, s=5, alpha=0.7, rasterized=True)
plt.colorbar(sc, ax=axes[0], label='lochNESS')
axes[0].set_title(f'GABAergic {region} louvain {louvain_cluster} - lochNESS')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')

sc2 = axes[1].scatter(umap_subset[:, 0], umap_subset[:, 1],
                      c=ages_subset, cmap='plasma',
                      s=5, alpha=0.7, rasterized=True)
plt.colorbar(sc2, ax=axes[1], label='age')
axes[1].set_title(f'GABAergic {region} louvain {louvain_cluster} - age')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')

plt.tight_layout()
plt.savefig(f'/scratch/easmit31/factor_analysis/GABAergic_{region}_louvain{louvain_cluster}_lochness_umap.png', dpi=150)
plt.close()
print("Done.")
