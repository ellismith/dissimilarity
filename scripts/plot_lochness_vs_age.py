import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
import matplotlib.pyplot as plt
import math

fpath = '/data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad'
region = "HIP"

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
    n_cells = len(indptr) - 1

knn = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))

region_mask = (regions == region) & (ages >= 1)
clusters = np.unique(louvain[region_mask])

ncols = 5
nrows = math.ceil(len(clusters) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
axes = axes.flatten()

for i, cl in enumerate(clusters):
    mask = region_mask & (louvain == cl)
    cell_indices = np.where(mask)[0]
    if len(cell_indices) < 20:
        continue

    animals_in_cluster = animal_ids[cell_indices]
    unique_animals = np.unique(animals_in_cluster)
    donor_props = {a: (animals_in_cluster == a).mean() for a in unique_animals}

    records = []

    for idx in cell_indices:
        row = knn.getrow(idx)
        neighbor_idx = row.indices
        neighbor_in_cluster = neighbor_idx[mask[neighbor_idx]]
        if len(neighbor_in_cluster) == 0:
            continue

        donor = animal_ids[idx]
        obs_frac = (animal_ids[neighbor_in_cluster] == donor).mean()
        exp_frac = donor_props[donor]
        if exp_frac == 0:
            continue

        score = obs_frac / exp_frac
        records.append((donor, ages[idx], score))

    df = pd.DataFrame(records, columns=["donor", "age", "lochness"])
    agg = df.groupby(["donor", "age"])["lochness"].mean().reset_index()

    ax = axes[i]
    ax.scatter(agg["age"], agg["lochness"], alpha=0.7)

    if len(agg) > 5:
        slope, intercept, r, p, _ = stats.linregress(agg["age"], agg["lochness"])
        xs = np.linspace(agg["age"].min(), agg["age"].max(), 100)
        ax.plot(xs, intercept + slope*xs)
        ax.set_title(f"Cl {cl} (r={r:.2f}, p={p:.3f})")
    else:
        ax.set_title(f"Cl {cl}")

    ax.set_xlabel("Age")
    ax.set_ylabel("Mean lochness")

plt.tight_layout()
out = "/scratch/easmit31/factor_analysis/lochness_vs_age_all_clusters.png"
plt.savefig(out, dpi=300)
print(f"Saved to {out}")
