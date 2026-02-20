import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy import stats

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
louvain_clusters = np.unique(louvain[region_mask])
print(f"Found {len(louvain_clusters)} louvain clusters in {region}")

for louvain_cluster in louvain_clusters:
    mask = region_mask & (louvain == louvain_cluster)
    cell_indices = np.where(mask)[0]
    if len(cell_indices) < 10:
        continue

    animals_in_cluster = animal_ids[cell_indices]
    donor_props = {a: (animals_in_cluster == a).mean() for a in np.unique(animals_in_cluster)}

    rows = []
    for idx in cell_indices:
        row = knn.getrow(idx)
        neighbor_in_cluster = row.indices[mask[row.indices]]
        if len(neighbor_in_cluster) == 0:
            continue
        donor = animal_ids[idx]
        obs_frac = (animal_ids[neighbor_in_cluster] == donor).mean()
        exp_frac = donor_props[donor]
        rows.append({'animal_id': donor, 'age': ages[idx],
                     'lochness': obs_frac / exp_frac if exp_frac > 0 else np.nan})

    df = pd.DataFrame(rows)
    agg = df.groupby(['animal_id', 'age'])['lochness'].max().reset_index().sort_values('age')

    if len(agg) < 5:
        continue

    slope, intercept, r, p, _ = stats.linregress(agg['age'], agg['lochness'])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(agg['age'], agg['lochness'], color='steelblue', s=40)
    x_line = np.linspace(agg['age'].min(), agg['age'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color='firebrick', lw=1.5)
    ax.set_xlabel('Age')
    ax.set_ylabel('Max lochNESS')
    ax.set_title(f'GABAergic {region} louvain {louvain_cluster}\nr={r:.2f}, p={p:.3f}, n={len(agg)} animals')
    plt.tight_layout()
    plt.savefig(f'/scratch/easmit31/factor_analysis/GABAergic_{region}_louvain{louvain_cluster}_lochness_max_age.png', dpi=150)
    plt.close()
    print(f"  louvain {louvain_cluster}: n_cells={len(cell_indices)}, n_animals={len(agg)}, r={r:.2f}, p={p:.3f}")

print("Done.")
