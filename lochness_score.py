import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

fpath = '/data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad'
region = "HIP"
n_pcs = 30

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

results = []

for cl in louvain_clusters:
    mask = region_mask & (louvain == cl)
    cell_indices = np.where(mask)[0]
    if len(cell_indices) < 10:
        continue

    animals_in_cluster = animal_ids[cell_indices]
    ages_in_cluster = ages[cell_indices]
    unique_animals = np.unique(animals_in_cluster)

    # expected fraction per donor = their proportion of cells in cluster
    donor_props = {a: (animals_in_cluster == a).mean() for a in unique_animals}

    lochness_per_cell = []
    cell_animal = []
    cell_age = []

    for idx in cell_indices:
        row = knn.getrow(idx)
        neighbor_idx = row.indices
        # filter to neighbors also in this cluster
        neighbor_in_cluster = neighbor_idx[mask[neighbor_idx]]
        if len(neighbor_in_cluster) == 0:
            continue
        donor = animal_ids[idx]
        obs_frac = (animal_ids[neighbor_in_cluster] == donor).mean()
        exp_frac = donor_props[donor]
        score = obs_frac / exp_frac if exp_frac > 0 else np.nan
        lochness_per_cell.append(score)
        cell_animal.append(donor)
        cell_age.append(ages[idx])

    df = pd.DataFrame({
        'animal_id': cell_animal,
        'age': cell_age,
        'lochness': lochness_per_cell
    })

    # aggregate per donor
    agg = df.groupby(['animal_id', 'age'])['lochness'].mean().reset_index()
    agg = agg.sort_values('age')

    if len(agg) < 5:
        continue

    slope, intercept, r, p, _ = stats.linregress(agg['age'], agg['lochness'])
    results.append({
        'louvain': cl,
        'n_cells': len(cell_indices),
        'n_animals': len(agg),
        'mean_lochness': agg['lochness'].mean(),
        'r': round(r, 3),
        'p': round(p, 4),
        'slope': round(slope, 5)
    })
    print(f"  louvain {cl}: n_cells={len(cell_indices)}, n_animals={len(agg)}, mean_lochness={agg['lochness'].mean():.3f}, r={r:.2f}, p={p:.3f}")

results_df = pd.DataFrame(results).sort_values('p')
out = f'/scratch/easmit31/factor_analysis/lochness_GABAergic_{region}.csv'
results_df.to_csv(out, index=False)
print(f"\nSaved to {out}")
print("\nTop hits (p < 0.05):")
print(results_df[results_df['p'] < 0.05].to_string())
