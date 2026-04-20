import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
import matplotlib.pyplot as plt

fpath = '/data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad'
region = "HIP"
n_pcs = 30

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

print("Loading data...")
with h5py.File(fpath, 'r') as f:
    X_pca = f['obsm']['X_pca'][:, :n_pcs]
    regions = decode_categorical(f['obs']['region'])
    louvain = decode_categorical(f['obs']['louvain'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages = f['obs']['age'][:]
    knn_data = f['obsp']['distances']['data'][:]
    knn_indices = f['obsp']['distances']['indices'][:]
    knn_indptr = f['obsp']['distances']['indptr'][:]
    n_cells = len(knn_indptr) - 1

knn = sp.csr_matrix((knn_data, knn_indices, knn_indptr), shape=(n_cells, n_cells))
region_mask = (regions == region) & (ages >= 1)
louvain_clusters = np.unique(louvain[region_mask])

centroid_rows = []
lochness_rows = []

for cl in louvain_clusters:
    mask = region_mask & (louvain == cl)
    cell_indices = np.where(mask)[0]
    if len(cell_indices) < 10:
        continue

    # centroid distance
    pca_subset = X_pca[mask]
    centroid = pca_subset.mean(axis=0)
    distances = np.sqrt(((pca_subset - centroid) ** 2).sum(axis=1))
    df_c = pd.DataFrame({'animal_id': animal_ids[mask], 'age': ages[mask], 'dist': distances})
    agg_c = df_c.groupby(['animal_id', 'age'])['dist'].mean().reset_index()
    if len(agg_c) >= 5:
        slope, intercept, r, p, _ = stats.linregress(agg_c['age'], agg_c['dist'])
        centroid_rows.append({'louvain': cl, 'n_cells': len(cell_indices),
                              'n_animals': len(agg_c), 'r_centroid': round(r, 3),
                              'p_centroid': round(p, 4)})

    # lochness
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
    df_l = pd.DataFrame(rows)
    agg_l = df_l.groupby(['animal_id', 'age'])['lochness'].mean().reset_index()
    if len(agg_l) >= 5:
        slope, intercept, r, p, _ = stats.linregress(agg_l['age'], agg_l['lochness'])
        lochness_rows.append({'louvain': cl, 'r_lochness': round(r, 3),
                              'p_lochness': round(p, 4)})

centroid_df = pd.DataFrame(centroid_rows)
lochness_df = pd.DataFrame(lochness_rows)

centroid_df['louvain'] = centroid_df['louvain'].astype(str)
lochness_df['louvain'] = lochness_df['louvain'].astype(str)

merged = pd.merge(centroid_df, lochness_df, on='louvain')
merged['sig_centroid'] = merged['p_centroid'] < 0.05
merged['sig_lochness'] = merged['p_lochness'] < 0.05
merged = merged.sort_values('louvain')

# save CSV
out = '/scratch/easmit31/factor_analysis/GABAergic_HIP_centroid_vs_lochness.csv'
merged.to_csv(out, index=False)
print(f"Saved to {out}")

# print table
print("\n=== Side by side comparison ===")
print(merged[['louvain', 'n_cells', 'n_animals', 'r_centroid', 'p_centroid',
              'r_lochness', 'p_lochness', 'sig_centroid', 'sig_lochness']].to_string(index=False))

# scatter plot
fig, ax = plt.subplots(figsize=(7, 6))
colors = ['red' if sl and sc else ('orange' if sl else ('dodgerblue' if sc else 'gray'))
          for sl, sc in zip(merged['sig_lochness'], merged['sig_centroid'])]
ax.scatter(merged['r_centroid'], merged['r_lochness'], c=colors,
           s=merged['n_cells'] / 30, alpha=0.7)
for _, row in merged.iterrows():
    if row['sig_lochness'] or row['sig_centroid']:
        ax.annotate(f"L{row['louvain']}", (row['r_centroid'], row['r_lochness']),
                    fontsize=7, ha='left', va='bottom')
ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.axvline(0, color='gray', lw=0.5, ls='--')
ax.set_xlabel('r (centroid distance ~ age)')
ax.set_ylabel('r (lochNESS ~ age)')
ax.set_title('GABAergic HIP - centroid distance vs lochNESS\nred=both sig, orange=lochNESS only, blue=centroid only')
from matplotlib.lines import Line2D
legend = [Line2D([0],[0], marker='o', color='w', markerfacecolor='red', label='both sig'),
          Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', label='lochNESS only'),
          Line2D([0],[0], marker='o', color='w', markerfacecolor='dodgerblue', label='centroid only'),
          Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', label='neither')]
ax.legend(handles=legend)
plt.tight_layout()
plt.savefig('/scratch/easmit31/factor_analysis/GABAergic_HIP_centroid_vs_lochness_scatter.png', dpi=150)
plt.close()
print("Done.")
