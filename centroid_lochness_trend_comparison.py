import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# load summary CSV
summary = pd.read_csv('/scratch/easmit31/factor_analysis/GABAergic_HIP_centroid_vs_lochness.csv')
summary['louvain'] = summary['louvain'].astype(str)

# classify louvains
def classify(row):
    sc, sl = row['sig_centroid'], row['sig_lochness']
    rc, rl = row['r_centroid'], row['r_lochness']
    if sc and sl:
        if np.sign(rc) == np.sign(rl):
            return 'both_same_direction'
        else:
            return 'both_opposite_direction'
    elif sc:
        return 'centroid_only'
    elif sl:
        return 'lochness_only'
    else:
        return 'neither'

summary['category'] = summary.apply(classify, axis=1)

# louvains to plot - sig for at least one
to_plot = summary[summary['category'] != 'neither'].sort_values('category')
print(f"Plotting {len(to_plot)} louvains")

# precompute agg data per louvain
agg_data = {}
for cl in to_plot['louvain']:
    mask = region_mask & (louvain == cl)
    cell_indices = np.where(mask)[0]

    # centroid
    pca_subset = X_pca[mask]
    centroid = pca_subset.mean(axis=0)
    distances = np.sqrt(((pca_subset - centroid) ** 2).sum(axis=1))
    df_c = pd.DataFrame({'animal_id': animal_ids[mask], 'age': ages[mask], 'dist': distances})
    agg_c = df_c.groupby(['animal_id', 'age'])['dist'].mean().reset_index().sort_values('age')

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
    agg_l = df_l.groupby(['animal_id', 'age'])['lochness'].mean().reset_index().sort_values('age')

    agg_data[cl] = {'centroid': agg_c, 'lochness': agg_l}

# color per category
cat_colors = {
    'both_same_direction': 'green',
    'both_opposite_direction': 'purple',
    'centroid_only': 'dodgerblue',
    'lochness_only': 'orange'
}
cat_labels = {
    'both_same_direction': 'Both sig, same direction',
    'both_opposite_direction': 'Both sig, opposite direction',
    'centroid_only': 'Centroid only',
    'lochness_only': 'lochNESS only'
}

n = len(to_plot)
fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
if n == 1:
    axes = axes[reshape(1, 2)]

for i, (_, row) in enumerate(to_plot.iterrows()):
    cl = row['louvain']
    cat = row['category']
    color = cat_colors[cat]
    agg_c = agg_data[cl]['centroid']
    agg_l = agg_data[cl]['lochness']

    for j, (agg, ylabel, r, p) in enumerate([
        (agg_c, 'Mean dist to centroid', row['r_centroid'], row['p_centroid']),
        (agg_l, 'Mean lochNESS', row['r_lochness'], row['p_lochness'])
    ]):
        ax = axes[i, j]
        sl, ic, _, _, _ = stats.linregress(agg['age'], agg.iloc[:, 2])
        x_line = np.linspace(agg['age'].min(), agg['age'].max(), 100)
        ax.scatter(agg['age'], agg.iloc[:, 2], color=color, s=20, alpha=0.7)
        ax.plot(x_line, sl * x_line + ic, color='black', lw=1.5,
                ls='-' if p < 0.05 else '--')
        ax.set_xlabel('Age')
        ax.set_ylabel(ylabel)
        ax.set_title(f'L{cl} [{cat_labels[cat]}]\nr={r:.2f}, p={p:.3f}', fontsize=8)

plt.suptitle('GABAergic HIP - centroid distance vs lochNESS age trends', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig('/scratch/easmit31/factor_analysis/GABAergic_HIP_trend_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Done.")
