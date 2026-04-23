"""
plot_centroid_age_scatter_top10.py

For the top 10 louvains by within-animal centroid age r,
plots per-animal mean distance to centroid vs age.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob

CENTROID_DIR = '/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100'
EMBED_DIR    = '/scratch/easmit31/factor_analysis/pca_embeddings'
OUT_DIR      = '/scratch/easmit31/factor_analysis'
N_TOP        = 10

# find top 10
rows = []
for f in glob.glob(os.path.join(CENTROID_DIR, '*_centroid_summary.csv')):
    ct = os.path.basename(f).replace('_centroid_summary.csv', '')
    df = pd.read_csv(f)
    df['cell_type'] = ct
    rows.append(df)

summary = pd.concat(rows, ignore_index=True)
summary['abs_r'] = summary['r_mean'].abs()

def has_embedding(row):
    p = os.path.join(EMBED_DIR, row['cell_type'], f"louvain{row['louvain']}_{row['region']}_pca.npy")
    return os.path.exists(p)

summary = summary[summary.apply(has_embedding, axis=1)]
top10 = summary.nlargest(N_TOP, 'abs_r')

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

for ax, (_, best) in zip(axes, top10.iterrows()):
    ct     = best['cell_type']
    region = best['region']
    louv   = str(best['louvain'])
    r_val  = best['r_mean']
    p_val  = best['p_mean']

    pca_path  = os.path.join(EMBED_DIR, ct, f'louvain{louv}_{region}_pca.npy')
    meta_path = os.path.join(EMBED_DIR, ct, f'louvain{louv}_{region}_metadata.csv')

    X    = np.load(pca_path)
    meta = pd.read_csv(meta_path).reset_index(drop=True)

    # compute per-animal centroids and distances
    animal_rows = []
    for animal in meta['animal_id'].unique():
        mask = meta['animal_id'] == animal
        idxs = np.where(mask)[0]
        if len(idxs) < 2:
            continue
        centroid = X[idxs].mean(axis=0)
        dists = np.sqrt(((X[idxs] - centroid) ** 2).sum(axis=1))
        animal_rows.append({
            'animal_id': animal,
            'age': meta.loc[mask, 'age'].iloc[0],
            'mean_dist': dists.mean(),
            'n_cells': len(idxs),
        })

    agg = pd.DataFrame(animal_rows).sort_values('age')

    slope, intercept, r, p, _ = stats.linregress(agg['age'], agg['mean_dist'])
    x_line = np.linspace(agg['age'].min(), agg['age'].max(), 100)

    sc = ax.scatter(agg['age'], agg['mean_dist'],
                    c=agg['age'], cmap='viridis', s=50,
                    edgecolors='black', linewidths=0.5, zorder=3)
    ax.plot(x_line, slope * x_line + intercept, color='red', lw=1.5, zorder=2)

    ct_short = ct.replace('-neurons','').replace('-cells','')
    ax.set_title(f'{ct_short} lou{louv} {region}\nr={r:.3f} p={p:.3f}', fontsize=9)
    ax.set_xlabel('age (years)', fontsize=8)
    ax.set_ylabel('mean dist to centroid', fontsize=8)
    ax.tick_params(labelsize=7)

plt.suptitle('Per-animal mean distance to within-animal centroid vs age\ntop 10 louvains by |r|', y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'centroid_age_scatter_top10.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
