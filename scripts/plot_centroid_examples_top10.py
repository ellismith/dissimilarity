"""
plot_centroid_examples_top10.py

For the top 10 louvain x region combos by within-animal centroid age r,
plots all cells in PC1/PC2 colored by age, with furthest/closest cells larger.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

CENTROID_DIR = '/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100'
EMBED_DIR    = '/scratch/easmit31/factor_analysis/pca_embeddings'
OUT_DIR      = '/scratch/easmit31/factor_analysis'
N_HIGHLIGHT  = 15
N_TOP        = 10

# find top 10 strongest age effects
rows = []
for f in glob.glob(os.path.join(CENTROID_DIR, '*_centroid_summary.csv')):
    ct = os.path.basename(f).replace('_centroid_summary.csv', '')
    df = pd.read_csv(f)
    df['cell_type'] = ct
    rows.append(df)

summary = pd.concat(rows, ignore_index=True)
summary['abs_r'] = summary['r_mean'].abs()

# only keep combos where pca embedding exists
def has_embedding(row):
    p = os.path.join(EMBED_DIR, row['cell_type'], f"louvain{row['louvain']}_{row['region']}_pca.npy")
    return os.path.exists(p)

summary = summary[summary.apply(has_embedding, axis=1)]
top10 = summary.nlargest(N_TOP, 'abs_r')

fig, axes = plt.subplots(2, 5, figsize=(28, 12))
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
    meta['dist_to_centroid'] = np.nan
    for animal in meta['animal_id'].unique():
        mask = meta['animal_id'] == animal
        idxs = np.where(mask)[0]
        if len(idxs) < 2:
            continue
        centroid = X[idxs].mean(axis=0)
        dists = np.sqrt(((X[idxs] - centroid) ** 2).sum(axis=1))
        meta.loc[mask, 'dist_to_centroid'] = dists

    meta = meta.dropna(subset=['dist_to_centroid']).reset_index(drop=True)
    X_plot = X[:len(meta)]

    dist = meta['dist_to_centroid'].values
    ages = meta['age'].values

    far_mask   = np.zeros(len(meta), dtype=bool)
    close_mask = np.zeros(len(meta), dtype=bool)
    far_mask[np.argsort(dist)[-N_HIGHLIGHT:]]  = True
    close_mask[np.argsort(dist)[:N_HIGHLIGHT]] = True
    neither_mask = ~far_mask & ~close_mask

    vmin, vmax = ages.min(), ages.max()

    # subsample background if large
    if neither_mask.sum() > 3000:
        bg_idx = np.where(neither_mask)[0]
        bg_idx = np.random.choice(bg_idx, 3000, replace=False)
        bg_mask = np.zeros(len(meta), dtype=bool)
        bg_mask[bg_idx] = True
    else:
        bg_mask = neither_mask

    sc = ax.scatter(X_plot[bg_mask, 0], X_plot[bg_mask, 1],
                    c=ages[bg_mask], cmap='viridis',
                    vmin=vmin, vmax=vmax,
                    s=3, alpha=0.4, zorder=1)
    ax.scatter(X_plot[far_mask, 0], X_plot[far_mask, 1],
               c=ages[far_mask], cmap='viridis',
               vmin=vmin, vmax=vmax,
               s=120, alpha=1.0, zorder=3,
               edgecolors='red', linewidths=0.8, label='far')
    ax.scatter(X_plot[close_mask, 0], X_plot[close_mask, 1],
               c=ages[close_mask], cmap='viridis',
               vmin=vmin, vmax=vmax,
               s=120, alpha=1.0, zorder=3,
               edgecolors='blue', linewidths=0.8, label='close')

    plt.colorbar(sc, ax=ax, shrink=0.7, label='age')
    ct_short = ct.replace('-neurons','').replace('-cells','')
    ax.set_title(f'{ct_short} lou{louv} {region}\nr={r_val:.3f} p={p_val:.3f}', fontsize=9)
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.tick_params(labelsize=7)

# legend on last axis
axes[-1].scatter([], [], s=80, edgecolors='red', facecolors='gray', label=f'furthest {N_HIGHLIGHT}')
axes[-1].scatter([], [], s=80, edgecolors='blue', facecolors='gray', label=f'closest {N_HIGHLIGHT}')
axes[-1].legend(fontsize=9, loc='center')

plt.suptitle(f'Top 10 louvains by within-animal centroid distance age effect\ncells colored by age (viridis), red outline = far, blue outline = close', y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'centroid_distance_top10.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
