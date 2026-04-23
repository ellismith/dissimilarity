"""
plot_centroid_examples.py

Finds the louvain x region with the strongest within-animal centroid age effect,
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

# find strongest age effect
rows = []
for f in glob.glob(os.path.join(CENTROID_DIR, '*_centroid_summary.csv')):
    ct = os.path.basename(f).replace('_centroid_summary.csv', '')
    df = pd.read_csv(f)
    df['cell_type'] = ct
    rows.append(df)

summary = pd.concat(rows, ignore_index=True)
summary['abs_r'] = summary['r_mean'].abs()
best = summary.nlargest(1, 'abs_r').iloc[0]

ct     = best['cell_type']
region = best['region']
louv   = str(best['louvain'])
r_val  = best['r_mean']
p_val  = best['p_mean']

print(f"Strongest effect: {ct} louvain {louv} {region} r={r_val:.3f} p={p_val:.4f}")

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
X = X[meta.index.values] if len(meta) < len(X) else X[:len(meta)]

dist = meta['dist_to_centroid'].values
ages = meta['age'].values

far_mask   = np.zeros(len(meta), dtype=bool)
close_mask = np.zeros(len(meta), dtype=bool)
far_mask[np.argsort(dist)[-N_HIGHLIGHT:]]   = True
close_mask[np.argsort(dist)[:N_HIGHLIGHT]]  = True
neither_mask = ~far_mask & ~close_mask

vmin, vmax = ages.min(), ages.max()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, highlight_mask, label in [
    (axes[0], far_mask,   f'furthest {N_HIGHLIGHT} from animal centroid'),
    (axes[1], close_mask, f'closest {N_HIGHLIGHT} to animal centroid'),
]:
    # background cells — small
    sc = ax.scatter(X[neither_mask, 0], X[neither_mask, 1],
                    c=ages[neither_mask], cmap='viridis',
                    vmin=vmin, vmax=vmax,
                    s=5, alpha=0.4, zorder=1)

    # highlighted cells — large, same colormap
    ax.scatter(X[highlight_mask, 0], X[highlight_mask, 1],
               c=ages[highlight_mask], cmap='viridis',
               vmin=vmin, vmax=vmax,
               s=120, alpha=1.0, zorder=3,
               edgecolors='black', linewidths=0.8)

    plt.colorbar(sc, ax=ax, label='age (years)', shrink=0.8)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{label}\n{ct} louvain {louv} {region}')

plt.suptitle(f'Within-animal centroid distance — cells colored by age\nr={r_val:.3f} p={p_val:.4f} (strongest age effect)', y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'centroid_distance_examples.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
