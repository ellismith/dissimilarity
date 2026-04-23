"""
plot_centroid_distance_heatmap.py

For a given cell type, plots a heatmap of per-animal mean centroid distance
across louvains. Rows = animals sorted by age, cols = louvains.
Uses the cell type with the most significant age effects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

CENTROID_DIR = '/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100'
EMBED_DIR    = '/scratch/easmit31/factor_analysis/pca_embeddings'
OUT_DIR      = '/scratch/easmit31/factor_analysis'

# find cell type x region with most significant louvains
rows = []
for f in glob.glob(os.path.join(CENTROID_DIR, '*_centroid_summary.csv')):
    ct = os.path.basename(f).replace('_centroid_summary.csv', '')
    df = pd.read_csv(f)
    df['cell_type'] = ct
    rows.append(df)

summary = pd.concat(rows, ignore_index=True)

# pick cell type + region with most sig louvains
sig_counts = summary[summary['p_mean'] < 0.05].groupby(['cell_type','region']).size().reset_index(name='n_sig')
best = sig_counts.nlargest(1, 'n_sig').iloc[0]
ct     = best['cell_type']
region = best['region']

print(f"Plotting heatmap for: {ct} {region}")

sub = summary[(summary['cell_type'] == ct) & (summary['region'] == region)]

# for each louvain compute per-animal mean distance
all_animals = set()
louvain_data = {}

for _, row in sub.iterrows():
    louv = str(row['louvain'])
    pca_path  = os.path.join(EMBED_DIR, ct, f'louvain{louv}_{region}_pca.npy')
    meta_path = os.path.join(EMBED_DIR, ct, f'louvain{louv}_{region}_metadata.csv')
    if not os.path.exists(pca_path):
        continue

    X    = np.load(pca_path)
    meta = pd.read_csv(meta_path).reset_index(drop=True)

    animal_dist = {}
    for animal in meta['animal_id'].unique():
        mask = meta['animal_id'] == animal
        idxs = np.where(mask)[0]
        if len(idxs) < 2:
            continue
        centroid = X[idxs].mean(axis=0)
        dists = np.sqrt(((X[idxs] - centroid) ** 2).sum(axis=1))
        animal_dist[animal] = dists.mean()
        all_animals.add((animal, meta.loc[mask, 'age'].iloc[0]))

    louvain_data[f'lou{louv}\nr={row["r_mean"]:.2f}'] = animal_dist

# build matrix
animal_age = {a: age for a, age in all_animals}
animals_sorted = sorted(animal_age.keys(), key=lambda a: animal_age[a])
ages_sorted = [animal_age[a] for a in animals_sorted]

mat = pd.DataFrame(index=animals_sorted, columns=list(louvain_data.keys()))
for louv_label, dist_dict in louvain_data.items():
    for animal in animals_sorted:
        mat.loc[animal, louv_label] = dist_dict.get(animal, np.nan)

mat = mat.astype(float)

# z-score across animals per louvain
mat_z = (mat - mat.mean()) / mat.std()

fig, axes = plt.subplots(1, 2, figsize=(20, 12))

for ax, data, title in [
    (axes[0], mat,   'mean distance to centroid (raw)'),
    (axes[1], mat_z, 'mean distance to centroid (z-scored per louvain)'),
]:
    sns.heatmap(data, ax=ax, cmap='RdBu_r', center=0 if 'z-scored' in title else None,
                yticklabels=[f'{a} ({animal_age[a]:.1f}y)' for a in animals_sorted],
                linewidths=0, cbar_kws={'shrink': 0.5})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('louvain (r = age correlation)')
    ax.set_ylabel('animal (sorted by age)')
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7, rotation=45)

plt.suptitle(f'Per-animal centroid distance heatmap\n{ct} {region}', y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'centroid_distance_heatmap_animals.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
