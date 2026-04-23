"""
plot_pca_age_panels.py

3-row panel of PC1/PC2 scatters colored by age:
  Row 1: top 5 louvains by lochNESS age r
  Row 2: top 5 louvains by population centroid age r
  Row 3: bottom 5 louvains (weakest age effect, from either metric)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob

EMBED_DIR   = '/scratch/easmit31/factor_analysis/pca_embeddings'
LOCH_DIR    = '/scratch/easmit31/factor_analysis/lochness_pca'
POP_DIR     = '/scratch/easmit31/factor_analysis/population_centroid_outputs'
OUT_DIR     = '/scratch/easmit31/factor_analysis'

CELL_TYPES = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'OPCs', 'oligodendrocytes', 'vascular-cells',
]
REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']

# load lochNESS summaries
loch_rows = []
for ct in CELL_TYPES:
    for region in REGIONS:
        f = os.path.join(LOCH_DIR, ct, f'lochness_summary_{region}.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            loch_rows.append({
                'cell_type': ct, 'region': region,
                'louvain': str(row['louvain']),
                'r_loch': row['r_age'], 'p_loch': row['p_age'],
                'n_cells': row['n_cells'],
            })

loch_df = pd.DataFrame(loch_rows)

# load population centroid summaries
pop_rows = []
for ct in CELL_TYPES:
    for region in REGIONS:
        f = os.path.join(POP_DIR, f'{ct}_{region}_population_centroid_summary.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            pop_rows.append({
                'cell_type': ct, 'region': region,
                'louvain': str(row['louvain']),
                'r_pop': row['r_mean_dist'], 'p_pop': row['p_mean_dist'],
                'n_cells': row['n_cells'],
            })

pop_df = pd.DataFrame(pop_rows)

# filter to louvains that have pca embeddings
def has_embedding(ct, region, louv):
    p = os.path.join(EMBED_DIR, ct, f'louvain{louv}_{region}_pca.npy')
    return os.path.exists(p)

loch_df = loch_df[loch_df.apply(lambda r: has_embedding(r['cell_type'], r['region'], r['louvain']), axis=1)]
pop_df  = pop_df[pop_df.apply(lambda r: has_embedding(r['cell_type'], r['region'], r['louvain']), axis=1)]

# top 5 lochNESS
top_loch = loch_df.nlargest(5, 'r_loch')

# top 5 population centroid
top_pop = pop_df.nlargest(5, 'r_pop')

# bottom 5 — weakest absolute lochNESS r with sufficient cells
bottom = loch_df[loch_df['n_cells'] >= 500].nsmallest(5, 'r_loch')

def plot_row(axes, rows, r_col, label):
    for ax, (_, row) in zip(axes, rows.iterrows()):
        ct     = row['cell_type']
        region = row['region']
        louv   = row['louvain']
        r_val  = row[r_col]

        pca_path  = os.path.join(EMBED_DIR, ct, f'louvain{louv}_{region}_pca.npy')
        meta_path = os.path.join(EMBED_DIR, ct, f'louvain{louv}_{region}_metadata.csv')

        X    = np.load(pca_path)
        meta = pd.read_csv(meta_path)
        ages = meta['age'].values

        # subsample for speed
        if len(X) > 5000:
            idx = np.random.choice(len(X), 5000, replace=False)
            X_plot = X[idx]; ages_plot = ages[idx]
        else:
            X_plot = X; ages_plot = ages

        sc = ax.scatter(X_plot[:, 0], X_plot[:, 1],
                        c=ages_plot, cmap='viridis', s=3, alpha=0.5)
        plt.colorbar(sc, ax=ax, shrink=0.7, label='age')
        ct_short = ct.replace('-neurons','').replace('-cells','')
        ax.set_title(f'{ct_short}\nlou{louv} {region}\n{label} r={r_val:.2f}', fontsize=8)
        ax.set_xlabel('PC1', fontsize=7)
        ax.set_ylabel('PC2', fontsize=7)
        ax.tick_params(labelsize=6)

fig, axes = plt.subplots(3, 5, figsize=(22, 14))

plot_row(axes[0], top_loch,  'r_loch', 'lochNESS')
plot_row(axes[1], top_pop,   'r_pop',  'pop centroid')
plot_row(axes[2], bottom,    'r_loch', 'lochNESS (weak)')

plt.suptitle('PC1/PC2 colored by age — strongest and weakest age effects', y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'pca_age_panels.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
