"""
pc_distributions_by_louvain.py

Visualizes the distribution of PCA values across louvain clusters for a given
cell type and brain region. For each PC, plots a violin or box plot showing
the distribution of that PC's values across all louvain clusters, allowing
comparison of how clusters differ in PCA space.

Output:
    - One plot per PC (or a grid), saved to /scratch/easmit31/factor_analysis/

Usage:
    python pc_distributions_by_louvain.py
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters ---
cell_class_file = "Res1_GABAergic-neurons_subset.h5ad"
region = "HIP"
n_pcs = 10        # how many PCs to visualize
min_cells = 100   # skip louvains with fewer cells
min_age = 1.0

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

# --- Load ---
fpath = f'/data/CEM/smacklab/U01/{cell_class_file}'
print(f"Loading {fpath}...")
with h5py.File(fpath, 'r') as f:
    X_pca = f['obsm']['X_pca'][:, :n_pcs]
    regions = decode_categorical(f['obs']['region'])
    louvain = decode_categorical(f['obs']['louvain'])
    ages = f['obs']['age'][:]

region_mask = (regions == region) & (ages >= min_age)
louvain_clusters = sorted(np.unique(louvain[region_mask]), key=lambda x: int(x))

# filter to min_cells
valid_clusters = [cl for cl in louvain_clusters
                  if (region_mask & (louvain == cl)).sum() >= min_cells]
print(f"{len(valid_clusters)} louvains with >= {min_cells} cells")

# build dataframe
rows = []
for cl in valid_clusters:
    mask = region_mask & (louvain == cl)
    pca_subset = X_pca[mask]
    for pc_idx in range(n_pcs):
        for val in pca_subset[:, pc_idx]:
            rows.append({'louvain': cl, 'pc': f'PC{pc_idx+1}', 'value': val})

df = pd.DataFrame(rows)

# one subplot per PC, violin plots across louvains
fig, axes = plt.subplots(n_pcs, 1, figsize=(14, 3 * n_pcs), sharex=False)
if n_pcs == 1:
    axes = [axes]

for pc_idx, ax in enumerate(axes):
    pc_label = f'PC{pc_idx+1}'
    data_per_louvain = [df[(df['louvain'] == cl) & (df['pc'] == pc_label)]['value'].values
                        for cl in valid_clusters]
    parts = ax.violinplot(data_per_louvain, positions=range(len(valid_clusters)),
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(valid_clusters)))
    ax.set_xticklabels([f'L{cl}' for cl in valid_clusters], fontsize=7)
    ax.set_ylabel(pc_label)
    ax.axhline(0, color='gray', lw=0.5, ls='--')

axes[0].set_title(f'GABAergic {region} — PC value distributions by louvain cluster\n(min {min_cells} cells)', fontsize=11)
plt.tight_layout()
out = f'/scratch/easmit31/factor_analysis/GABAergic_{region}_PC_distributions_by_louvain.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {out}")
