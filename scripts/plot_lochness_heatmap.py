"""
plot_lochness_heatmap.py

Cross-cell-type heatmap of lochNESS results.
Two heatmaps:
    1. mean lochNESS per cell type x region (collapsed across louvains)
    2. % significant cells per cell type x region

Output:
    /scratch/easmit31/factor_analysis/lochness_heatmap_celltype_region.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

LOCHNESS_DIR = '/scratch/easmit31/factor_analysis/lochness_pca'

CELL_TYPES = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'opc-olig', 'vascular-cells',
]

REGIONS = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']

# load all lochNESS summaries
rows = []
for ct in CELL_TYPES:
    for region in REGIONS:
        f = os.path.join(LOCHNESS_DIR, ct, f'lochness_summary_{region}.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        # weight by n_cells when collapsing across louvains
        total_cells = df['n_cells'].sum()
        mean_loch   = (df['mean_lochness'] * df['n_cells']).sum() / total_cells
        pct_sig     = (df['n_significant'].sum() / df['n_cells'].sum()) * 100
        n_sig_louv  = (df['p_age'] < 0.05).sum()
        pct_sig_louv = n_sig_louv / len(df) * 100
        rows.append({
            'cell_type': ct,
            'region':    region,
            'mean_lochness':    mean_loch,
            'pct_significant':  pct_sig,
            'pct_sig_louvains': pct_sig_louv,
            'n_louvains':       len(df),
        })

if not rows:
    print("No lochNESS summary files found")
    exit(1)

summary = pd.DataFrame(rows)
print(f"Loaded {len(summary)} cell type x region combinations")

# pivot to matrices
mean_loch_mat = summary.pivot(index='cell_type', columns='region', values='mean_lochness').reindex(index=CELL_TYPES, columns=REGIONS)
pct_sig_mat   = summary.pivot(index='cell_type', columns='region', values='pct_significant').reindex(index=CELL_TYPES, columns=REGIONS)
pct_sig_louv_mat = summary.pivot(index='cell_type', columns='region', values='pct_sig_louvains').reindex(index=CELL_TYPES, columns=REGIONS)

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for ax, mat, title, cmap, center in [
    (axes[0], mean_loch_mat,    'mean lochNESS\n(weighted by n_cells)', 'RdBu_r', 0),
    (axes[1], pct_sig_mat,      '% significant cells\n(p<0.05)',         'Reds',   None),
    (axes[2], pct_sig_louv_mat, '% louvains with\nsig age r (p<0.05)',   'Reds',   None),
]:
    sns.heatmap(mat, ax=ax, cmap=cmap, center=center,
                annot=True, fmt='.1f', annot_kws={'size': 7},
                linewidths=0.5, linecolor='white',
                mask=mat.isna(),
                cbar_kws={'shrink': 0.8})
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plt.suptitle('lochNESS summary across cell types and regions', y=1.02)
plt.tight_layout()
out = '/scratch/easmit31/factor_analysis/lochness_heatmap_celltype_region.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
