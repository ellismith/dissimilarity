"""
plot_lochness_heatmap.py

Cross-cell-type heatmap of lochNESS results (continuous age).

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
    'midbrain-neurons', 'OPCs', 'oligodendrocytes', 'vascular-cells',
]

REGIONS = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']

rows = []
for ct in CELL_TYPES:
    for region in REGIONS:
        f = os.path.join(LOCHNESS_DIR, ct, f'lochness_summary_{region}.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        total_cells   = df['n_cells'].sum()
        mean_zscore   = (df['mean_zscore'] * df['n_cells']).sum() / total_cells
        mean_r_age    = (df['r_age'] * df['n_cells']).sum() / total_cells
        n_sig_louv    = (df['p_age'] < 0.05).sum()
        pct_sig_louv  = n_sig_louv / len(df) * 100
        rows.append({
            'cell_type':       ct,
            'region':          region,
            'mean_zscore':     mean_zscore,
            'mean_r_age':      mean_r_age,
            'pct_sig_louvains': pct_sig_louv,
            'n_louvains':      len(df),
        })

if not rows:
    print("No lochNESS summary files found")
    exit(1)

summary = pd.DataFrame(rows)
print(f"Loaded {len(summary)} cell type x region combinations")

zscore_mat   = summary.pivot(index='cell_type', columns='region', values='mean_zscore').reindex(index=CELL_TYPES, columns=REGIONS)
r_age_mat    = summary.pivot(index='cell_type', columns='region', values='mean_r_age').reindex(index=CELL_TYPES, columns=REGIONS)
pct_sig_mat  = summary.pivot(index='cell_type', columns='region', values='pct_sig_louvains').reindex(index=CELL_TYPES, columns=REGIONS)

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for ax, mat, title, cmap, center in [
    (axes[0], zscore_mat,  'mean neighbor age z-score\n(weighted by n_cells)', 'RdBu_r', 0),
    (axes[1], r_age_mat,   'mean r (age vs neighbor age)\n(weighted by n_cells)', 'RdBu_r', 0),
    (axes[2], pct_sig_mat, '% louvains with sig age r\n(p<0.05)',                'Reds',   None),
]:
    sns.heatmap(mat, ax=ax, cmap=cmap, center=center,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                linewidths=0.5, linecolor='white',
                mask=mat.isna(),
                cbar_kws={'shrink': 0.8})
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plt.suptitle('lochNESS summary — continuous age', y=1.02)
plt.tight_layout()
out = '/scratch/easmit31/factor_analysis/lochness_heatmap_celltype_region.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
