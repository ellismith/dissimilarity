"""
summarize_population_centroid.py

Loads all population centroid summary CSVs, applies FDR correction
within cell type, and plots heatmaps of r_mean_dist and % sig louvains.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from statsmodels.stats.multitest import multipletests

POP_DIR  = '/scratch/easmit31/factor_analysis/population_centroid_outputs'
OUT_DIR  = '/scratch/easmit31/factor_analysis'

CELL_TYPES = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'OPCs', 'oligodendrocytes', 'vascular-cells',
]
REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']

# load all
rows = []
for ct in CELL_TYPES:
    for region in REGIONS:
        f = os.path.join(POP_DIR, f'{ct}_{region}_population_centroid_summary.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df['cell_type'] = ct
        df['region']    = region
        rows.append(df)

summary = pd.concat(rows, ignore_index=True)
summary['louvain'] = summary['louvain'].astype(str)
summary = summary.dropna(subset=['p_mean_dist'])

print(f"Loaded {len(summary)} louvains across {summary['cell_type'].nunique()} cell types")

# FDR within cell type
results = []
for ct, group in summary.groupby('cell_type'):
    reject, pvals_fdr, _, _ = multipletests(group['p_mean_dist'], method='fdr_bh')
    group = group.copy()
    group['p_fdr'] = pvals_fdr
    group['sig_fdr'] = reject
    results.append(group)

summary = pd.concat(results)

print(f"\nSignificant at p<0.05:        {(summary['p_mean_dist'] < 0.05).sum()} ({(summary['p_mean_dist'] < 0.05).mean()*100:.1f}%)")
print(f"Significant after within-CT FDR: {summary['sig_fdr'].sum()} ({summary['sig_fdr'].mean()*100:.1f}%)")

# direction by cell type
print("\nDirection of r_mean_dist by cell type:")
dir_df = summary.groupby('cell_type').apply(lambda x: pd.Series({
    'n_louvains':   len(x),
    'pct_positive': (x['r_mean_dist'] > 0).sum() / len(x) * 100,
    'mean_r':       x['r_mean_dist'].mean(),
    'n_sig_fdr':    x['sig_fdr'].sum(),
})).reset_index().sort_values('mean_r', ascending=False)
print(dir_df.to_string(index=False))

# pivot for heatmaps
r_mat   = summary.groupby(['cell_type','region'])['r_mean_dist'].mean().reset_index()
r_mat   = r_mat.pivot(index='cell_type', columns='region', values='r_mean_dist').reindex(index=CELL_TYPES, columns=REGIONS)

sig_mat = summary.groupby(['cell_type','region'])['sig_fdr'].mean().reset_index()
sig_mat = sig_mat.pivot(index='cell_type', columns='region', values='sig_fdr').reindex(index=CELL_TYPES, columns=REGIONS) * 100

fig, axes = plt.subplots(1, 2, figsize=(22, 8))

sns.heatmap(r_mat, ax=axes[0], cmap='RdBu_r', center=0, vmin=-0.6, vmax=0.6,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='white', mask=r_mat.isna(),
            cbar_kws={'shrink': 0.8, 'label': 'mean r'})
axes[0].set_title('mean r (population centroid distance ~ age)\ncollapsed across louvains')
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

sns.heatmap(sig_mat, ax=axes[1], cmap='Reds', vmin=0, vmax=100,
            annot=True, fmt='.0f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='white', mask=sig_mat.isna(),
            cbar_kws={'shrink': 0.8, 'label': '% louvains sig (within-CT FDR)'})
axes[1].set_title('% louvains significant\n(within cell type FDR q<0.05)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

plt.suptitle('Population centroid distance ~ age', y=1.02)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'population_centroid_heatmap.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: {out}")

# save with FDR
summary.to_csv(os.path.join(OUT_DIR, 'population_centroid_summary_with_fdr.csv'), index=False)
print(f"✓ Saved: population_centroid_summary_with_fdr.csv")
