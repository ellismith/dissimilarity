import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('/scratch/easmit31/factor_analysis/pca_embedding_sizes.csv')

CELL_TYPES = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'opc-olig', 'vascular-cells',
]
REGIONS = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']

# collapse across louvains — sum cells, mean pc1_std weighted by n_cells
agg = df.groupby(['cell_type', 'region']).apply(lambda x: pd.Series({
    'total_cells':    x['n_cells'].sum(),
    'n_louvains':     len(x),
    'n_animals':      x['n_animals'].max(),
    'pc1_std_wtd':    (x['pc1_std'] * x['n_cells']).sum() / x['n_cells'].sum(),
    'mean_l2norm_wtd': (x['mean_l2norm'] * x['n_cells']).sum() / x['n_cells'].sum(),
})).reset_index()

cells_mat  = agg.pivot(index='cell_type', columns='region', values='total_cells').reindex(index=CELL_TYPES, columns=REGIONS)
pc1_mat    = agg.pivot(index='cell_type', columns='region', values='pc1_std_wtd').reindex(index=CELL_TYPES, columns=REGIONS)
l2_mat     = agg.pivot(index='cell_type', columns='region', values='mean_l2norm_wtd').reindex(index=CELL_TYPES, columns=REGIONS)
louv_mat   = agg.pivot(index='cell_type', columns='region', values='n_louvains').reindex(index=CELL_TYPES, columns=REGIONS)

fig, axes = plt.subplots(2, 2, figsize=(22, 14))

for ax, mat, title, fmt, cmap, center in [
    (axes[0,0], np.log10(cells_mat.fillna(0).replace(0, np.nan)), 'log10 total cells',         '.1f', 'Blues',  None),
    (axes[0,1], louv_mat,                                          'n louvains',                '.0f', 'Purples',None),
    (axes[1,0], pc1_mat,                                           'PC1 std (weighted)',        '.2f', 'Reds',   None),
    (axes[1,1], l2_mat,                                            'mean L2 norm (weighted)',   '.2f', 'Oranges',None),
]:
    sns.heatmap(mat, ax=ax, cmap=cmap, center=center,
                annot=True, fmt=fmt, annot_kws={'size': 7},
                linewidths=0.5, linecolor='white',
                mask=mat.isna(),
                cbar_kws={'shrink': 0.8})
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plt.suptitle('PCA embedding summary across cell types and regions', y=1.01)
plt.tight_layout()
out = '/scratch/easmit31/factor_analysis/pca_embedding_summary.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
