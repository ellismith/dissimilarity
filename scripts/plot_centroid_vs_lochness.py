"""
plot_centroid_vs_lochness.py

Scatter plot of centroid distance age r vs lochNESS age r per louvain x region,
across all cell types. Tests whether subtypes with more transcriptional spread
with age also show more age-based neighborhood clustering.

Output:
    /scratch/easmit31/factor_analysis/centroid_vs_lochness_scatter.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

CENTROID_DIR = '/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100'
LOCHNESS_DIR = '/scratch/easmit31/factor_analysis/lochness_pca'

CELL_TYPES = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'opc-olig', 'vascular-cells',
]

COLORS = {
    'GABAergic-neurons':     '#E24B4A',
    'glutamatergic-neurons': '#378ADD',
    'astrocytes':            '#1D9E75',
    'microglia':             '#BA7517',
    'basket-cells':          '#D4537E',
    'medium-spiny-neurons':  '#7F77DD',
    'cerebellar-neurons':    '#639922',
    'ependymal-cells':       '#185FA5',
    'midbrain-neurons':      '#993C1D',
    'opc-olig':              '#0F6E56',
    'vascular-cells':        '#5F5E5A',
}

# load centroid summaries
centroid_dfs = []
for ct in CELL_TYPES:
    f = os.path.join(CENTROID_DIR, f'{ct}_centroid_summary.csv')
    if os.path.exists(f):
        df = pd.read_csv(f)
        df['cell_type'] = ct
        centroid_dfs.append(df)

centroid = pd.concat(centroid_dfs, ignore_index=True)
centroid['louvain'] = centroid['louvain'].astype(str)

# load lochNESS summaries
lochness_dfs = []
for ct in CELL_TYPES:
    pattern = os.path.join(LOCHNESS_DIR, ct, 'lochness_summary_*.csv')
    for f in glob.glob(pattern):
        region = os.path.basename(f).replace('lochness_summary_', '').replace('.csv', '')
        df = pd.read_csv(f)
        df['cell_type'] = ct
        df['region']    = region
        lochness_dfs.append(df)

if not lochness_dfs:
    print("No lochNESS summary files found — run summarize_lochness_results.py first")
    exit(1)

lochness = pd.concat(lochness_dfs, ignore_index=True)
lochness['louvain'] = lochness['louvain'].astype(str)

# merge
merged = centroid.merge(lochness, on=['cell_type', 'region', 'louvain'], suffixes=('_centroid', '_lochness'))
print(f"Merged: {len(merged)} louvain x region combinations")

# plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, x_col, x_label in [
    (axes[0], 'r_mean', 'centroid distance (mean) age r'),
    (axes[1], 'r_var',  'centroid distance (variance) age r'),
]:
    for ct in CELL_TYPES:
        sub = merged[merged['cell_type'] == ct]
        if sub.empty:
            continue
        sig = (sub['p_mean'] < 0.05) | (sub['p_age'] < 0.05)
        ax.scatter(sub.loc[~sig, x_col], sub.loc[~sig, 'r_age'],
                   color=COLORS[ct], s=20, alpha=0.4, edgecolors='none')
        ax.scatter(sub.loc[sig, x_col], sub.loc[sig, 'r_age'],
                   color=COLORS[ct], s=50, alpha=0.9, edgecolors='black', linewidths=0.5,
                   label=ct if ax == axes[0] else None)

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel('lochNESS age r')
    ax.set_title(f'{x_label}\nvs lochNESS age r')

    # correlation
    valid = merged[[x_col, 'r_age']].dropna()
    if len(valid) > 5:
        from scipy import stats
        r, p = stats.pearsonr(valid[x_col], valid['r_age'])
        ax.annotate(f'r={r:.2f} p={p:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9)

axes[0].legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle('centroid distance vs lochNESS age correlations\nper louvain x region')
plt.tight_layout()
out = '/scratch/easmit31/factor_analysis/centroid_vs_lochness_scatter.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {out}')
