#!/usr/bin/env python3
"""
plot_lochness_summary_heatmap.py

Summary heatmap (cell type x region) for lochNESS ~ age.
Left panel: mean r_age collapsed across louvains.
Right panel: n louvains significant (p_age < fdr_thresh, within-CT FDR).
Bold outline = any louvain significant.

Reads: lochness_pca/{ct}/lochness_summary_{region}.csv

Output: centroid_heatmaps/lochness_summary_heatmap.png
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--lochness-dir', default='/scratch/easmit31/factor_analysis/lochness_pca')
parser.add_argument('--output-dir',   default='/scratch/easmit31/factor_analysis/centroid_heatmaps')
parser.add_argument('--fdr-thresh',   type=float, default=0.05)
parser.add_argument('--vmin',         type=float, default=-0.6)
parser.add_argument('--vmax',         type=float, default=0.6)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

REGION_ORDER = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']
CELL_TYPE_ORDER = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'opc', 'OPCs', 'oligodendrocytes', 'vascular-cells',
]

# ── Load all lochNESS summaries ───────────────────────────────────────────────
print('loading lochNESS summaries...')
rows = []
for ct in os.listdir(args.lochness_dir):
    ct_dir = os.path.join(args.lochness_dir, ct)
    if not os.path.isdir(ct_dir):
        continue
    for fpath in glob.glob(os.path.join(ct_dir, 'lochness_summary_*.csv')):
        region = os.path.basename(fpath).replace('lochness_summary_', '').replace('.csv', '')
        df = pd.read_csv(fpath)
        df['cell_type'] = ct
        df['region']    = region
        rows.append(df)

if not rows:
    raise FileNotFoundError(f'No lochNESS summary CSVs found in {args.lochness_dir}')

data = pd.concat(rows, ignore_index=True)
print(f'loaded {len(data)} louvain x region records across {data["cell_type"].nunique()} cell types')

# ── FDR within each cell type ─────────────────────────────────────────────────
data = data.dropna(subset=['p_age']).copy().reset_index(drop=True)
data['sig'] = False
for ct, grp in data.groupby('cell_type'):
    reject, _, _, _ = multipletests(grp['p_age'].fillna(1.0), alpha=args.fdr_thresh, method='fdr_bh')
    data.loc[grp.index, 'sig'] = reject.tolist()

# ── Collapse across louvains ──────────────────────────────────────────────────
summary = data.groupby(['cell_type', 'region']).agg(
    mean_r_age  = ('r_age', 'mean'),
    n_sig       = ('sig',   'sum'),
    any_sig     = ('sig',   'any'),
    n_louvains  = ('r_age', 'count'),
).reset_index()

present_cts = [c for c in CELL_TYPE_ORDER if c in summary['cell_type'].unique()]

def to_mat(col):
    return summary.pivot(index='cell_type', columns='region', values=col) \
                  .reindex(index=present_cts, columns=REGION_ORDER)

r_mat   = to_mat('mean_r_age')
n_mat   = to_mat('n_sig').fillna(0)
sig_mat = to_mat('any_sig').fillna(False)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(24, 8))
fig.suptitle(
    f'lochNESS ~ age — mean r_age across louvains\n'
    f'Bold outline = any louvain FDR q<{args.fdr_thresh}; number = n significant louvains',
    fontsize=12, y=1.02
)

for ax, mat, title in [
    (axes[0], r_mat,  'mean r_age (lochNESS ~ age) collapsed across louvains'),
    (axes[1], n_mat,  'n louvains significant (within-CT FDR q<{:.2f})'.format(args.fdr_thresh)),
]:
    cmap  = 'RdBu_r' if ax == axes[0] else 'OrRd'
    vmin  = args.vmin if ax == axes[0] else 0
    vmax  = args.vmax if ax == axes[0] else n_mat.max().max()
    clabel = 'mean r_age' if ax == axes[0] else 'n significant louvains'

    sns.heatmap(
        mat,
        cmap=cmap, center=0 if ax == axes[0] else None,
        vmin=vmin, vmax=vmax,
        annot=False,
        linewidths=0.4, linecolor='lightgrey',
        cbar_kws={'label': clabel, 'shrink': 0.5},
        ax=ax, mask=mat.isnull()
    )

    for i, ct in enumerate(present_cts):
        for j, region in enumerate(REGION_ORDER):
            if ct not in mat.index or region not in mat.columns:
                continue
            val = mat.loc[ct, region]
            sig = sig_mat.loc[ct, region] if ct in sig_mat.index and region in sig_mat.columns else False
            n   = int(n_mat.loc[ct, region]) if ct in n_mat.index and region in n_mat.columns else 0

            if pd.isna(val):
                continue

            txt_color = 'white' if (ax == axes[0] and abs(val) > 0.35) or \
                                   (ax == axes[1] and val > vmax * 0.6) else 'black'

            if sig:
                ax.text(j + 0.5, i + 0.38, f'{val:.2f}' if ax == axes[0] else str(int(val)),
                        ha='center', va='center', fontsize=9,
                        fontweight='bold', color=txt_color)
                if ax == axes[0]:
                    ax.text(j + 0.5, i + 0.68, f'n={n}',
                            ha='center', va='center', fontsize=7,
                            fontweight='bold', color=txt_color)
                ax.add_patch(mpatches.Rectangle(
                    (j + 0.03, i + 0.03), 0.94, 0.94,
                    fill=False, edgecolor='black', lw=2.5, zorder=5
                ))
            else:
                ax.text(j + 0.5, i + 0.5, f'{val:.2f}' if ax == axes[0] else str(int(val)),
                        ha='center', va='center', fontsize=9,
                        fontweight='normal', color=txt_color)

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel('region', fontsize=9)
    ax.set_ylabel('cell type', fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.tight_layout()
outpath = os.path.join(args.output_dir, f'lochness_summary_heatmap_q{int(args.fdr_thresh*100):02d}.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {outpath}')
