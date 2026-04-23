#!/usr/bin/env python3
"""
plot_centroid_summary_comparison.py

Two-panel summary heatmap (cell type x region) showing mean r collapsed
across louvains for within-animal (left) and population (right) centroid
distance ~ age. Cells are outlined/filled if any louvain survives FDR q<0.1.

Output: centroid_heatmaps/centroid_summary_comparison.png
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
parser.add_argument('--within-dir', default='/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100')
parser.add_argument('--pop-dir',    default='/scratch/easmit31/factor_analysis/population_centroid_outputs')
parser.add_argument('--output-dir', default='/scratch/easmit31/factor_analysis/centroid_heatmaps')
parser.add_argument('--min-animals',type=int,   default=5)
parser.add_argument('--fdr-thresh', type=float, default=0.1)
parser.add_argument('--vmin',       type=float, default=-0.6)
parser.add_argument('--vmax',       type=float, default=0.6)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

REGION_ORDER = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']
CELL_TYPE_ORDER = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'opc', 'OPCs', 'oligodendrocytes', 'vascular-cells',
]

def load_within(within_dir, min_animals, fdr_thresh):
    dfs = []
    for fpath in glob.glob(os.path.join(within_dir, '*_centroid_summary.csv')):
        ct = os.path.basename(fpath).replace('_centroid_summary.csv', '')
        df = pd.read_csv(fpath)
        df['cell_type'] = ct
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    data = pd.concat(dfs, ignore_index=True)
    data = data[data['n_animals'] >= min_animals].copy().reset_index(drop=True)
    data['sig'] = False
    for ct, grp in data.groupby('cell_type'):
        reject, _, _, _ = multipletests(grp['p_mean'].fillna(1.0), alpha=fdr_thresh, method='fdr_bh')
        data.loc[grp.index, 'sig'] = reject.tolist()
    return data[['cell_type', 'region', 'louvain', 'r_mean', 'sig']]

def load_pop(pop_dir, min_animals, fdr_thresh):
    dfs = []
    for fpath in glob.glob(os.path.join(pop_dir, '*_population_centroid_summary.csv')):
        fname = os.path.basename(fpath).replace('_population_centroid_summary.csv', '')
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        ct, region = parts
        df = pd.read_csv(fpath)
        df['cell_type'] = ct
        df['region'] = region
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    data = pd.concat(dfs, ignore_index=True)
    data = data[data['n_animals'] >= min_animals].copy().reset_index(drop=True)
    data['sig'] = False
    for ct, grp in data.groupby('cell_type'):
        reject, _, _, _ = multipletests(grp['p_mean_dist'].fillna(1.0), alpha=fdr_thresh, method='fdr_bh')
        data.loc[grp.index, 'sig'] = reject.tolist()
    return data[['cell_type', 'region', 'louvain', 'r_mean_dist', 'sig']]

print('loading data...')
within = load_within(args.within_dir, args.min_animals, args.fdr_thresh)
pop    = load_pop(args.pop_dir, args.min_animals, args.fdr_thresh)

# ── Collapse across louvains ──────────────────────────────────────────────────
def collapse(df, r_col, sig_col):
    out = df.groupby(['cell_type', 'region']).agg(
        mean_r      = (r_col,   'mean'),
        any_sig     = (sig_col, 'any'),
        n_sig       = (sig_col, 'sum'),
        n_louvains  = (r_col,   'count'),
    ).reset_index()
    return out

w_sum = collapse(within, 'r_mean',     'sig')
p_sum = collapse(pop,    'r_mean_dist', 'sig')

present_cts = [c for c in CELL_TYPE_ORDER if c in pd.concat([w_sum['cell_type'], p_sum['cell_type']]).unique()]

def to_mat(df, val_col):
    return df.pivot(index='cell_type', columns='region', values=val_col) \
             .reindex(index=present_cts, columns=REGION_ORDER)

w_r   = to_mat(w_sum, 'mean_r')
p_r   = to_mat(p_sum, 'mean_r')
w_sig = to_mat(w_sum, 'any_sig').fillna(False)
p_sig = to_mat(p_sum, 'any_sig').fillna(False)
w_n   = to_mat(w_sum, 'n_sig').fillna(0)
p_n   = to_mat(p_sum, 'n_sig').fillna(0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(24, 8))
fig.suptitle(
    f'Centroid distance ~ age — mean r across louvains\n'
    f'Bold outline = any louvain FDR q<{args.fdr_thresh}; number = n significant louvains',
    fontsize=12, y=1.02
)

for ax, r_mat, sig_mat, n_mat, title in [
    (axes[0], w_r, w_sig, w_n, 'within-animal centroid distance ~ age'),
    (axes[1], p_r, p_sig, p_n, 'population centroid distance ~ age'),
]:
    # base heatmap — no seaborn annot, we do it manually
    sns.heatmap(
        r_mat,
        cmap='RdBu_r', center=0, vmin=args.vmin, vmax=args.vmax,
        annot=False,
        linewidths=0.4, linecolor='lightgrey',
        cbar_kws={'label': 'mean Pearson r', 'shrink': 0.5},
        ax=ax, mask=r_mat.isnull()
    )

    # annotate each cell
    for i, ct in enumerate(present_cts):
        for j, region in enumerate(REGION_ORDER):
            r = r_mat.loc[ct, region] if ct in r_mat.index and region in r_mat.columns else np.nan
            sig = sig_mat.loc[ct, region] if ct in sig_mat.index and region in sig_mat.columns else False
            n = int(n_mat.loc[ct, region]) if ct in n_mat.index and region in n_mat.columns else 0

            if pd.isna(r):
                continue

            txt_color = 'white' if abs(r) > 0.35 else 'black'
            fs = 9

            if sig:
                # bold r value + n sig louvains below
                ax.text(j + 0.5, i + 0.38, f'{r:.2f}',
                        ha='center', va='center', fontsize=fs,
                        fontweight='bold', color=txt_color)
                ax.text(j + 0.5, i + 0.68, f'n={n}',
                        ha='center', va='center', fontsize=7,
                        fontweight='bold', color=txt_color)
                # bold border
                ax.add_patch(mpatches.Rectangle(
                    (j + 0.03, i + 0.03), 0.94, 0.94,
                    fill=False, edgecolor='black', lw=2.5, zorder=5
                ))
            else:
                ax.text(j + 0.5, i + 0.5, f'{r:.2f}',
                        ha='center', va='center', fontsize=fs,
                        fontweight='normal', color=txt_color)

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel('region', fontsize=9)
    ax.set_ylabel('cell type', fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.tight_layout()
outpath = os.path.join(args.output_dir, 'centroid_summary_comparison.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {outpath}')
