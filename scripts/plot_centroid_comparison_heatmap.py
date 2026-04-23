#!/usr/bin/env python3
"""
plot_centroid_comparison_heatmap.py

Per cell type: louvain x region heatmap of Pearson r for within-animal (left)
and population (right) centroid distance ~ age, with asterisks marking
FDR-significant louvains within each mode.

* = sig within only
+ = sig population only
** = sig both

Reads:
  within:     pc_centroid_outputs_min100/{ct}_centroid_summary.csv
  population: population_centroid_outputs/{ct}_{region}_population_centroid_summary.csv

Output: centroid_heatmaps/comparison/{ct}_centroid_comparison.png
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--within-dir', default='/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100')
parser.add_argument('--pop-dir',    default='/scratch/easmit31/factor_analysis/population_centroid_outputs')
parser.add_argument('--output-dir', default='/scratch/easmit31/factor_analysis/centroid_heatmaps/comparison')
parser.add_argument('--min-animals',type=int,   default=5)
parser.add_argument('--fdr-thresh', type=float, default=0.05)
parser.add_argument('--vmin',       type=float, default=-0.6)
parser.add_argument('--vmax',       type=float, default=0.6)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

REGION_ORDER = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']

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
    data['sig_within'] = False
    for ct, grp in data.groupby('cell_type'):
        reject, _, _, _ = multipletests(grp['p_mean'].fillna(1.0), alpha=fdr_thresh, method='fdr_bh')
        data.loc[grp.index, 'sig_within'] = reject.tolist()
    return data[['cell_type', 'region', 'louvain', 'r_mean', 'sig_within']]

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
    data['sig_pop'] = False
    for ct, grp in data.groupby('cell_type'):
        reject, _, _, _ = multipletests(grp['p_mean_dist'].fillna(1.0), alpha=fdr_thresh, method='fdr_bh')
        data.loc[grp.index, 'sig_pop'] = reject.tolist()
    return data[['cell_type', 'region', 'louvain', 'r_mean_dist', 'sig_pop']]

print('loading data...')
within = load_within(args.within_dir, args.min_animals, args.fdr_thresh)
pop    = load_pop(args.pop_dir, args.min_animals, args.fdr_thresh)

merged = pd.merge(within, pop, on=['cell_type', 'region', 'louvain'], how='outer')
merged['sig_within'] = merged['sig_within'].fillna(False)
merged['sig_pop']    = merged['sig_pop'].fillna(False)

for ct, grp in merged.groupby('cell_type'):
    louvains = sorted(grp['louvain'].unique(), key=lambda x: int(x))
    nrows    = len(louvains)

    within_r  = grp.pivot(index='louvain', columns='region', values='r_mean') \
                   .reindex(index=louvains, columns=REGION_ORDER)
    pop_r     = grp.pivot(index='louvain', columns='region', values='r_mean_dist') \
                   .reindex(index=louvains, columns=REGION_ORDER)
    sig_w_mat = grp.pivot(index='louvain', columns='region', values='sig_within') \
                   .reindex(index=louvains, columns=REGION_ORDER).fillna(False)
    sig_p_mat = grp.pivot(index='louvain', columns='region', values='sig_pop') \
                   .reindex(index=louvains, columns=REGION_ORDER).fillna(False)

    # build annotation: r value on top line, significance marker below
    def make_annot(r_mat, sig_w, sig_p):
        annot = pd.DataFrame('', index=r_mat.index, columns=r_mat.columns)
        for idx in r_mat.index:
            for col in r_mat.columns:
                r = r_mat.loc[idx, col]
                w = sig_w.loc[idx, col]
                p = sig_p.loc[idx, col]
                if pd.isna(r):
                    annot.loc[idx, col] = ''
                else:
                    marker = '**' if (w and p) else ('*' if w else ('+' if p else ''))
                    annot.loc[idx, col] = f'{r:.2f}\n{marker}' if marker else f'{r:.2f}'
        return annot

    annot_w = make_annot(within_r, sig_w_mat, sig_p_mat)
    annot_p = make_annot(pop_r,    sig_w_mat, sig_p_mat)

    # scale figure height to number of louvains
    fig_h = max(8, nrows * 0.55 + 3)
    fig, axes = plt.subplots(1, 2, figsize=(24, fig_h))
    fig.suptitle(
        f'{ct}\n'
        f'* = sig within only   + = sig population only   ** = sig both   (FDR q<{args.fdr_thresh})',
        fontsize=12, y=1.01
    )

    for ax, mat, annot, title in [
        (axes[0], within_r, annot_w, 'within-animal centroid distance ~ age'),
        (axes[1], pop_r,    annot_p, 'population centroid distance ~ age'),
    ]:
        # compute font size based on number of louvains
        annot_fs = max(5, min(9, int(180 / max(nrows, 10))))
        sns.heatmap(
            mat,
            cmap='RdBu_r', center=0, vmin=args.vmin, vmax=args.vmax,
            annot=annot, fmt='', annot_kws={'size': annot_fs, 'weight': 'normal'},
            linewidths=0.4, linecolor='lightgrey',
            cbar_kws={'label': 'Pearson r', 'shrink': 0.5},
            ax=ax, mask=mat.isnull()
        )
        # make significant cells bold by overlaying larger text
        for i, louv in enumerate(louvains):
            for j, region in enumerate(REGION_ORDER):
                w = sig_w_mat.loc[louv, region] if region in sig_w_mat.columns else False
                p = sig_p_mat.loc[louv, region] if region in sig_p_mat.columns else False
                if w or p:
                    r_val = mat.loc[louv, region]
                    if pd.notna(r_val):
                        marker = '**' if (w and p) else ('*' if w else '+')
                        ax.text(j + 0.5, i + 0.5, f'{r_val:.2f}\n{marker}',
                                ha='center', va='center',
                                fontsize=annot_fs, fontweight='bold',
                                color='black' if abs(r_val) < 0.4 else 'white')

        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlabel('region', fontsize=9)
        ax.set_ylabel('louvain', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0,  labelsize=8)

    plt.tight_layout()
    outpath = os.path.join(args.output_dir, f'{ct}_centroid_comparison.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved: {outpath}')

print('done')
