#!/usr/bin/env python3
"""
plot_centroid_sig_scatters.py

For every louvain significant in within and/or population centroid distance ~ age,
assembles the pre-existing scatter PNGs into one figure per cell type x region.

Within PNG checked in order:
  1. pc_centroid_outputs_min100/{region}/{ct}_{region}_louvain{cl}_centroid_mean_var.png
  2. pc_centroid_outputs_min100/by_celltype/{ct}_{region}_louvain{cl}_animal_centroid_mean_var.png

Population PNG:
  population_centroid_outputs/{ct}_{region}_louvain{cl}_population_centroid_mean_var.png

Output: centroid_heatmaps/sig_scatters/{ct}_{region}_sig_louvains.png
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--within-dir',  default='/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100')
parser.add_argument('--pop-dir',     default='/scratch/easmit31/factor_analysis/population_centroid_outputs')
parser.add_argument('--output-dir',  default='/scratch/easmit31/factor_analysis/centroid_heatmaps/sig_scatters')
parser.add_argument('--min-animals', type=int,   default=5)
parser.add_argument('--fdr-thresh',  type=float, default=0.05)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

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
    return data[['cell_type', 'region', 'louvain', 'r_mean', 'p_mean', 'sig_within']]

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
    return data[['cell_type', 'region', 'louvain', 'r_mean_dist', 'p_mean_dist', 'sig_pop']]

def find_within_png(within_dir, ct, region, louv):
    p1 = os.path.join(within_dir, region, f'{ct}_{region}_louvain{louv}_centroid_mean_var.png')
    p2 = os.path.join(within_dir, 'by_celltype', f'{ct}_{region}_louvain{louv}_animal_centroid_mean_var.png')
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

print('loading summaries...')
within_sum = load_within(args.within_dir, args.min_animals, args.fdr_thresh)
pop_sum    = load_pop(args.pop_dir, args.min_animals, args.fdr_thresh)

merged = pd.merge(within_sum, pop_sum, on=['cell_type', 'region', 'louvain'], how='outer')
merged['sig_within'] = merged['sig_within'].fillna(False)
merged['sig_pop']    = merged['sig_pop'].fillna(False)
sig = merged[merged['sig_within'] | merged['sig_pop']].copy()
print(f'{len(sig)} significant louvains across {sig["cell_type"].nunique()} cell types')

for (ct, region), grp in sig.groupby(['cell_type', 'region']):
    louvains = sorted(grp['louvain'].unique(), key=lambda x: int(x))
    nrows = len(louvains)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows), squeeze=False)
    fig.suptitle(f'{ct}  x  {region}\nwithin-animal (left) | population (right)',
                 fontsize=12, y=1.01)

    for row_idx, louv in enumerate(louvains):
        srow = grp[grp['louvain'] == louv].iloc[0]

        if srow['sig_within'] and srow['sig_pop']:
            sig_tag = 'sig: BOTH'
            title_color = '#8B0000'
        elif srow['sig_within']:
            sig_tag = 'sig: within only'
            title_color = '#1f77b4'
        else:
            sig_tag = 'sig: population only'
            title_color = '#ff7f0e'

        # within
        ax_w = axes[row_idx][0]
        within_png = find_within_png(args.within_dir, ct, region, louv)
        if within_png:
            ax_w.imshow(mpimg.imread(within_png))
            r_str = f'r={srow["r_mean"]:.2f} p={srow["p_mean"]:.3f}' if pd.notna(srow.get('r_mean')) else ''
            ax_w.set_title(f'louvain {louv} | within-animal | {sig_tag}\n{r_str}',
                           fontsize=9, color=title_color)
        else:
            ax_w.text(0.5, 0.5, 'not run for this region', ha='center', va='center',
                      transform=ax_w.transAxes, color='grey', fontsize=9)
            ax_w.set_title(f'louvain {louv} | within-animal | {sig_tag}',
                           fontsize=9, color=title_color)
        ax_w.axis('off')

        # population
        ax_p = axes[row_idx][1]
        pop_png = os.path.join(args.pop_dir,
                               f'{ct}_{region}_louvain{louv}_population_centroid_mean_var.png')
        if os.path.exists(pop_png):
            ax_p.imshow(mpimg.imread(pop_png))
            r_str2 = f'r={srow["r_mean_dist"]:.2f} p={srow["p_mean_dist"]:.3f}' if pd.notna(srow.get('r_mean_dist')) else ''
            ax_p.set_title(f'louvain {louv} | population | {sig_tag}\n{r_str2}',
                           fontsize=9, color=title_color)
        else:
            ax_p.text(0.5, 0.5, 'not run for this region', ha='center', va='center',
                      transform=ax_p.transAxes, color='grey', fontsize=9)
            ax_p.set_title(f'louvain {louv} | population | {sig_tag}',
                           fontsize=9, color=title_color)
        ax_p.axis('off')

    plt.tight_layout()
    outpath = os.path.join(args.output_dir, f'{ct}_{region}_sig_louvains.png')
    plt.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'saved: {outpath}')

print('done')
