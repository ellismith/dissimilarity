#!/usr/bin/env python3
"""
plot_centroid_summary_heatmap.py

Summary heatmap (cell type x region) for centroid distance ~ age, collapsed across louvains.
Two panels: mean r (left) and % louvains significant within-CT FDR (right).

Modes:
  --mode within      reads pc_centroid_outputs_min100/{ct}_centroid_summary.csv
                     uses columns: r_mean, p_mean  (or r_var, p_var)

  --mode population  reads population_centroid_outputs/{ct}_{region}_population_centroid_summary.csv
                     uses columns: r_mean_dist, p_mean_dist  (or r_var_dist, p_var_dist)

Usage:
  python plot_centroid_summary_heatmap.py --mode within
  python plot_centroid_summary_heatmap.py --mode population
  python plot_centroid_summary_heatmap.py --mode within --metric var
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
parser.add_argument('--mode',        required=True, choices=['within', 'population'],
                    help='within = per-animal centroid; population = population centroid')
parser.add_argument('--within-dir',  default='/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100')
parser.add_argument('--pop-dir',     default='/scratch/easmit31/factor_analysis/population_centroid_outputs')
parser.add_argument('--output-dir',  default='/scratch/easmit31/factor_analysis/centroid_heatmaps')
parser.add_argument('--metric',      default='mean', choices=['mean', 'var'])
parser.add_argument('--min-animals', type=int, default=5)
parser.add_argument('--fdr-thresh',  type=float, default=0.05)
parser.add_argument('--vmin',        type=float, default=-0.6)
parser.add_argument('--vmax',        type=float, default=0.6)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

REGION_ORDER = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']

CELL_TYPE_ORDER = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'OPCs', 'oligodendrocytes', 'vascular-cells',
]

METRIC_LABEL = 'Mean' if args.metric == 'mean' else 'Variance'

# ── Column names differ between within and population CSVs ───────────────────
# within:     r_mean / p_mean  (or r_var / p_var)
# population: r_mean_dist / p_mean_dist  (or r_var_dist / p_var_dist)
if args.mode == 'within':
    R_COL = f'r_{args.metric}'
    P_COL = f'p_{args.metric}'
else:
    R_COL = f'r_{args.metric}_dist'
    P_COL = f'p_{args.metric}_dist'

# ── Load data ────────────────────────────────────────────────────────────────
all_dfs = []

if args.mode == 'within':
    # one CSV per cell type, contains all regions
    for fpath in glob.glob(os.path.join(args.within_dir, '*_centroid_summary.csv')):
        ct = os.path.basename(fpath).replace('_centroid_summary.csv', '')
        df = pd.read_csv(fpath)
        df['cell_type'] = ct
        all_dfs.append(df)
else:
    # one CSV per cell type × region
    for fpath in glob.glob(os.path.join(args.pop_dir, '*_population_centroid_summary.csv')):
        fname = os.path.basename(fpath)
        # filename pattern: {cell_type}_{REGION}_population_centroid_summary.csv
        # region is always 2-5 chars before _population_centroid_summary
        parts = fname.replace('_population_centroid_summary.csv', '').rsplit('_', 1)
        if len(parts) != 2:
            print(f'  skipping unexpected filename: {fname}')
            continue
        ct, region = parts
        df = pd.read_csv(fpath)
        df['cell_type'] = ct
        df['region'] = region
        all_dfs.append(df)

if not all_dfs:
    raise FileNotFoundError(f'No CSVs found for mode={args.mode}')

data = pd.concat(all_dfs, ignore_index=True)
data = data[data['n_animals'] >= args.min_animals].copy()

# ── FDR correction within each cell type ─────────────────────────────────────
data['sig_fdr'] = False
for ct, grp in data.groupby('cell_type'):
    pvals = grp[P_COL].fillna(1.0).values
    reject, _, _, _ = multipletests(pvals, alpha=args.fdr_thresh, method='fdr_bh')
    data.loc[grp.index, 'sig_fdr'] = reject

# ── Collapse across louvains ─────────────────────────────────────────────────
summary = (
    data.groupby(['cell_type', 'region'])
    .agg(
        mean_r         = (R_COL,    'mean'),
        n_louvains     = (R_COL,    'count'),
        n_sig_louvains = ('sig_fdr', 'sum'),
    )
    .reset_index()
)
summary['pct_sig'] = (summary['n_sig_louvains'] / summary['n_louvains'] * 100).round(1)
summary['annot_r'] = summary['mean_r'].round(2)
summary['annot_sig'] = summary['n_sig_louvains'].apply(lambda x: str(int(x)) if x > 0 else '0')

# ── Pivot ─────────────────────────────────────────────────────────────────────
present_cts = [c for c in CELL_TYPE_ORDER if c in summary['cell_type'].unique()]

mean_r_mat  = summary.pivot(index='cell_type', columns='region', values='mean_r') \
                     .reindex(index=present_cts, columns=REGION_ORDER)
pct_sig_mat = summary.pivot(index='cell_type', columns='region', values='pct_sig') \
                     .reindex(index=present_cts, columns=REGION_ORDER)
annot_r_mat  = summary.pivot(index='cell_type', columns='region', values='annot_r') \
                      .reindex(index=present_cts, columns=REGION_ORDER)
annot_sig_mat = summary.pivot(index='cell_type', columns='region', values='annot_sig') \
                       .reindex(index=present_cts, columns=REGION_ORDER)

# ── Plot ──────────────────────────────────────────────────────────────────────
mode_label = 'Within-animal' if args.mode == 'within' else 'Population'
fig, axes = plt.subplots(1, 2, figsize=(22, 7))
fig.suptitle(f'{mode_label} centroid distance ~ age', fontsize=13, y=1.01)

ax = axes[0]
sns.heatmap(
    mean_r_mat,
    cmap='RdBu_r', center=0, vmin=args.vmin, vmax=args.vmax,
    annot=annot_r_mat, fmt='', annot_kws={'size': 7},
    linewidths=0.3, linecolor='lightgrey',
    cbar_kws={'label': 'mean r', 'shrink': 0.6},
    ax=ax, mask=mean_r_mat.isnull()
)
ax.set_title(f'mean r ({mode_label.lower()} centroid distance ~ age)\ncollapsed across louvains', fontsize=9)
ax.set_xlabel('region'); ax.set_ylabel('cell_type')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)

ax = axes[1]
sns.heatmap(
    pct_sig_mat,
    cmap='OrRd', vmin=0, vmax=100,
    annot=annot_sig_mat, fmt='', annot_kws={'size': 7},
    linewidths=0.3, linecolor='lightgrey',
    cbar_kws={'label': '% louvains sig (within-CT FDR)', 'shrink': 0.6},
    ax=ax, mask=pct_sig_mat.isnull()
)
ax.set_title(f'% louvains significant\n(within cell type FDR q<{args.fdr_thresh})', fontsize=9)
ax.set_xlabel('region'); ax.set_ylabel('')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.tight_layout()
outpath = os.path.join(args.output_dir, f'{args.mode}_centroid_summary_heatmap_{args.metric}.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {outpath}')
