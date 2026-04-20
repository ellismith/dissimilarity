"""
plot_centroid_heatmaps.py

Plots Pearson r heatmaps (louvain x region) for both:
  - Part 1: within-animal centroid distance ~ age
  - Part 2: population centroid distance ~ age

One figure per cell type, saved to output_dir.

Usage:
    python plot_centroid_heatmaps.py
    python plot_centroid_heatmaps.py --metric var
    python plot_centroid_heatmaps.py --cell-type GABAergic-neurons
"""
import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--within-dir',  default='/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100')
parser.add_argument('--pop-dir',     default='/scratch/easmit31/factor_analysis/population_centroid_outputs')
parser.add_argument('--output-dir',  default='/scratch/easmit31/factor_analysis/centroid_heatmaps')
parser.add_argument('--metric',      default='mean', choices=['mean', 'var'])
parser.add_argument('--cell-type',   default=None)
parser.add_argument('--min-animals', type=int, default=5)
parser.add_argument('--vmin',        type=float, default=-0.6)
parser.add_argument('--vmax',        type=float, default=0.6)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

REGION_ORDER = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']
P1_COL = 'r_mean' if args.metric == 'mean' else 'r_var'
P2_COL = 'r_mean_dist' if args.metric == 'mean' else 'r_var_dist'
METRIC_LABEL = 'Mean' if args.metric == 'mean' else 'Variance'

def make_heatmap(ax, pivot, title):
    cols = [r for r in REGION_ORDER if r in pivot.columns]
    pivot = pivot[cols]
    sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=args.vmin, vmax=args.vmax,
                linewidths=0.3, linecolor='lightgrey',
                cbar_kws={'label': 'Pearson r'}, ax=ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Region')
    ax.set_ylabel('Louvain')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

if args.cell_type:
    p1_files = glob.glob(os.path.join(args.within_dir, f'{args.cell_type}_centroid_summary.csv'))
else:
    p1_files = glob.glob(os.path.join(args.within_dir, '*_centroid_summary.csv'))

for p1_path in sorted(p1_files):
    cell_type = os.path.basename(p1_path).replace('_centroid_summary.csv', '')
    print(f"Processing {cell_type}...")

    df1 = pd.read_csv(p1_path)
    df1 = df1[df1['n_animals'] >= args.min_animals]
    df1['louvain'] = df1['louvain'].astype(str)
    pivot1 = df1.pivot_table(index='louvain', columns='region', values=P1_COL)
    pivot1 = pivot1.reindex(sorted(pivot1.index, key=lambda x: int(x)))

    p2_files = glob.glob(os.path.join(args.pop_dir, f'{cell_type}_*_population_centroid_summary.csv'))
    if p2_files:
        dfs = []
        for f in p2_files:
            region = os.path.basename(f).replace(f'{cell_type}_', '').replace('_population_centroid_summary.csv', '')
            df = pd.read_csv(f)
            df['region'] = region
            dfs.append(df)
        df2 = pd.concat(dfs, ignore_index=True)
        df2 = df2[df2['n_animals'] >= args.min_animals]
        df2['louvain'] = df2['louvain'].astype(str)
        pivot2 = df2.pivot_table(index='louvain', columns='region', values=P2_COL)
        pivot2 = pivot2.reindex(sorted(pivot2.index, key=lambda x: int(x)))
        has_p2 = True
    else:
        has_p2 = False
        print(f"  No Part 2 CSVs found for {cell_type}")

    ncols = 2 if has_p2 else 1
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, max(6, len(pivot1) * 0.4 + 2)))
    if ncols == 1:
        axes = [axes]

    make_heatmap(axes[0], pivot1,
                 f'{cell_type}\nPart 1: {METRIC_LABEL} Within-Animal Centroid Distance ~ Age')
    if has_p2:
        make_heatmap(axes[1], pivot2,
                     f'{cell_type}\nPart 2: {METRIC_LABEL} Population Centroid Distance ~ Age')

    plt.tight_layout()
    out = os.path.join(args.output_dir, f'{cell_type}_centroid_heatmap_{args.metric}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

print("Done.")
