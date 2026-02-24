"""
plot_heatmap_neighbor_enrichment.py

Reads summary CSV from compute_neighbor_enrichment.py and plots heatmaps
of r values (lochNESS ~ age and age deviation ~ age) across all
region x louvain combinations for a given cell type.

Usage:
    python plot_heatmap_neighbor_enrichment.py --cell_type GABAergic-neurons
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--cell_type', required=True)
parser.add_argument('--indir',  default='/scratch/easmit31/factor_analysis/neighbor_enrichment/')
parser.add_argument('--outdir', default='/scratch/easmit31/factor_analysis/neighbor_enrichment/')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

csv_path = f'{args.indir}{args.cell_type}_neighbor_enrichment_summary.csv'
df = pd.read_csv(csv_path)
print(f"Loaded {csv_path}: {len(df)} rows")

for metric, r_col, p_col in [
    ('lochNESS ~ age',      'r_lochness', 'p_lochness'),
    ('Age deviation ~ age', 'r_age_dev',  'p_age_dev'),
]:
    r_pivot = df.pivot(index='louvain', columns='region', values=r_col)
    p_pivot = df.pivot(index='louvain', columns='region', values=p_col)
    n_pivot = df.pivot(index='louvain', columns='region', values='n_animals')

    r_pivot = r_pivot.reindex(sorted(r_pivot.index, key=lambda x: int(x)))
    p_pivot = p_pivot.reindex(r_pivot.index)
    n_pivot = n_pivot.reindex(r_pivot.index)

    annot = r_pivot.copy().astype(object)
    for lou in r_pivot.index:
        for reg in r_pivot.columns:
            r_val = r_pivot.loc[lou, reg]
            p_val = p_pivot.loc[lou, reg]
            n_val = n_pivot.loc[lou, reg]
            if pd.isna(r_val):
                annot.loc[lou, reg] = ''
            else:
                sig = '*' if p_val < 0.05 else ''
                annot.loc[lou, reg] = f"{r_val:.2f}{sig}\n(n={int(n_val)})"

    fig, ax = plt.subplots(figsize=(len(r_pivot.columns) * 1.4 + 2, len(r_pivot) * 0.6 + 2))
    sns.heatmap(r_pivot.astype(float), annot=annot, fmt='', center=0,
                cmap='RdBu_r', vmin=-0.6, vmax=0.6, linewidths=0.5,
                ax=ax, cbar_kws={'label': 'Pearson r'},
                mask=r_pivot.isna(), annot_kws={'size': 7})
    ax.set_title(f'{args.cell_type} | {metric}\n(* p<0.05)', fontsize=10)
    ax.set_xlabel('Region')
    ax.set_ylabel('Louvain')
    plt.tight_layout()

    safe = metric.replace(' ', '_').replace('~', '').replace('__', '_').strip('_')
    out = f'{args.outdir}{args.cell_type}_heatmap_{safe}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

print("Done.")
