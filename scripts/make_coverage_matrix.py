#!/usr/bin/env python3
"""
make_coverage_matrix.py

Produces a summary table showing, per cell type × region:
  - n_cells total (from centroid summary CSVs, summed across louvains)
  - n_animals (max across louvains — animals present in any louvain)
  - n_louvains_total (all louvains present in that CT × region)
  - n_louvains_passing (louvains with n_animals >= min_animals)

Reads from pc_centroid_outputs_min100/{ct}_centroid_summary.csv
(adult-only, min100 filter already applied upstream)

Output: centroid_heatmaps/coverage_matrix.png
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

parser = argparse.ArgumentParser()
parser.add_argument('--within-dir',  default='/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100')
parser.add_argument('--output-dir',  default='/scratch/easmit31/factor_analysis/centroid_heatmaps')
parser.add_argument('--min-animals', type=int, default=5)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

REGION_ORDER   = ['ACC', 'CN', 'dlPFC', 'EC', 'HIP', 'IPP', 'lCb', 'M1', 'MB', 'mdTN', 'NAc']
CELL_TYPE_ORDER = [
    'GABAergic-neurons', 'glutamatergic-neurons', 'astrocytes', 'microglia',
    'basket-cells', 'medium-spiny-neurons', 'cerebellar-neurons', 'ependymal-cells',
    'midbrain-neurons', 'OPCs', 'oligodendrocytes', 'vascular-cells',
]

# ── Load ──────────────────────────────────────────────────────────────────────
all_dfs = []
for fpath in glob.glob(os.path.join(args.within_dir, '*_centroid_summary.csv')):
    ct = os.path.basename(fpath).replace('_centroid_summary.csv', '')
    df = pd.read_csv(fpath)
    df['cell_type'] = ct
    all_dfs.append(df)

if not all_dfs:
    raise FileNotFoundError(f'No CSVs found in {args.within_dir}')

data = pd.concat(all_dfs, ignore_index=True)

# ── Aggregate per cell type × region ─────────────────────────────────────────
rows = []
for (ct, region), grp in data.groupby(['cell_type', 'region']):
    rows.append({
        'cell_type':           ct,
        'region':              region,
        'n_cells':             grp['n_cells'].sum(),
        'n_animals':           grp['n_animals'].max(),
        'n_louvains_total':    len(grp),
        'n_louvains_passing':  (grp['n_animals'] >= args.min_animals).sum(),
    })

summary = pd.DataFrame(rows)

# ── Pivot each metric ─────────────────────────────────────────────────────────
present_cts = [c for c in CELL_TYPE_ORDER if c in summary['cell_type'].unique()]

def pivot(col):
    return summary.pivot(index='cell_type', columns='region', values=col) \
                  .reindex(index=present_cts, columns=REGION_ORDER)

mat_cells    = pivot('n_cells')
mat_animals  = pivot('n_animals')
mat_louv_tot = pivot('n_louvains_total')
mat_louv_pass= pivot('n_louvains_passing')

# ── Plot 4-panel figure ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(26, 14))
fig.suptitle('Coverage matrix — adult animals only (min100 cells/louvain)', fontsize=13)

panels = [
    (axes[0,0], mat_cells,     'n_cells total',               'YlOrRd', None,  None),
    (axes[0,1], mat_animals,   'n_animals',                   'YlGn',   0,     55),
    (axes[1,0], mat_louv_tot,  'n_louvains total',            'Blues',  0,     None),
    (axes[1,1], mat_louv_pass, f'n_louvains passing (≥{args.min_animals} animals)', 'Purples', 0, None),
]

for ax, mat, title, cmap, vmin, vmax in panels:
    # integer annotations, blank for NaN
    annot = mat.applymap(lambda x: str(int(x)) if pd.notna(x) else '')
    kwargs = dict(fmt='', annot_kws={'size': 7}, linewidths=0.3, linecolor='lightgrey',
                  cbar_kws={'shrink': 0.6}, mask=mat.isnull())
    if vmin is not None: kwargs['vmin'] = vmin
    if vmax is not None: kwargs['vmax'] = vmax
    sns.heatmap(mat, cmap=cmap, annot=annot, ax=ax, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('region'); ax.set_ylabel('cell_type')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.tight_layout()
outpath = os.path.join(args.output_dir, 'coverage_matrix.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {outpath}')
