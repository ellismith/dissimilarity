#!/usr/bin/env python3
"""
permutation_test_centroid.py

True permutation test for centroid distance ~ age.
For each permutation, shuffles animal ages across all louvains within each
cell type x region, recomputes r, applies within-CT FDR, counts survivors.

Usage:
    python permutation_test_centroid.py --mode within
    python permutation_test_centroid.py --mode population
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--mode',       required=True, choices=['within', 'population'])
parser.add_argument('--within-dir', default='/scratch/easmit31/factor_analysis/pc_centroid_outputs_min100')
parser.add_argument('--pop-dir',    default='/scratch/easmit31/factor_analysis/population_centroid_outputs')
parser.add_argument('--out-dir',    default='/scratch/easmit31/factor_analysis/centroid_heatmaps')
parser.add_argument('--n-perms',    type=int,   default=1000)
parser.add_argument('--min-animals',type=int,   default=5)
parser.add_argument('--fdr-thresh', type=float, default=0.05)
args = parser.parse_args()

IN_DIR = args.within_dir if args.mode == 'within' else args.pop_dir
os.makedirs(args.out_dir, exist_ok=True)

# ── Load all per-animal CSVs ──────────────────────────────────────────────────
print(f'loading per-animal CSVs from {IN_DIR}...')
dfs = []
for fpath in glob.glob(os.path.join(IN_DIR, '*_per_animal.csv')):
    fname = os.path.basename(fpath).replace('_per_animal.csv', '')
    if '_louvain' not in fname:
        continue
    left, cl = fname.rsplit('_louvain', 1)
    parts = left.rsplit('_', 1)
    if len(parts) != 2:
        continue
    ct, region = parts
    df = pd.read_csv(fpath)
    df['cell_type'] = ct
    df['region']    = region
    df['louvain']   = cl
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f'loaded {len(data)} animal x louvain records across {data["cell_type"].nunique()} cell types')

counts = data.groupby(['cell_type', 'region', 'louvain'])['animal_id'].count()
valid  = counts[counts >= args.min_animals].reset_index()
valid.columns = ['cell_type', 'region', 'louvain', 'n_animals']
data  = data.merge(valid[['cell_type', 'region', 'louvain']], on=['cell_type', 'region', 'louvain'])
print(f'after min_animals filter: {data.groupby(["cell_type","region","louvain"]).ngroups} louvains')

# ── Real r values ─────────────────────────────────────────────────────────────
def compute_r(data):
    rows = []
    for (ct, region, louv), grp in data.groupby(['cell_type', 'region', 'louvain']):
        if len(grp) < args.min_animals:
            continue
        r, p = stats.pearsonr(grp['age'], grp['mean_dist'])
        rows.append({'cell_type': ct, 'region': region, 'louvain': louv, 'r': r, 'p': p})
    return pd.DataFrame(rows)

def count_sig(df):
    df = df.copy().reset_index(drop=True)
    df['sig'] = False
    for ct, grp in df.groupby('cell_type'):
        reject, _, _, _ = multipletests(grp['p'].fillna(1.0), alpha=args.fdr_thresh, method='fdr_bh')
        df.loc[grp.index, 'sig'] = reject.tolist()
    return df['sig'].sum()

print('computing real r values...')
real_df    = compute_r(data)
n_real_sig = count_sig(real_df)
print(f'real significant louvains: {n_real_sig} / {len(real_df)}')

# ── Permutation ───────────────────────────────────────────────────────────────
print(f'running {args.n_perms} permutations...')
perm_sig_counts = []

for perm in range(args.n_perms):
    perm_data = data.copy()
    for (ct, region), grp in perm_data.groupby(['cell_type', 'region']):
        animals       = grp[['animal_id', 'age']].drop_duplicates()
        shuffled_ages = animals['age'].values.copy()
        np.random.shuffle(shuffled_ages)
        age_map = dict(zip(animals['animal_id'], shuffled_ages))
        perm_data.loc[grp.index, 'age'] = grp['animal_id'].map(age_map)
    perm_df = compute_r(perm_data)
    perm_sig_counts.append(count_sig(perm_df))
    if (perm + 1) % 100 == 0:
        print(f'  {perm+1}/{args.n_perms}, mean sig so far: {np.mean(perm_sig_counts):.2f}')

perm_sig_counts = np.array(perm_sig_counts)
p_value = (perm_sig_counts >= n_real_sig).mean()

print(f'\nReal significant: {n_real_sig}')
print(f'Permutation mean: {perm_sig_counts.mean():.2f} +/- {perm_sig_counts.std():.2f}')
print(f'Permutation p-value: {p_value:.4f}')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(perm_sig_counts, bins=30, color='steelblue', alpha=0.7, label='permuted')
ax.axvline(n_real_sig, color='firebrick', lw=2, label=f'real (n={n_real_sig})')
ax.axvline(perm_sig_counts.mean(), color='steelblue', lw=1.5, ls='--',
           label=f'perm mean={perm_sig_counts.mean():.1f}')
ax.set_xlabel('n louvains significant (FDR q<0.05)')
ax.set_ylabel('count (permutations)')
ax.set_title(f'Permutation test: {args.mode} centroid distance ~ age\n'
             f'real={n_real_sig}, perm mean={perm_sig_counts.mean():.1f}, p={p_value:.4f}')
ax.legend()
plt.tight_layout()
outpath = os.path.join(args.out_dir, f'permutation_test_{args.mode}_centroid.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved: {outpath}')
