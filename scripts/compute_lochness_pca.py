"""
compute_lochness_pca.py

Computes neighborhood age enrichment scores using continuous age.
For each cell, computes mean age of k nearest neighbors in PCA space.
Permutation test asks: is the correlation between a cell's own age and
its neighbors' mean age greater than expected by chance?

Per-cell metric: neighbor_mean_age (continuous)
Per-louvain summary: Pearson r between cell age and neighbor_mean_age,
                     tested against permuted null.

Usage:
    python compute_lochness_pca.py \
        --cell-type GABAergic-neurons \
        --region HIP

Inputs:
    /scratch/easmit31/factor_analysis/pca_embeddings/{cell_type}/
        louvain{X}_{region}_pca.npy
        louvain{X}_{region}_metadata.csv

Outputs per louvain:
    /scratch/easmit31/factor_analysis/lochness_pca/{cell_type}/
        louvain{X}_{region}_lochness_scores.csv
        louvain{X}_{region}_lochness_analysis.png

Author: Elli Smith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import argparse
import os
import glob

K      = 10
N_PERMS = 100

def compute_lochness(pca_path, meta_path, output_dir):

    base = os.path.basename(pca_path).replace('_pca.npy', '')

    X    = np.load(pca_path)
    meta = pd.read_csv(meta_path)

    assert len(X) == len(meta), "PCA rows and metadata rows don't match"

    ages = meta['age'].values.astype(np.float32)

    print(f"\n{base}: {len(meta)} cells | age {ages.min():.1f}-{ages.max():.1f}y")

    # kNN in PCA space (k+1 to exclude self)
    nn = NearestNeighbors(n_neighbors=K + 1, metric='euclidean', n_jobs=2)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    indices = indices[:, 1:]  # drop self

    # per-cell: mean age of k neighbors
    neighbor_mean_age = ages[indices].mean(axis=1)

    # observed correlation: cell age vs neighbor mean age
    r_obs, p_obs = stats.pearsonr(ages, neighbor_mean_age)
    print(f"  observed r={r_obs:.4f} p={p_obs:.4f}")

    # permutation test on the correlation
    null_r = np.empty(N_PERMS, dtype=np.float32)
    for p in range(N_PERMS):
        perm_ages = np.random.permutation(ages)
        perm_neighbor_mean = perm_ages[indices].mean(axis=1)
        null_r[p], _ = stats.pearsonr(perm_ages, perm_neighbor_mean)
        if (p + 1) % 20 == 0:
            print(f"  perm {p+1}/{N_PERMS}...")

    # permutation p-value for the louvain-level correlation
    if r_obs >= 0:
        perm_pval = (null_r >= r_obs).sum() / N_PERMS
    else:
        perm_pval = (null_r <= r_obs).sum() / N_PERMS

    print(f"  perm p={perm_pval:.4f} | null r mean={null_r.mean():.4f} std={null_r.std():.4f}")

    # per-cell: z-score relative to null (per-cell permutation)
    # compute per-cell neighbor mean age under permutation
    null_cell = np.empty((N_PERMS, len(meta)), dtype=np.float32)
    for p in range(N_PERMS):
        perm_ages = np.random.permutation(ages)
        null_cell[p] = perm_ages[indices].mean(axis=1)

    null_mean = null_cell.mean(axis=0)
    null_std  = null_cell.std(axis=0)
    cell_zscore = np.where(null_std > 0,
                           (neighbor_mean_age - null_mean) / null_std,
                           0.0)

    meta['neighbor_mean_age'] = neighbor_mean_age
    meta['neighbor_age_zscore'] = cell_zscore

    print(f"  cell zscore mean={cell_zscore.mean():.3f} std={cell_zscore.std():.3f}")

    os.makedirs(output_dir, exist_ok=True)
    score_file = os.path.join(output_dir, f'{base}_lochness_scores.csv')
    meta.to_csv(score_file, index=False)

    # plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.scatter(ages, neighbor_mean_age, s=3, alpha=0.3, c=ages, cmap='viridis')
    x_line = np.linspace(ages.min(), ages.max(), 100)
    slope, intercept, _, _, _ = stats.linregress(ages, neighbor_mean_age)
    ax.plot(x_line, slope * x_line + intercept, color='red', lw=1.5)
    ax.set_xlabel('cell age (years)')
    ax.set_ylabel('mean neighbor age (years)')
    ax.set_title(f'cell age vs neighbor age\nr={r_obs:.3f} p={p_obs:.4f} perm_p={perm_pval:.4f}')

    ax = axes[0, 1]
    ax.hist(cell_zscore, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('neighbor age z-score')
    ax.set_ylabel('count')
    ax.set_title(f'per-cell z-score\n(mean={cell_zscore.mean():.3f})')

    ax = axes[0, 2]
    ax.scatter(ages, cell_zscore, s=3, alpha=0.3, c=ages, cmap='viridis')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('cell age (years)')
    ax.set_ylabel('neighbor age z-score')
    ax.set_title('cell age vs z-score')

    ax = axes[1, 0]
    ax.hist(null_r, bins=30, edgecolor='black', alpha=0.7, label='null r')
    ax.axvline(r_obs, color='red', linestyle='--', label=f'observed r={r_obs:.3f}')
    ax.set_xlabel('Pearson r')
    ax.set_ylabel('count')
    ax.set_title(f'observed vs null r\nperm_p={perm_pval:.4f}')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    animal_agg = meta.groupby(['animal_id', 'age'])['neighbor_mean_age'].mean().reset_index()
    ax.scatter(animal_agg['age'], animal_agg['neighbor_mean_age'], s=40, edgecolor='black', alpha=0.7)
    slope2, intercept2, r2, p2, _ = stats.linregress(animal_agg['age'], animal_agg['neighbor_mean_age'])
    x2 = np.linspace(animal_agg['age'].min(), animal_agg['age'].max(), 100)
    ax.plot(x2, slope2 * x2 + intercept2, color='red', lw=1.5)
    ax.set_xlabel('animal age (years)')
    ax.set_ylabel('mean neighbor age')
    ax.set_title(f'per-animal mean neighbor age vs age\nr={r2:.3f} p={p2:.3f}')

    ax = axes[1, 2]
    animal_agg2 = meta.groupby(['animal_id', 'age'])['neighbor_age_zscore'].mean().reset_index()
    ax.scatter(animal_agg2['age'], animal_agg2['neighbor_age_zscore'], s=40, edgecolor='black', alpha=0.7)
    slope3, intercept3, r3, p3, _ = stats.linregress(animal_agg2['age'], animal_agg2['neighbor_age_zscore'])
    x3 = np.linspace(animal_agg2['age'].min(), animal_agg2['age'].max(), 100)
    ax.plot(x3, slope3 * x3 + intercept3, color='red', lw=1.5)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('animal age (years)')
    ax.set_ylabel('mean neighbor age z-score')
    ax.set_title(f'per-animal z-score vs age\nr={r3:.3f} p={p3:.3f}')

    plt.suptitle(f'lochNESS continuous age: {base} | k={K}', y=1.01)
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f'{base}_lochness_analysis.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    return meta, r_obs, p_obs, perm_pval


def run(cell_type, region, embed_dir, output_base):
    in_dir    = os.path.join(embed_dir, cell_type)
    out_dir   = os.path.join(output_base, cell_type)
    pattern   = os.path.join(in_dir, f'*_{region}_pca.npy')
    pca_files = sorted(glob.glob(pattern))

    if not pca_files:
        print(f"No PCA files found for {cell_type} {region} in {in_dir}")
        return

    print(f"Found {len(pca_files)} louvains for {cell_type} {region}")

    for pca_path in pca_files:
        meta_path = pca_path.replace('_pca.npy', '_metadata.csv')
        if not os.path.exists(meta_path):
            print(f"  WARNING: missing metadata for {pca_path}, skipping")
            continue
        compute_lochness(pca_path, meta_path, out_dir)

    print(f"\nAll done. Results in {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-type',  required=True)
    parser.add_argument('--region',     required=True)
    parser.add_argument('--embed-dir',  default='/scratch/easmit31/factor_analysis/pca_embeddings')
    parser.add_argument('--output-dir', default='/scratch/easmit31/factor_analysis/lochness_pca')
    args = parser.parse_args()

    run(args.cell_type, args.region, args.embed_dir, args.output_dir)
