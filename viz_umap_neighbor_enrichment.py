"""
viz_umap_neighbor_enrichment.py

For a given cell type and region, plots UMAP colored by:
1. Per-cell lochNESS score (within-donor neighbor enrichment)
2. Per-cell mean neighbor age deviation (|age_cell - mean_age_neighbors|)

One figure per louvain cluster, side by side.

Usage:
    python viz_umap_neighbor_enrichment.py --cell_type GABAergic-neurons --region HIP
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp

CELL_TYPE_FILES = {
    'GABAergic-neurons':      'Res1_GABAergic-neurons_subset.h5ad',
    'glutamatergic-neurons':  'Res1_glutamatergic-neurons_update.h5ad',
    'astrocytes':             'Res1_astrocytes_update.h5ad',
    'microglia':              'Res1_microglia_new.h5ad',
    'opc':                    'Res1_opc-olig_subset.h5ad',
    'oligodendrocytes':       'Res1_opc-olig_subset.h5ad',
    'vascular-cells':         'Res1_vascular-cells_subset.h5ad',
    'ependymal-cells':        'Res1_ependymal-cells_new.h5ad',
}
FILTER_MAP = {
    'opc':            ('cell_class_annotation', 'oligodendrocyte precursor cells'),
    'oligodendrocytes': ('cell_class_annotation', 'oligodendrocytes'),
}
DATA_DIR = '/data/CEM/smacklab/U01/'

parser = argparse.ArgumentParser()
parser.add_argument('--cell_type', default='GABAergic-neurons', choices=CELL_TYPE_FILES.keys())
parser.add_argument('--region',    default='HIP')
parser.add_argument('--min_cells', type=int,   default=100)
parser.add_argument('--min_age',   type=float, default=1.0)
parser.add_argument('--outdir',    default='/scratch/easmit31/factor_analysis/neighbor_enrichment/')
args = parser.parse_args()

import os
os.makedirs(args.outdir, exist_ok=True)

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

fpath = DATA_DIR + CELL_TYPE_FILES[args.cell_type]
print(f"Loading {fpath}...")
with h5py.File(fpath, 'r') as f:
    umap       = f['obsm']['X_umap'][:]
    regions    = decode_categorical(f['obs']['region'])
    louvain    = decode_categorical(f['obs']['louvain'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages       = f['obs']['age'][:]
    knn_data   = f['obsp']['distances']['data'][:]
    knn_idx    = f['obsp']['distances']['indices'][:]
    knn_indptr = f['obsp']['distances']['indptr'][:]
    n_cells    = len(knn_indptr) - 1
    if args.cell_type in FILTER_MAP:
        col, val = FILTER_MAP[args.cell_type]
        fv = decode_categorical(f['obs'][col])
        type_mask = fv == val
    else:
        type_mask = np.ones(n_cells, dtype=bool)

knn = sp.csr_matrix((knn_data, knn_idx, knn_indptr), shape=(n_cells, n_cells))
region_mask = type_mask & (regions == args.region) & (ages >= args.min_age)
louvain_clusters = sorted(np.unique(louvain[region_mask]), key=lambda x: int(x))

for cl in louvain_clusters:
    mask = region_mask & (louvain == cl)
    cell_indices = np.where(mask)[0]
    if len(cell_indices) < args.min_cells:
        continue

    # per-animal donor proportions for lochNESS
    animals_in = animal_ids[cell_indices]
    donor_props = {a: (animals_in == a).mean() for a in np.unique(animals_in)}

    lochness_scores = []
    age_dev_scores  = []

    for idx in cell_indices:
        row = knn.getrow(idx)
        neighbor_in_cluster = row.indices[mask[row.indices]]
        k = len(neighbor_in_cluster)

        if k == 0:
            lochness_scores.append(np.nan)
            age_dev_scores.append(np.nan)
            continue

        # lochNESS
        donor = animal_ids[idx]
        obs_frac = (animal_ids[neighbor_in_cluster] == donor).mean()
        exp_frac = donor_props[donor]
        lochness_scores.append(obs_frac / exp_frac if exp_frac > 0 else np.nan)

        # age deviation: |own age - mean neighbor age|
        mean_neighbor_age = ages[neighbor_in_cluster].mean()
        age_dev_scores.append(abs(ages[idx] - mean_neighbor_age))

    lochness_scores = np.array(lochness_scores)
    age_dev_scores  = np.array(age_dev_scores)
    umap_sub = umap[cell_indices]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # lochNESS
    vmax_l = np.nanpercentile(lochness_scores, 95)
    sc0 = axes[0].scatter(umap_sub[:, 0], umap_sub[:, 1],
                          c=lochness_scores, cmap='YlOrRd', s=1, alpha=0.5,
                          vmin=0, vmax=vmax_l)
    plt.colorbar(sc0, ax=axes[0], label='lochNESS')
    axes[0].set_title('Within-donor neighbor enrichment\n(lochNESS)')

    # age deviation
    vmax_a = np.nanpercentile(age_dev_scores, 95)
    sc1 = axes[1].scatter(umap_sub[:, 0], umap_sub[:, 1],
                          c=age_dev_scores, cmap='viridis_r', s=1, alpha=0.5,
                          vmin=0, vmax=vmax_a)
    plt.colorbar(sc1, ax=axes[1], label='|own age - mean neighbor age|')
    axes[1].set_title('Age-based neighbor deviation')

    # cell age
    sc2 = axes[2].scatter(umap_sub[:, 0], umap_sub[:, 1],
                          c=ages[cell_indices], cmap='plasma', s=1, alpha=0.5)
    plt.colorbar(sc2, ax=axes[2], label='Age')
    axes[2].set_title('Cell age')

    for ax in axes:
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'{args.cell_type} {args.region} louvain {cl} (n={len(cell_indices)} cells)', fontsize=11)
    plt.tight_layout()
    out = f'{args.outdir}{args.cell_type}_{args.region}_louvain{cl}_umap_neighbor_enrichment.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved louvain {cl}")

print("Done.")
