"""
viz_scatter_neighbor_enrichment.py

For a given cell type and region, plots per-animal scatter:
    x = mean lochNESS score
    y = mean age deviation of neighbors (|own age - mean neighbor age|)
colored by animal age. One plot per louvain cluster.

Usage:
    python viz_scatter_neighbor_enrichment.py --cell_type GABAergic-neurons --region HIP
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp
from scipy import stats

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
    'opc':              ('cell_class_annotation', 'oligodendrocyte precursor cells'),
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

    animals_in = animal_ids[cell_indices]
    donor_props = {a: (animals_in == a).mean() for a in np.unique(animals_in)}

    rows = []
    for idx in cell_indices:
        row = knn.getrow(idx)
        neighbor_in_cluster = row.indices[mask[row.indices]]
        if len(neighbor_in_cluster) == 0:
            continue
        donor = animal_ids[idx]
        obs_frac = (animal_ids[neighbor_in_cluster] == donor).mean()
        exp_frac = donor_props[donor]
        lochness = obs_frac / exp_frac if exp_frac > 0 else np.nan
        age_dev  = abs(ages[idx] - ages[neighbor_in_cluster].mean())
        rows.append({'animal_id': donor, 'age': ages[idx],
                     'lochness': lochness, 'age_dev': age_dev})

    df = pd.DataFrame(rows)
    agg = df.groupby(['animal_id', 'age'])[['lochness', 'age_dev']].mean().reset_index()

    if len(agg) < 5:
        continue

    norm = plt.Normalize(agg['age'].min(), agg['age'].max())
    colors = cm.plasma(norm(agg['age'].values))

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(agg['lochness'], agg['age_dev'], c=agg['age'],
                    cmap='plasma', s=50, edgecolors='black', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='Age')

    # regression lines
    for x_col, y_col, color, label in [
        ('age', 'lochness', 'darkorange', 'lochNESS~age'),
        ('age', 'age_dev',  'steelblue',  'age_dev~age'),
    ]:
        sl, ic, r, p, _ = stats.linregress(agg[x_col], agg[y_col])
        print(f"  L{cl} {label}: r={r:.2f}, p={p:.3f}")

    ax.set_xlabel('Mean lochNESS (within-donor enrichment)')
    ax.set_ylabel('Mean |own age - neighbor age|')
    ax.set_title(f'{args.cell_type} {args.region} louvain {cl}\n(n={len(agg)} animals)')
    plt.tight_layout()
    out = f'{args.outdir}{args.cell_type}_{args.region}_louvain{cl}_scatter_neighbor_enrichment.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved louvain {cl}")

print("Done.")
