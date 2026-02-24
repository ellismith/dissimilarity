"""
compute_neighbor_enrichment.py

For a given cell type, loops over all brain regions and louvain clusters,
computing two per-cell neighbor enrichment metrics:

1. lochNESS: within-donor neighbor enrichment
   = (same-donor neighbors / k_within_cluster) / (donor cells / N_cluster)

2. Age deviation: |own age - mean age of within-cluster neighbors|

Aggregates both per animal (mean) and regresses against age.
Saves one summary CSV per cell type.

Usage:
    python compute_neighbor_enrichment.py --cell_type GABAergic-neurons
"""

import argparse
import os
import h5py
import numpy as np
import pandas as pd
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
parser.add_argument('--cell_type', required=True, choices=CELL_TYPE_FILES.keys())
parser.add_argument('--min_cells', type=int,   default=100)
parser.add_argument('--min_age',   type=float, default=1.0)
parser.add_argument('--outdir',    default='/scratch/easmit31/factor_analysis/neighbor_enrichment/')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

def decode_categorical(grp):
    """Decode h5py categorical (categories + codes) to numpy string array."""
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

# --- Load ---
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
all_results = []

for region in sorted(np.unique(regions)):
    region_mask = type_mask & (regions == region) & (ages >= args.min_age)
    if region_mask.sum() == 0:
        continue

    louvain_clusters = sorted(np.unique(louvain[region_mask]), key=lambda x: int(x))
    print(f"\n{region}: {region_mask.sum()} cells, {len(louvain_clusters)} louvains")

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

            # lochNESS
            obs_frac = (animal_ids[neighbor_in_cluster] == donor).mean()
            exp_frac = donor_props[donor]
            lochness = obs_frac / exp_frac if exp_frac > 0 else np.nan

            # age deviation
            age_dev = abs(ages[idx] - ages[neighbor_in_cluster].mean())

            rows.append({'animal_id': donor, 'age': ages[idx],
                         'lochness': lochness, 'age_dev': age_dev})

        df = pd.DataFrame(rows)
        agg = df.groupby(['animal_id', 'age'])[['lochness', 'age_dev']].mean().reset_index()

        if len(agg) < 5:
            continue

        _, _, r_l, p_l, _ = stats.linregress(agg['age'], agg['lochness'])
        _, _, r_a, p_a, _ = stats.linregress(agg['age'], agg['age_dev'])

        all_results.append({
            'cell_type':  args.cell_type,
            'region':     region,
            'louvain':    cl,
            'n_cells':    len(cell_indices),
            'n_animals':  len(agg),
            'r_lochness': round(r_l, 3), 'p_lochness': round(p_l, 4),
            'r_age_dev':  round(r_a, 3), 'p_age_dev':  round(p_a, 4),
        })
        print(f"  L{cl}: r_lochness={r_l:.2f} p={p_l:.4f}, r_age_dev={r_a:.2f} p={p_a:.4f}")

df_res = pd.DataFrame(all_results)
csv_out = f'{args.outdir}{args.cell_type}_neighbor_enrichment_summary.csv'
df_res.to_csv(csv_out, index=False)
print(f"\nSaved {csv_out}")
print("Done.")
