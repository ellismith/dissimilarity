"""
pca_centroid_all_regions.py

For a given cell type, loops over all brain regions and louvain clusters,
computing per-animal centroid distances (mean and variance) and regressing
against age. Saves:
  - One scatter plot per louvain (mean and variance vs age)
  - One summary CSV per cell type with r/p for all region x louvain combos

Supports splitting combined files (e.g. opc-olig) by cell_class_annotation.

Usage:
    python pca_centroid_all_regions.py --cell_type opc
    python pca_centroid_all_regions.py --cell_type oligodendrocytes
"""

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CELL_TYPE_CONFIG = {
    'GABAergic-neurons':     {'file': 'Res1_GABAergic-neurons_subset.h5ad',        'filter_col': None, 'filter_val': None},
    'glutamatergic-neurons': {'file': 'Res1_glutamatergic-neurons_update.h5ad',     'filter_col': None, 'filter_val': None},
    'astrocytes':            {'file': 'Res1_astrocytes_update.h5ad',                'filter_col': None, 'filter_val': None},
    'microglia':             {'file': 'Res1_microglia_new.h5ad',                    'filter_col': None, 'filter_val': None},
    'opc':                   {'file': 'Res1_opc-olig_subset.h5ad',                  'filter_col': 'cell_class_annotation', 'filter_val': 'oligodendrocyte precursor cells'},
    'oligodendrocytes':      {'file': 'Res1_opc-olig_subset.h5ad',                  'filter_col': 'cell_class_annotation', 'filter_val': 'oligodendrocytes'},
    'vascular-cells':        {'file': 'Res1_vascular-cells_subset.h5ad',            'filter_col': None, 'filter_val': None},
    'ependymal-cells':       {'file': 'Res1_ependymal-cells_new.h5ad',              'filter_col': None, 'filter_val': None},
}
DATA_DIR = '/data/CEM/smacklab/U01/'

parser = argparse.ArgumentParser()
parser.add_argument('--cell_type', required=True, choices=CELL_TYPE_CONFIG.keys())
parser.add_argument('--outdir',    default='/scratch/easmit31/factor_analysis/pc_centroid_outputs/')
parser.add_argument('--n_pcs',     type=int,   default=50)
parser.add_argument('--min_cells', type=int,   default=100)
parser.add_argument('--min_age',   type=float, default=1.0)
parser.add_argument('--min_cells_per_animal_old', type=int, default=2)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

def decode_categorical(grp):
    """Decode h5py categorical (categories + codes) to numpy string array."""
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    return categories[grp['codes'][:]]

# --- Load ---
cfg   = CELL_TYPE_CONFIG[args.cell_type]
fpath = DATA_DIR + cfg['file']
print(f"Loading {fpath}...")
with h5py.File(fpath, 'r') as f:
    X_pca      = f['obsm']['X_pca'][:, :args.n_pcs]
    regions    = decode_categorical(f['obs']['region'])
    louvain    = decode_categorical(f['obs']['louvain'])
    animal_ids = decode_categorical(f['obs']['animal_id'])
    ages       = f['obs']['age'][:]
    if cfg['filter_col'] is not None:
        filter_vals = decode_categorical(f['obs'][cfg['filter_col']])
    else:
        filter_vals = None

# apply cell type filter if needed
if filter_vals is not None:
    type_mask = filter_vals == cfg['filter_val']
    print(f"Filtering to '{cfg['filter_val']}': {type_mask.sum()} / {len(type_mask)} cells")
else:
    type_mask = np.ones(len(ages), dtype=bool)

all_results = []

for region in sorted(np.unique(regions)):
    region_mask = type_mask & (regions == region) & (ages >= args.min_age)
    if region_mask.sum() == 0:
        continue

    louvain_clusters = sorted(np.unique(louvain[region_mask]), key=lambda x: int(x))
    print(f"\n{region}: {region_mask.sum()} cells, {len(louvain_clusters)} louvains")

    for cl in louvain_clusters:
        mask = region_mask & (louvain == cl)
        if mask.sum() < args.min_cells:
            continue

        # per-animal centroid distances
        rows = []
        for animal in np.unique(animal_ids[np.where(mask)[0]]):
            idxs = np.where(mask & (animal_ids == animal))[0]
            if len(idxs) < args.min_cells_per_animal_old:
                continue
            local_pca = X_pca[idxs]
            centroid  = local_pca.mean(axis=0)
            dists     = np.sqrt(((local_pca - centroid) ** 2).sum(axis=1))
            age_val   = ages[idxs[0]]
            for d in dists:
                rows.append({'animal_id': animal, 'age': age_val, 'dist': d})

        df = pd.DataFrame(rows)
        if df.empty:
            continue

        agg_mean = df.groupby(['animal_id', 'age'])['dist'].mean().reset_index().sort_values('age')
        agg_var  = df.groupby(['animal_id', 'age'])['dist'].var().reset_index().sort_values('age').dropna()

        if len(agg_mean) < 5 or len(agg_var) < 5:
            continue

        sl_m, ic_m, r_m, p_m, _ = stats.linregress(agg_mean['age'], agg_mean['dist'])
        sl_v, ic_v, r_v, p_v, _ = stats.linregress(agg_var['age'],  agg_var['dist'])

        all_results.append({
            'cell_type': args.cell_type, 'region': region, 'louvain': cl,
            'n_cells': mask.sum(), 'n_animals': len(agg_mean),
            'r_mean': round(r_m, 3), 'p_mean': round(p_m, 4),
            'r_var':  round(r_v, 3), 'p_var':  round(p_v, 4),
        })

        # scatter plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, agg, sl, ic, r, p, ylabel in [
            (axes[0], agg_mean, sl_m, ic_m, r_m, p_m, 'Mean dist to animal centroid'),
            (axes[1], agg_var,  sl_v, ic_v, r_v, p_v, 'Variance of dist to animal centroid'),
        ]:
            ax.scatter(agg['age'], agg['dist'], color='steelblue', s=40)
            x_line = np.linspace(agg['age'].min(), agg['age'].max(), 100)
            ax.plot(x_line, sl * x_line + ic, color='firebrick', lw=1.5)
            ax.set_xlabel('Age')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{args.cell_type} {region} louvain {cl}\nr={r:.2f}, p={p:.3f}, n={len(agg)} animals')

        plt.tight_layout()
        out = f'{args.outdir}{args.cell_type}_{region}_louvain{cl}_centroid_mean_var.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  louvain {cl}: r_mean={r_m:.2f} p={p_m:.4f}, r_var={r_v:.2f} p={p_v:.4f}")

# save summary CSV
df_res = pd.DataFrame(all_results)
csv_out = f'{args.outdir}{args.cell_type}_centroid_summary.csv'
df_res.to_csv(csv_out, index=False)
print(f"\nSaved summary to {csv_out}")
print("Done.")
