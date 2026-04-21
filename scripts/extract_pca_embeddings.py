"""
extract_pca_embeddings.py

For a given cell type and region, extracts PCA coordinates and metadata
for each louvain cluster and saves them as intermediate files for downstream
lochNESS analysis.

Outputs per louvain (skips if fewer than min_cells total):
    louvain{X}_{region}_pca.npy       - shape (n_cells, n_pcs), float32
    louvain{X}_{region}_metadata.csv  - animal_id, age, louvain, region

Usage:
    python extract_pca_embeddings.py \
        --cell-type GABAergic-neurons \
        --region HIP
"""

import h5py
import numpy as np
import pandas as pd
import argparse
import os

H5AD_MAP = {
    'GABAergic-neurons':     'Res1_GABAergic-neurons_subset.h5ad',
    'glutamatergic-neurons': 'Res1_glutamatergic-neurons_update.h5ad',
    'astrocytes':            'Res1_astrocytes_update.h5ad',
    'microglia':             'Res1_microglia_new.h5ad',
    'basket-cells':          'Res1_basket-cells_update.h5ad',
    'medium-spiny-neurons':  'Res1_medium-spiny-neurons_subset.h5ad',
    'cerebellar-neurons':    'Res1_cerebellar-neurons_subset.h5ad',
    'ependymal-cells':       'Res1_ependymal-cells_new.h5ad',
    'midbrain-neurons':      'Res1_midbrain-neurons_update.h5ad',
    'opc-olig':              'Res1_opc-olig_subset.h5ad',
    'vascular-cells':        'Res1_vascular-cells_subset.h5ad',
}

DATA_DIR  = '/data/CEM/smacklab/U01'
N_PCS     = 50
MIN_CELLS = 100
MIN_AGE   = 1.0

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

def decode_col(grp):
    vals = grp[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vals])

def extract(cell_type, region, output_dir):
    fname = H5AD_MAP.get(cell_type)
    if fname is None:
        raise ValueError(f"Unknown cell type: {cell_type}. Available: {list(H5AD_MAP.keys())}")

    fpath = os.path.join(DATA_DIR, fname)
    out_dir = os.path.join(output_dir, cell_type)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {fpath}...")
    with h5py.File(fpath, 'r') as f:
        X_pca      = f['obsm']['X_pca'][:, :N_PCS]
        regions    = decode_categorical(f['obs']['region'])
        louvain    = decode_categorical(f['obs']['louvain'])
        animal_ids = decode_categorical(f['obs']['animal_id'])
        ages       = f['obs']['age'][:]

    region_mask = (regions == region) & (ages >= MIN_AGE)
    louvain_clusters = np.unique(louvain[region_mask])
    print(f"Region {region}: {region_mask.sum()} cells, {len(louvain_clusters)} louvain clusters")

    saved = 0
    skipped = 0
    for cl in louvain_clusters:
        mask = region_mask & (louvain == cl)
        n = mask.sum()
        if n < MIN_CELLS:
            skipped += 1
            continue

        idxs = np.where(mask)[0]

        pca_out  = os.path.join(out_dir, f'louvain{cl}_{region}_pca.npy')
        meta_out = os.path.join(out_dir, f'louvain{cl}_{region}_metadata.csv')

        np.save(pca_out, X_pca[idxs].astype(np.float32))

        meta = pd.DataFrame({
            'animal_id': animal_ids[idxs],
            'age':       ages[idxs],
            'louvain':   cl,
            'region':    region,
        })
        meta.to_csv(meta_out, index=False)

        print(f"  louvain {cl}: {n} cells, {meta['animal_id'].nunique()} animals -> saved")
        saved += 1

    print(f"\nDone. Saved {saved} louvains, skipped {skipped} (< {MIN_CELLS} cells)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-type', required=True)
    parser.add_argument('--region',    required=True)
    parser.add_argument('--output-dir', default='/scratch/easmit31/dissimilarity_analysis/pca_embeddings')
    args = parser.parse_args()
    extract(args.cell_type, args.region, args.output_dir)
