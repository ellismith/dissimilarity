"""
extract_pca_embeddings_opcolig.py

Special case extraction for opc-olig h5ad which contains two cell types.
Splits by cell_class_assign into OPCs and oligodendrocytes.
Unknowns are assigned based on their prefix.

Usage:
    python extract_pca_embeddings_opcolig.py --region HIP
"""

import h5py
import numpy as np
import pandas as pd
import argparse
import os

DATA_DIR  = '/data/CEM/smacklab/U01'
N_PCS     = 50
MIN_CELLS = 100
MIN_AGE   = 1.0

FNAME = 'Res1_opc-olig_subset.h5ad'

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

def assign_cell_type(label):
    l = label.lower()
    if 'vascular' in l:
        return None
    if 'precursor' in l or 'opc' in l:
        return 'OPCs'
    if 'oligodendrocyte' in l:
        return 'oligodendrocytes'
    return None

def extract(region, output_dir):
    fpath = os.path.join(DATA_DIR, FNAME)

    print(f"Loading {fpath}...")
    with h5py.File(fpath, 'r') as f:
        X_pca      = f['obsm']['X_pca'][:, :N_PCS]
        regions    = decode_categorical(f['obs']['region'])
        louvain    = decode_categorical(f['obs']['louvain'])
        animal_ids = decode_categorical(f['obs']['animal_id'])
        ages       = f['obs']['age'][:]
        cell_class = decode_categorical(f['obs']['cell_class_assign'])

    # assign each cell to OPCs or oligodendrocytes
    assigned = np.array([assign_cell_type(c) for c in cell_class])

    for ct in ['OPCs', 'oligodendrocytes']:
        out_dir = os.path.join(output_dir, ct)
        os.makedirs(out_dir, exist_ok=True)

        region_mask = (regions == region) & (ages >= MIN_AGE) & (assigned == ct)
        louvain_clusters = np.unique(louvain[region_mask])
        print(f"\n{ct} {region}: {region_mask.sum()} cells, {len(louvain_clusters)} louvains")

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
            pd.DataFrame({
                'animal_id': animal_ids[idxs],
                'age':       ages[idxs],
                'louvain':   cl,
                'region':    region,
            }).to_csv(meta_out, index=False)

            print(f"  louvain {cl}: {n} cells -> saved")
            saved += 1

        print(f"  saved {saved}, skipped {skipped}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region',     required=True)
    parser.add_argument('--output-dir', default='/scratch/easmit31/factor_analysis/pca_embeddings')
    args = parser.parse_args()
    extract(args.region, args.output_dir)
