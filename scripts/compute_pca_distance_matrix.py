"""
compute_pca_distance_matrix.py

Replacement for compute_dissimilarity_matrix.py that builds pairwise
Euclidean distance matrices in PCA space (X_pca from h5ad) rather than
raw gene expression space.

This produces the same output format (distance_matrix.npy + cell_metadata.csv)
expected by compute_lochness_scores_no_animal_filter.py, so the rest of the
lochNESS pipeline is unchanged.

Advantages over raw expression:
  - Much faster: 50 PCs vs ~20,000 genes
  - Less memory: distance matrix same size, but computation is cheaper
  - Consistent with centroid distance analyses

Output files per louvain × region:
  - louvain{N}_{region}_minage{min_age}_pca_distance_matrix.npy
  - louvain{N}_{region}_minage{min_age}_pca_cell_metadata.csv

Usage:
    python compute_pca_distance_matrix.py \\
        --louvain 1 \\
        --region HIP \\
        --h5ad /data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad \\
        --output-dir /scratch/easmit31/dissimilarity_analysis/pca_distance_matrices/GABAergic-neurons \\
        --min-age 1.0 \\
        --n-pcs 50

    # Or run all louvains in a region at once:
    python compute_pca_distance_matrix.py \\
        --region HIP \\
        --h5ad /data/CEM/smacklab/U01/Res1_GABAergic-neurons_subset.h5ad \\
        --output-dir /scratch/easmit31/dissimilarity_analysis/pca_distance_matrices/GABAergic-neurons \\
        --all-louvains
"""

import argparse
import os
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist

# --- Helper functions (shared with centroid scripts) ---

def decode_categorical(grp):
    categories = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in grp['categories'][:]])
    codes = grp['codes'][:]
    return categories[codes]

def decode_col(grp):
    vals = grp[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vals])


def compute_for_louvain(louvain_id, region, X_pca, regions, louvain, animal_ids, ages, barcodes,
                         output_dir, min_age, min_cells):
    """Compute and save PCA distance matrix + metadata for one louvain × region."""

    mask = (regions == region) & (louvain == str(louvain_id)) & (ages >= min_age)
    n_cells = mask.sum()

    if n_cells < min_cells:
        print(f"  Skipping louvain {louvain_id} × {region}: only {n_cells} cells (min={min_cells})")
        return False

    idxs = np.where(mask)[0]
    pca_subset = X_pca[idxs]

    print(f"  louvain {louvain_id} × {region}: {n_cells} cells — computing distance matrix...")

    # Full pairwise Euclidean in PCA space
    dist_matrix = cdist(pca_subset, pca_subset, metric='euclidean').astype(np.float32)

    # Metadata — match format expected by lochNESS scripts
    meta = pd.DataFrame({
        'barcode':   barcodes[idxs],
        'animal_id': animal_ids[idxs],
        'age':       ages[idxs],
        'region':    regions[idxs],
        'louvain':   louvain[idxs],
    })

    # Save
    base = f"louvain{louvain_id}_{region}_minage{min_age}_pca"
    npy_path  = os.path.join(output_dir, f"{base}_distance_matrix.npy")
    csv_path  = os.path.join(output_dir, f"{base}_cell_metadata.csv")

    np.save(npy_path, dist_matrix)
    meta.to_csv(csv_path, index=False)

    print(f"    Saved: {npy_path}  ({dist_matrix.shape[0]}×{dist_matrix.shape[1]}, "
          f"{os.path.getsize(npy_path)/1e6:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compute PCA-space distance matrices for lochNESS")
    parser.add_argument('--h5ad',       required=True,  help='Path to cell-class h5ad file')
    parser.add_argument('--region',     required=True,  help='Brain region (e.g. HIP)')
    parser.add_argument('--output-dir', required=True,  help='Output directory for .npy and .csv files')
    parser.add_argument('--louvain',    type=str, default=None,
                        help='Single louvain cluster ID to process (omit to use --all-louvains)')
    parser.add_argument('--all-louvains', action='store_true',
                        help='Process all louvain clusters in the given region')
    parser.add_argument('--min-age',    type=float, default=1.0, help='Minimum animal age to include')
    parser.add_argument('--min-cells',  type=int,   default=50,  help='Minimum cells to process a louvain')
    parser.add_argument('--n-pcs',      type=int,   default=50,  help='Number of PCs to use')
    args = parser.parse_args()

    if args.louvain is None and not args.all_louvains:
        parser.error("Provide either --louvain <id> or --all-louvains")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load h5ad
    print(f"Loading {args.h5ad}...")
    with h5py.File(args.h5ad, 'r') as f:
        X_pca      = f['obsm']['X_pca'][:, :args.n_pcs]
        regions    = decode_categorical(f['obs']['region'])
        louvain    = decode_categorical(f['obs']['louvain'])
        animal_ids = decode_categorical(f['obs']['animal_id'])
        ages       = f['obs']['age'][:]
        barcodes   = decode_col(f['obs']['_index'])

    print(f"Loaded {len(ages)} cells, X_pca shape: {X_pca.shape}")

    # Determine which louvains to run
    region_mask = (regions == args.region) & (ages >= args.min_age)
    if not region_mask.any():
        print(f"ERROR: No cells found for region={args.region} with age >= {args.min_age}")
        return

    if args.all_louvains:
        louvain_ids = np.unique(louvain[region_mask])
        print(f"Found {len(louvain_ids)} louvain clusters in {args.region}")
    else:
        louvain_ids = [args.louvain]

    # Run
    n_done = 0
    for lid in louvain_ids:
        ok = compute_for_louvain(
            louvain_id=lid,
            region=args.region,
            X_pca=X_pca,
            regions=regions,
            louvain=louvain,
            animal_ids=animal_ids,
            ages=ages,
            barcodes=barcodes,
            output_dir=args.output_dir,
            min_age=args.min_age,
            min_cells=args.min_cells,
        )
        if ok:
            n_done += 1

    print(f"\nDone. Processed {n_done}/{len(louvain_ids)} louvain clusters.")


if __name__ == '__main__':
    main()
