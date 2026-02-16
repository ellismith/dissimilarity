import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import argparse
import os

def compute_dissimilarity_matrix_zscore(louvain, region, h5ad_path, output_dir, 
                                        expr_threshold=0.05, metric='euclidean', min_age=None):
    """
    Compute pairwise dissimilarity matrix with Z-SCORED gene expression.
    
    Key difference from original: Genes are z-scored (standardized) before distance computation.
    This removes the influence of gene expression magnitude and focuses on relative patterns.
    
    Parameters:
    -----------
    louvain : str or int
        Louvain cluster number
    region : str
        Brain region
    h5ad_path : str
        Path to h5ad file
    output_dir : str
        Directory to save outputs
    expr_threshold : float
        Minimum fraction of cells expressing a gene (default: 0.05 = 5%)
    metric : str
        Distance metric for sklearn.metrics.pairwise_distances (default: 'euclidean')
    min_age : float, optional
        Minimum age to include (default: None = include all ages)
    """
    
    # Extract cell type from filename
    cell_class = os.path.basename(h5ad_path).replace('Res1_', '').replace('_update.h5ad', '').replace('.h5ad', '').replace('_subset', '').replace('_new', '')
    
    # Create organized output directory: output_dir/cell_class/
    cell_output_dir = os.path.join(output_dir, cell_class)
    os.makedirs(cell_output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path, backed='r')
    
    # Select cells
    louvain = str(louvain)  # Ensure string for comparison
    mask = (adata.obs['louvain'] == louvain) & (adata.obs['region'] == region)
    
    # Apply age filter if specified
    if min_age is not None:
        mask = mask & (adata.obs['age'] >= min_age)
    
    cell_indices = np.where(mask)[0]
    
    print(f"\n{'='*70}")
    print(f"Selected: {cell_class.replace('-', ' ').title()}, Louvain {louvain}, Region {region}")
    if min_age is not None:
        print(f"Age filter: >= {min_age} years")
    print(f"Z-SCORING ENABLED: Genes will be standardized")
    print(f"{'='*70}")
    print(f"Cells: {len(cell_indices)}")
    
    if len(cell_indices) == 0:
        print("ERROR: No cells found for this combination!")
        return
    
    # Get metadata for these cells
    cell_metadata = adata.obs.iloc[cell_indices][['animal_id', 'age']].copy()
    cell_metadata['cell_index'] = cell_indices
    cell_metadata['louvain'] = louvain
    cell_metadata['region'] = region
    cell_metadata.reset_index(drop=True, inplace=True)
    
    print(f"Animals: {cell_metadata['animal_id'].nunique()}")
    print(f"Age range: {cell_metadata['age'].min():.2f} - {cell_metadata['age'].max():.2f}")
    
    # Load expression data for these cells
    print(f"\nLoading expression data...")
    X_subset = adata.X[cell_indices, :]
    print(f"Expression matrix shape: {X_subset.shape}")
    
    # Filter genes: keep only those expressed in ≥threshold of cells
    print(f"\nFiltering genes (threshold: {expr_threshold*100}% of cells)...")
    if sp.issparse(X_subset):
        gene_expressed = np.array((X_subset > 0).sum(axis=0)).flatten()
    else:
        gene_expressed = (X_subset > 0).sum(axis=0)
    
    pct_expressing = gene_expressed / len(cell_indices)
    genes_pass = pct_expressing >= expr_threshold
    
    print(f"Genes passing filter: {genes_pass.sum()} / {len(genes_pass)}")
    
    # Filter to passing genes
    X_filtered = X_subset[:, genes_pass]
    print(f"Filtered expression matrix: {X_filtered.shape}")
    
    # Convert to dense array
    print(f"\nConverting to dense array...")
    if sp.issparse(X_filtered):
        X_dense = X_filtered.toarray()
    else:
        X_dense = X_filtered
    
    print(f"Memory size: {X_dense.nbytes / 1e9:.2f} GB")
    
    # Z-SCORE GENES (this is the key difference!)
    print(f"\n{'='*70}")
    print("Z-SCORING GENES")
    print(f"{'='*70}")
    print("Standardizing each gene: (x - mean) / std")
    print("This removes magnitude effects and focuses on expression patterns")
    
    scaler = StandardScaler()
    X_zscore = scaler.fit_transform(X_dense)
    
    print(f"\nZ-scored matrix:")
    print(f"  Mean per gene (should be ~0): {X_zscore.mean(axis=0).mean():.6f}")
    print(f"  Std per gene (should be ~1): {X_zscore.std(axis=0).mean():.6f}")
    print(f"  Min value: {X_zscore.min():.2f}")
    print(f"  Max value: {X_zscore.max():.2f}")
    
    # Compute all pairwise distances
    print(f"\n{'='*70}")
    print(f"Computing pairwise {metric} distances on Z-SCORED data...")
    print(f"{'='*70}")
    dist_matrix = pairwise_distances(X_zscore, metric=metric)
    
    print(f"Distance matrix shape: {dist_matrix.shape}")
    print(f"Distance matrix diagonal (should be ~0): {np.diag(dist_matrix)[:5]}")
    print(f"Distance range: {dist_matrix.min():.2f} - {dist_matrix.max():.2f}")
    print(f"Mean distance: {dist_matrix.mean():.2f}")
    
    # Create output filenames with "zscore" suffix
    base_name = f"louvain{louvain}_{region}"
    if min_age is not None:
        base_name += f"_minage{min_age}"
    base_name += "_zscore"  # Add zscore suffix
    
    # Save outputs
    dist_file = os.path.join(cell_output_dir, f"{base_name}_distance_matrix.npy")
    meta_file = os.path.join(cell_output_dir, f"{base_name}_cell_metadata.csv")
    
    np.save(dist_file, dist_matrix)
    cell_metadata.to_csv(meta_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Saved outputs to: {cell_output_dir}/")
    print(f"  Distance matrix: {os.path.basename(dist_file)}")
    print(f"  Cell metadata:   {os.path.basename(meta_file)}")
    print(f"{'='*70}")
    
    return dist_matrix, cell_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute pairwise dissimilarity matrix with Z-SCORED gene expression'
    )
    parser.add_argument('--louvain', type=str, required=True,
                        help='Louvain cluster number')
    parser.add_argument('--region', type=str, required=True,
                        help='Brain region (e.g., EC, ACC, HIP)')
    parser.add_argument('--h5ad', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--output-dir', type=str, 
                        default='/scratch/easmit31/dissimilarity_analysis/dissimilarity_matrices',
                        help='Output directory (default: /scratch/easmit31/dissimilarity_analysis/dissimilarity_matrices)')
    parser.add_argument('--expr-threshold', type=float, default=0.05,
                        help='Minimum fraction of cells expressing gene (default: 0.05)')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='Distance metric (default: euclidean)')
    parser.add_argument('--min-age', type=float, default=None,
                        help='Minimum age to include (default: None = all ages)')
    
    args = parser.parse_args()
    
    compute_dissimilarity_matrix_zscore(
        louvain=args.louvain,
        region=args.region,
        h5ad_path=args.h5ad,
        output_dir=args.output_dir,
        expr_threshold=args.expr_threshold,
        metric=args.metric,
        min_age=args.min_age
    )
