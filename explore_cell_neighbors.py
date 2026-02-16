#!/usr/bin/env python
"""
Interactive tool to explore k-nearest neighbors for any cell
"""
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import argparse

def explore_neighbors(cell_type, louvain, region, cell_index, distance_type, k=10, min_age=1.0):
    """
    Show k-nearest neighbors for a specific cell using specified distance metric
    
    Parameters:
    -----------
    cell_type : str (e.g., 'GABAergic-neurons', 'astrocytes')
    louvain : str or int
    region : str (e.g., 'HIP', 'dlPFC')
    cell_index : int (index of cell to explore)
    distance_type : str (one of: 'raw_euclidean', 'raw_correlation', 'raw_cosine',
                        'zscore_euclidean', 'zscore_correlation', 'zscore_cosine')
    k : int (number of neighbors to show)
    """
    
    print(f"\n{'='*70}")
    print(f"EXPLORING NEIGHBORS FOR SINGLE CELL")
    print(f"{'='*70}")
    print(f"Cell type: {cell_type}")
    print(f"Louvain: {louvain}")
    print(f"Region: {region}")
    print(f"Cell index: {cell_index}")
    print(f"Distance metric: {distance_type}")
    print(f"k neighbors: {k}")
    print(f"{'='*70}\n")
    
    # Map cell type to h5ad file
    h5ad_map = {
        'astrocytes': 'Res1_astrocytes_update.h5ad',
        'GABAergic-neurons': 'Res1_GABAergic-neurons_subset.h5ad',
        'glutamatergic-neurons': 'Res1_glutamatergic-neurons_update.h5ad',
    }
    
    if cell_type not in h5ad_map:
        print(f"ERROR: Unknown cell type '{cell_type}'")
        print(f"Available: {list(h5ad_map.keys())}")
        return
    
    h5ad_path = f'/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/{h5ad_map[cell_type]}'
    
    # Parse distance type
    parts = distance_type.lower().split('_')
    if len(parts) != 2:
        print(f"ERROR: Invalid distance_type '{distance_type}'")
        print(f"Format: <data>_<metric>  (e.g., 'raw_euclidean', 'zscore_cosine')")
        return
    
    data_type, metric = parts
    if data_type not in ['raw', 'zscore']:
        print(f"ERROR: Data type must be 'raw' or 'zscore', got '{data_type}'")
        return
    if metric not in ['euclidean', 'correlation', 'cosine']:
        print(f"ERROR: Metric must be 'euclidean', 'correlation', or 'cosine', got '{metric}'")
        return
    
    # Load data
    print("Loading data...")
    adata = sc.read_h5ad(h5ad_path, backed='r')
    
    louvain = str(louvain)
    mask = (adata.obs['louvain'] == louvain) & (adata.obs['region'] == region)
    if min_age is not None:
        mask = mask & (adata.obs['age'] >= min_age)
    
    cell_indices = np.where(mask)[0]
    
    if len(cell_indices) == 0:
        print(f"ERROR: No cells found for louvain={louvain}, region={region}")
        return
    
    print(f"Found {len(cell_indices)} cells total")
    
    if cell_index >= len(cell_indices):
        print(f"ERROR: cell_index {cell_index} out of range (max: {len(cell_indices)-1})")
        return
    
    # Load expression
    X_subset = adata.X[cell_indices, :]
    
    # Filter genes
    if sp.issparse(X_subset):
        gene_expressed = np.array((X_subset > 0).sum(axis=0)).flatten()
    else:
        gene_expressed = (X_subset > 0).sum(axis=0)
    
    pct_expressing = gene_expressed / len(cell_indices)
    genes_pass = pct_expressing >= 0.05
    X_filtered = X_subset[:, genes_pass]
    
    if sp.issparse(X_filtered):
        X_dense = X_filtered.toarray()
    else:
        X_dense = X_filtered
    
    print(f"Filtered to {genes_pass.sum()} genes")
    
    # Get metadata
    metadata = adata.obs.iloc[cell_indices].copy().reset_index(drop=True)
    
    # Prepare data based on type
    if data_type == 'zscore':
        scaler = StandardScaler()
        X = scaler.fit_transform(X_dense)
        print(f"Using Z-scored expression")
    else:
        X = X_dense
        print(f"Using raw expression")
    
    # Compute distances
    print(f"Computing {metric} distances...")
    dist_matrix = pairwise_distances(X, metric=metric)
    
    # Get distances for the query cell
    distances = dist_matrix[cell_index, :].copy()
    distances[cell_index] = np.inf  # Exclude self
    
    # Find k nearest neighbors
    knn_indices = np.argsort(distances)[:k]
    knn_distances = distances[knn_indices]
    
    # Get query cell info
    query_cell = metadata.iloc[cell_index]
    
    print(f"\n{'='*70}")
    print(f"QUERY CELL (index {cell_index}):")
    print(f"{'='*70}")
    print(f"Animal ID: {query_cell['animal_id']}")
    print(f"Age: {query_cell['age']:.2f} years")
    print(f"Louvain: {query_cell['louvain']}")
    print(f"Region: {query_cell['region']}")
    
    # Get neighbor info
    neighbors = metadata.iloc[knn_indices].copy()
    neighbors['distance'] = knn_distances
    neighbors['age_diff'] = np.abs(neighbors['age'].values - query_cell['age'])
    neighbors['same_animal'] = neighbors['animal_id'] == query_cell['animal_id']
    neighbors['rank'] = range(1, k+1)
    
    # Reorder columns
    cols_to_show = ['rank', 'animal_id', 'age', 'age_diff', 'same_animal', 'distance']
    neighbors_display = neighbors[cols_to_show].copy()
    
    print(f"\n{'='*70}")
    print(f"K={k} NEAREST NEIGHBORS:")
    print(f"{'='*70}")
    print(neighbors_display.to_string(index=False))
    
    # Summary stats
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"{'='*70}")
    print(f"Same-animal neighbors: {neighbors['same_animal'].sum()} / {k} ({neighbors['same_animal'].sum()/k*100:.1f}%)")
    print(f"Mean distance: {knn_distances.mean():.2f}")
    print(f"Mean age difference: {neighbors['age_diff'].mean():.2f} years")
    print(f"Min age difference: {neighbors['age_diff'].min():.2f} years (rank {neighbors[neighbors['age_diff'] == neighbors['age_diff'].min()]['rank'].values[0]})")
    
    # Save to CSV option
    output_file = f"/scratch/easmit31/dissimilarity_analysis/cell{cell_index}_{cell_type}_louvain{louvain}_{region}_{distance_type}_k{k}_neighbors.csv"
    neighbors_display.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    
    return neighbors_display


def interactive_mode():
    """
    Interactive prompt-based exploration
    """
    print("\n" + "="*70)
    print("INTERACTIVE NEIGHBOR EXPLORER")
    print("="*70 + "\n")
    
    # Get inputs
    cell_type = input("Cell type (GABAergic-neurons, astrocytes, glutamatergic-neurons): ").strip()
    louvain = input("Louvain cluster: ").strip()
    region = input("Region (HIP, dlPFC, ACC, etc.): ").strip()
    cell_index = int(input("Cell index (integer): ").strip())
    
    print("\nDistance type options:")
    print("  1. raw_euclidean")
    print("  2. raw_correlation")
    print("  3. raw_cosine")
    print("  4. zscore_euclidean")
    print("  5. zscore_correlation")
    print("  6. zscore_cosine")
    
    distance_type = input("Distance type: ").strip()
    k = int(input("Number of neighbors (k): ").strip())
    
    # Run analysis
    explore_neighbors(cell_type, louvain, region, cell_index, distance_type, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Explore k-nearest neighbors for a specific cell')
    parser.add_argument('--cell-type', type=str, help='Cell type')
    parser.add_argument('--louvain', type=str, help='Louvain cluster')
    parser.add_argument('--region', type=str, help='Brain region')
    parser.add_argument('--cell-index', type=int, help='Index of cell to explore')
    parser.add_argument('--distance-type', type=str, help='Distance metric (e.g., raw_euclidean)')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode with prompts')
    
    args = parser.parse_args()
    
    if args.interactive or not all([args.cell_type, args.louvain, args.region, args.cell_index, args.distance_type]):
        interactive_mode()
    else:
        explore_neighbors(args.cell_type, args.louvain, args.region, args.cell_index, args.distance_type, args.k)
