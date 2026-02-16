import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import scanpy as sc
import os

def calculate_neighbor_overlap(louvain, region, h5ad_path, min_age=1.0, k=10):
    """
    Calculate how much overlap there is in the actual k neighbors chosen by each metric
    """
    
    cell_class = os.path.basename(h5ad_path).replace('Res1_', '').replace('_update.h5ad', '').replace('.h5ad', '').replace('_subset', '')
    
    print(f"\n{'='*70}")
    print(f"NEIGHBOR OVERLAP ANALYSIS")
    print(f"Cell type: {cell_class}, Louvain: {louvain}, Region: {region}")
    print(f"{'='*70}\n")
    
    # Load data
    adata = sc.read_h5ad(h5ad_path, backed='r')
    louvain = str(louvain)
    mask = (adata.obs['louvain'] == louvain) & (adata.obs['region'] == region) & (adata.obs['age'] >= min_age)
    cell_indices = np.where(mask)[0]
    
    print(f"Found {len(cell_indices)} cells")
    
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
    
    # Z-scored version
    scaler = StandardScaler()
    X_zscore = scaler.fit_transform(X_dense)
    
    print(f"Filtered to {genes_pass.sum()} genes")
    print(f"Computing neighbors for all 6 metrics...\n")
    
    # Compute neighbors for all 6 metrics
    combinations = [
        ('Raw_euclidean', X_dense, 'euclidean'),
        ('Raw_correlation', X_dense, 'correlation'),
        ('Raw_cosine', X_dense, 'cosine'),
        ('Z-scored_euclidean', X_zscore, 'euclidean'),
        ('Z-scored_correlation', X_zscore, 'correlation'),
        ('Z-scored_cosine', X_zscore, 'cosine')
    ]
    
    # Store neighbor indices for each method
    all_neighbors = {}
    
    for name, X, metric in combinations:
        print(f"  Computing {name}...")
        dist = pairwise_distances(X, metric=metric)
        
        neighbors = []
        for i in range(len(X)):
            d = dist[i, :].copy()
            d[i] = np.inf
            knn_indices = np.argsort(d)[:k]
            neighbors.append(set(knn_indices))
        
        all_neighbors[name] = neighbors
    
    # Calculate pairwise overlap
    print(f"\n{'='*70}")
    print(f"PAIRWISE OVERLAP ANALYSIS")
    print(f"{'='*70}\n")
    
    method_names = list(all_neighbors.keys())
    n_methods = len(method_names)
    
    # Overlap matrix: for each pair, calculate mean Jaccard similarity
    overlap_matrix = np.zeros((n_methods, n_methods))
    
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            if i == j:
                overlap_matrix[i, j] = 100.0  # Perfect overlap with self
            else:
                # Calculate Jaccard similarity for each cell
                jaccard_sims = []
                for cell_idx in range(len(X)):
                    neighbors1 = all_neighbors[method1][cell_idx]
                    neighbors2 = all_neighbors[method2][cell_idx]
                    
                    intersection = len(neighbors1 & neighbors2)
                    union = len(neighbors1 | neighbors2)
                    
                    jaccard = intersection / union if union > 0 else 0
                    jaccard_sims.append(jaccard)
                
                # Mean Jaccard across all cells
                overlap_matrix[i, j] = np.mean(jaccard_sims) * 100
    
    # Create dataframe for easy viewing
    overlap_df = pd.DataFrame(overlap_matrix, 
                             index=method_names, 
                             columns=method_names)
    
    print("Mean Neighbor Overlap (Jaccard Similarity %):")
    print("(100% = identical neighbors, 0% = no overlap)")
    print(overlap_df.round(1).to_string())
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")
    
    # Within raw metrics
    raw_methods = ['Raw_euclidean', 'Raw_correlation', 'Raw_cosine']
    raw_overlaps = []
    for i in range(len(raw_methods)):
        for j in range(i+1, len(raw_methods)):
            raw_overlaps.append(overlap_df.loc[raw_methods[i], raw_methods[j]])
    
    print(f"Overlap among RAW metrics:")
    print(f"  Mean: {np.mean(raw_overlaps):.1f}%")
    print(f"  Range: {np.min(raw_overlaps):.1f}% - {np.max(raw_overlaps):.1f}%")
    
    # Within z-scored metrics
    zscore_methods = ['Z-scored_euclidean', 'Z-scored_correlation', 'Z-scored_cosine']
    zscore_overlaps = []
    for i in range(len(zscore_methods)):
        for j in range(i+1, len(zscore_methods)):
            zscore_overlaps.append(overlap_df.loc[zscore_methods[i], zscore_methods[j]])
    
    print(f"\nOverlap among Z-SCORED metrics:")
    print(f"  Mean: {np.mean(zscore_overlaps):.1f}%")
    print(f"  Range: {np.min(zscore_overlaps):.1f}% - {np.max(zscore_overlaps):.1f}%")
    
    # Raw vs Z-scored
    raw_vs_zscore = []
    for raw in raw_methods:
        for zs in zscore_methods:
            raw_vs_zscore.append(overlap_df.loc[raw, zs])
    
    print(f"\nOverlap between RAW and Z-SCORED:")
    print(f"  Mean: {np.mean(raw_vs_zscore):.1f}%")
    print(f"  Range: {np.min(raw_vs_zscore):.1f}% - {np.max(raw_vs_zscore):.1f}%")
    
    # Specific comparisons
    print(f"\n{'='*70}")
    print("KEY COMPARISONS")
    print(f"{'='*70}\n")
    
    print(f"Raw Euclidean vs Raw Cosine: {overlap_df.loc['Raw_euclidean', 'Raw_cosine']:.1f}%")
    print(f"Raw Euclidean vs Raw Correlation: {overlap_df.loc['Raw_euclidean', 'Raw_correlation']:.1f}%")
    print(f"Raw Cosine vs Raw Correlation: {overlap_df.loc['Raw_cosine', 'Raw_correlation']:.1f}%")
    print(f"\nRaw Euclidean vs Z-scored Euclidean: {overlap_df.loc['Raw_euclidean', 'Z-scored_euclidean']:.1f}%")
    
    # Visualization
    print(f"\n{'='*70}")
    print("Creating visualization...")
    print(f"{'='*70}\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Heatmap
    ax = axes[0]
    sns.heatmap(overlap_df, annot=True, fmt='.1f', cmap='RdYlGn', 
               vmin=0, vmax=100, ax=ax, square=True,
               cbar_kws={'label': 'Neighbor Overlap (%)'})
    ax.set_title(f'Neighbor Overlap Matrix (Jaccard Similarity)\n{cell_class}, Louvain {louvain}, {region}',
                fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    # Summary bars
    ax = axes[1]
    
    comparisons = {
        'Raw metrics\n(Euc vs Cos vs Corr)': np.mean(raw_overlaps),
        'Z-scored metrics\n(Euc vs Cos vs Corr)': np.mean(zscore_overlaps),
        'Raw vs Z-scored\n(same metric)': np.mean([
            overlap_df.loc['Raw_euclidean', 'Z-scored_euclidean'],
            overlap_df.loc['Raw_correlation', 'Z-scored_correlation'],
            overlap_df.loc['Raw_cosine', 'Z-scored_cosine']
        ]),
        'Raw vs Z-scored\n(diff metrics)': np.mean(raw_vs_zscore)
    }
    
    x = np.arange(len(comparisons))
    bars = ax.bar(x, list(comparisons.values()), 
                  color=['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728'],
                  edgecolor='black', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(comparisons.keys()), fontsize=10)
    ax.set_ylabel('Mean Neighbor Overlap (%)', fontsize=11)
    ax.set_title('Summary of Neighbor Overlap', fontweight='bold', fontsize=12)
    ax.set_ylim([0, 100])
    ax.axhline(50, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, comparisons.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_file = f'/scratch/easmit31/dissimilarity_analysis/neighbor_overlap_{cell_class}_louvain{louvain}_{region}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_file}")
    plt.close()
    
    # Save overlap matrix
    csv_file = f'/scratch/easmit31/dissimilarity_analysis/neighbor_overlap_{cell_class}_louvain{louvain}_{region}.csv'
    overlap_df.to_csv(csv_file)
    print(f"✓ Saved overlap matrix: {csv_file}")
    
    return overlap_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--louvain', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--h5ad', type=str, required=True)
    parser.add_argument('--min-age', type=float, default=1.0)
    parser.add_argument('--k', type=int, default=10)
    
    args = parser.parse_args()
    
    calculate_neighbor_overlap(args.louvain, args.region, args.h5ad, args.min_age, args.k)
