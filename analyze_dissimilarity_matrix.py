import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import argparse
import os

def load_data(dist_matrix_path, metadata_path):
    """Load distance matrix and metadata"""
    print(f"Loading distance matrix from {dist_matrix_path}...")
    dist_matrix = np.load(dist_matrix_path)
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)
    
    print(f"\nDistance matrix shape: {dist_matrix.shape}")
    print(f"Metadata shape: {metadata.shape}")
    
    return dist_matrix, metadata


def plot_distance_heatmap(dist_matrix, metadata, output_dir, base_name, 
                          max_cells=500, cluster=True):
    """
    Plot heatmap of distance matrix
    """
    n_cells = dist_matrix.shape[0]
    
    # Subsample if too many cells
    if n_cells > max_cells:
        print(f"\nSubsampling {max_cells} cells from {n_cells} for visualization...")
        np.random.seed(42)
        sample_idx = np.random.choice(n_cells, max_cells, replace=False)
        sample_idx = np.sort(sample_idx)  # Keep order
        
        dist_subset = dist_matrix[np.ix_(sample_idx, sample_idx)]
        meta_subset = metadata.iloc[sample_idx].copy().reset_index(drop=True)
    else:
        dist_subset = dist_matrix
        meta_subset = metadata.copy()
    
    print(f"\nPlotting heatmap for {dist_subset.shape[0]} cells...")
    
    # Plot clustered heatmap
    if cluster:
        print("Performing hierarchical clustering...")
        
        # Ensure matrix is symmetric (fix numerical precision issues)
        dist_subset = (dist_subset + dist_subset.T) / 2
        
        # Convert to condensed distance matrix for linkage
        try:
            condensed_dist = squareform(dist_subset)
            linkage_matrix = linkage(condensed_dist, method='average')
            
            # Plot clustered heatmap
            g = sns.clustermap(dist_subset, 
                              cmap='viridis',
                              figsize=(14, 12),
                              row_linkage=linkage_matrix,
                              col_linkage=linkage_matrix,
                              xticklabels=False,
                              yticklabels=False,
                              cbar_kws={'label': 'Euclidean Distance'})
            
            g.fig.suptitle(f'Clustered Distance Matrix\n{base_name}', y=1.02)
            
            # Save
            output_file = os.path.join(output_dir, f"{base_name}_heatmap_clustered.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved clustered heatmap: {output_file}")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create clustered heatmap: {e}")
    
    # Plot sorted by animal
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sort by animal and age
    meta_subset_sorted = meta_subset.sort_values(['animal_id', 'age']).reset_index(drop=True)
    sort_idx = meta_subset_sorted.index.values  # These are now 0-based indices into meta_subset
    dist_sorted = dist_subset[np.ix_(sort_idx, sort_idx)]
    
    im = ax.imshow(dist_sorted, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Euclidean Distance')
    
    ax.set_title(f'Distance Matrix (sorted by animal ID)\n{base_name}')
    ax.set_xlabel('Cell index (sorted by animal)')
    ax.set_ylabel('Cell index (sorted by animal)')
    
    # Add lines to separate animals
    animal_boundaries = meta_subset_sorted.groupby('animal_id').size().cumsum().values[:-1]
    for boundary in animal_boundaries:
        ax.axhline(boundary, color='red', linewidth=0.5, alpha=0.3)
        ax.axvline(boundary, color='red', linewidth=0.5, alpha=0.3)
    
    output_file = os.path.join(output_dir, f"{base_name}_heatmap_by_animal.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved animal-sorted heatmap: {output_file}")
    plt.close()


def analyze_knn(dist_matrix, metadata, output_dir, base_name, k=10):
    """
    Analyze k-nearest neighbors
    """
    print(f"\n{'='*70}")
    print(f"K-NEAREST NEIGHBOR ANALYSIS (k={k})")
    print(f"{'='*70}")
    
    n_cells = dist_matrix.shape[0]
    results = []
    
    for i in range(n_cells):
        # Get distances for this cell to all others
        distances = dist_matrix[i, :]
        
        # Exclude self (distance = 0) and cells from same animal
        same_animal_mask = metadata['animal_id'] == metadata.iloc[i]['animal_id']
        distances_copy = distances.copy()
        distances_copy[same_animal_mask] = np.inf  # Exclude same animal
        
        # Find k nearest neighbors
        knn_indices = np.argsort(distances_copy)[:k]
        knn_distances = distances[knn_indices]
        
        # Get metadata for these neighbors
        knn_animals = metadata.iloc[knn_indices]['animal_id'].values
        knn_ages = metadata.iloc[knn_indices]['age'].values
        
        # Calculate age differences
        cell_age = metadata.iloc[i]['age']
        age_diffs = np.abs(knn_ages - cell_age)
        
        # Store results
        results.append({
            'cell_index': i,
            'animal_id': metadata.iloc[i]['animal_id'],
            'age': cell_age,
            'mean_knn_distance': knn_distances.mean(),
            'min_knn_distance': knn_distances.min(),
            'max_knn_distance': knn_distances.max(),
            'mean_age_diff': age_diffs.mean(),
            'min_age_diff': age_diffs.min(),
            'max_age_diff': age_diffs.max(),
            'nearest_neighbor_animal': knn_animals[0],
            'nearest_neighbor_age': knn_ages[0],
            'nearest_neighbor_distance': knn_distances[0]
        })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = os.path.join(output_dir, f"{base_name}_knn_analysis_k{k}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved KNN analysis: {output_file}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS:")
    print(f"{'='*70}")
    print(f"Mean distance to nearest neighbor: {results_df['nearest_neighbor_distance'].mean():.2f}")
    print(f"Mean age difference to nearest neighbor: {results_df['min_age_diff'].mean():.2f} years")
    print(f"\nMean of mean k={k} neighbor distances: {results_df['mean_knn_distance'].mean():.2f}")
    print(f"Mean of mean age differences (k={k}): {results_df['mean_age_diff'].mean():.2f} years")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distribution of nearest neighbor distances
    axes[0, 0].hist(results_df['nearest_neighbor_distance'], bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Distance to Nearest Neighbor')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Nearest Neighbor Distances')
    axes[0, 0].axvline(results_df['nearest_neighbor_distance'].mean(), 
                       color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Plot 2: Distribution of age differences to nearest neighbor
    axes[0, 1].hist(results_df['min_age_diff'], bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Age Difference to Nearest Neighbor (years)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Age Differences (Nearest Neighbor)')
    axes[0, 1].axvline(results_df['min_age_diff'].mean(), 
                       color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # Plot 3: Mean KNN distance vs cell age
    axes[1, 0].scatter(results_df['age'], results_df['mean_knn_distance'], alpha=0.5)
    axes[1, 0].set_xlabel('Cell Age (years)')
    axes[1, 0].set_ylabel(f'Mean Distance to {k} Nearest Neighbors')
    axes[1, 0].set_title(f'KNN Distance vs Age')
    
    # Plot 4: Mean age difference vs cell age
    axes[1, 1].scatter(results_df['age'], results_df['mean_age_diff'], alpha=0.5)
    axes[1, 1].set_xlabel('Cell Age (years)')
    axes[1, 1].set_ylabel(f'Mean Age Difference to {k} Nearest Neighbors')
    axes[1, 1].set_title(f'Age Difference vs Cell Age')
    
    plt.suptitle(f'K-Nearest Neighbor Analysis\n{base_name}', y=1.00)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"{base_name}_knn_analysis_k{k}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved KNN plots: {output_file}")
    plt.close()
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze dissimilarity matrix and perform KNN analysis'
    )
    parser.add_argument('--dist-matrix', type=str, required=True,
                        help='Path to distance matrix (.npy file)')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to cell metadata (.csv file)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for plots and results')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of nearest neighbors (default: 10)')
    parser.add_argument('--max-cells-heatmap', type=int, default=500,
                        help='Maximum cells for heatmap visualization (default: 500)')
    parser.add_argument('--skip-heatmap', action='store_true',
                        help='Skip heatmap generation')
    parser.add_argument('--skip-knn', action='store_true',
                        help='Skip KNN analysis')
    
    args = parser.parse_args()
    
    # Load data
    dist_matrix, metadata = load_data(args.dist_matrix, args.metadata)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract base name from input files
    base_name = os.path.basename(args.dist_matrix).replace('_distance_matrix.npy', '')
    
    # Generate heatmaps
    if not args.skip_heatmap:
        plot_distance_heatmap(dist_matrix, metadata, args.output_dir, base_name,
                             max_cells=args.max_cells_heatmap, cluster=True)
    
    # Perform KNN analysis
    if not args.skip_knn:
        results_df = analyze_knn(dist_matrix, metadata, args.output_dir, base_name, k=args.k)
