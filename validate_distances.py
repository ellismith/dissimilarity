import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def validate_dissimilarity_matrix(dist_matrix_path, metadata_path, k=10):
    """
    Validate dissimilarity matrix and analyze neighbor composition
    Saves outputs in same directory as input files
    
    Parameters:
    -----------
    dist_matrix_path : str
        Path to distance matrix (.npy file)
    metadata_path : str
        Path to cell metadata (.csv file)
    k : int
        Number of nearest neighbors to analyze
    """
    
    # Output directory is same as input directory
    output_dir = os.path.dirname(dist_matrix_path)
    base_name = os.path.basename(dist_matrix_path).replace('_distance_matrix.npy', '')
    
    # Load data
    print(f"Loading distance matrix from {dist_matrix_path}...")
    dist_matrix = np.load(dist_matrix_path)
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)
    
    print(f"\nDistance matrix shape: {dist_matrix.shape}")
    print(f"Metadata shape: {metadata.shape}")
    
    # =================================================================
    # TEST 1: Same-animal vs different-animal distances
    # =================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Same-animal vs Different-animal Distances")
    print(f"{'='*70}")
    
    same_animal_dists = []
    diff_animal_dists = []
    
    print("Computing pairwise distances...")
    for i in range(len(metadata)):
        for j in range(i+1, len(metadata)):
            if metadata.iloc[i]['animal_id'] == metadata.iloc[j]['animal_id']:
                same_animal_dists.append(dist_matrix[i, j])
            else:
                diff_animal_dists.append(dist_matrix[i, j])
        
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{len(metadata)} cells...")
    
    mean_same = np.mean(same_animal_dists)
    mean_diff = np.mean(diff_animal_dists)
    
    print(f"\nRESULTS:")
    print(f"  Same-animal pairs: {len(same_animal_dists):,}")
    print(f"  Different-animal pairs: {len(diff_animal_dists):,}")
    print(f"  Mean distance (same animal): {mean_same:.2f}")
    print(f"  Mean distance (different animal): {mean_diff:.2f}")
    print(f"  Ratio (diff/same): {mean_diff / mean_same:.2f}x")
    print(f"  Difference: {((mean_diff / mean_same - 1) * 100):.1f}% more distant")
    
    # =================================================================
    # TEST 2: Nearest neighbor vs random distances
    # =================================================================
    print(f"\n{'='*70}")
    print("TEST 2: Nearest Neighbor vs Random Distances")
    print(f"{'='*70}")
    
    # Load KNN results if available
    knn_path = dist_matrix_path.replace('_distance_matrix.npy', f'_knn_analysis_k{k}.csv')
    if os.path.exists(knn_path):
        knn_results = pd.read_csv(knn_path)
        actual_nn_dist = knn_results['nearest_neighbor_distance'].mean()
    else:
        print("KNN results not found, computing nearest neighbors...")
        nn_dists = []
        for i in range(len(metadata)):
            distances = dist_matrix[i, :].copy()
            distances[i] = np.inf
            nn_dists.append(distances.min())
        actual_nn_dist = np.mean(nn_dists)
    
    all_dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    random_dist = all_dists.mean()
    
    print(f"\nRESULTS:")
    print(f"  Mean nearest neighbor distance: {actual_nn_dist:.2f}")
    print(f"  Mean random pair distance: {random_dist:.2f}")
    print(f"  Ratio: {random_dist / actual_nn_dist:.2f}x")
    
    if random_dist > actual_nn_dist * 1.5:
        print(f"  ✓ PASS - Distances capture meaningful structure!")
    else:
        print(f"  ✗ WARNING - Distances may not be meaningful")
    
    # =================================================================
    # TEST 3: Same-animal composition of KNN
    # =================================================================
    print(f"\n{'='*70}")
    print(f"TEST 3: Same-animal vs Different-animal in k={k} Nearest Neighbors")
    print(f"{'='*70}")
    
    same_animal_counts = []
    diff_animal_counts = []
    
    print(f"Analyzing k={k} nearest neighbors for each cell...")
    for i in range(len(metadata)):
        distances = dist_matrix[i, :].copy()
        cell_animal = metadata.iloc[i]['animal_id']
        
        # Exclude self
        distances[i] = np.inf
        
        # Find k nearest neighbors
        knn_indices = np.argsort(distances)[:k]
        knn_animals = metadata.iloc[knn_indices]['animal_id'].values
        
        same_animal = (knn_animals == cell_animal).sum()
        diff_animal = k - same_animal
        
        same_animal_counts.append(same_animal)
        diff_animal_counts.append(diff_animal)
        
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{len(metadata)} cells...")
    
    same_animal_counts = np.array(same_animal_counts)
    diff_animal_counts = np.array(diff_animal_counts)
    
    print(f"\nRESULTS:")
    print(f"  Same animal neighbors:")
    print(f"    Mean: {same_animal_counts.mean():.2f} / {k}")
    print(f"    Median: {np.median(same_animal_counts):.0f} / {k}")
    print(f"  Different animal neighbors:")
    print(f"    Mean: {diff_animal_counts.mean():.2f} / {k}")
    print(f"    Median: {np.median(diff_animal_counts):.0f} / {k}")
    print(f"\n  {(same_animal_counts.mean() / k * 100):.1f}% from same animal")
    print(f"  {(diff_animal_counts.mean() / k * 100):.1f}% from different animals")
    
    print(f"\n  Distribution of same-animal neighbors:")
    for i in range(k+1):
        count = (same_animal_counts == i).sum()
        pct = count / len(same_animal_counts) * 100
        print(f"    {i:2d} same-animal neighbors: {count:4d} cells ({pct:5.1f}%)")
    
    # =================================================================
    # TEST 4: Age analysis in KNN
    # =================================================================
    if os.path.exists(knn_path):
        print(f"\n{'='*70}")
        print(f"TEST 4: Age Differences in k={k} Nearest Neighbors")
        print(f"{'='*70}")
        
        knn_results = pd.read_csv(knn_path)
        
        print(f"\nRESULTS:")
        print(f"  Mean age diff to nearest neighbor: {knn_results['min_age_diff'].mean():.2f} years")
        print(f"  Median age diff to nearest neighbor: {knn_results['min_age_diff'].median():.2f} years")
        print(f"  Mean age diff across k={k} neighbors: {knn_results['mean_age_diff'].mean():.2f} years")
        
        print(f"\n  Age range in data: {metadata['age'].min():.2f} - {metadata['age'].max():.2f} years")
        print(f"  Total age span: {metadata['age'].max() - metadata['age'].min():.2f} years")
    
    # =================================================================
    # Create summary plots
    # =================================================================
    print(f"\n{'='*70}")
    print("Creating summary plots...")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Same vs different animal distance distributions
    axes[0, 0].hist([same_animal_dists, diff_animal_dists], 
                    bins=50, label=['Same animal', 'Different animal'],
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(mean_same, color='blue', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(mean_diff, color='orange', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Distance')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Same vs Different Animal Distances')
    axes[0, 0].legend()
    
    # Plot 2: Distribution of same-animal neighbor counts
    axes[0, 1].bar(range(k+1), 
                   [(same_animal_counts == i).sum() for i in range(k+1)],
                   edgecolor='black')
    axes[0, 1].set_xlabel(f'Number of same-animal neighbors (out of {k})')
    axes[0, 1].set_ylabel('Number of cells')
    axes[0, 1].set_title(f'Distribution of Same-Animal Neighbors (k={k})')
    axes[0, 1].axvline(same_animal_counts.mean(), color='red', linestyle='--', 
                       label=f'Mean: {same_animal_counts.mean():.2f}')
    axes[0, 1].legend()
    
    # Plot 3: Age difference distribution (if available)
    if os.path.exists(knn_path):
        axes[1, 0].hist(knn_results['min_age_diff'], bins=50, edgecolor='black')
        axes[1, 0].axvline(knn_results['min_age_diff'].mean(), color='red', 
                          linestyle='--', label=f"Mean: {knn_results['min_age_diff'].mean():.2f}y")
        axes[1, 0].set_xlabel('Age difference to nearest neighbor (years)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Age Differences to Nearest Neighbor')
        axes[1, 0].legend()
        
        # Plot 4: Distance percentiles
        axes[1, 1].hist(all_dists, bins=100, edgecolor='black', alpha=0.7)
        for percentile, label in [(25, '25th'), (50, 'Median'), (75, '75th')]:
            val = np.percentile(all_dists, percentile)
            axes[1, 1].axvline(val, linestyle='--', label=f'{label}: {val:.0f}')
        axes[1, 1].set_xlabel('Distance')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Overall Distance Distribution')
        axes[1, 1].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'KNN analysis not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'KNN analysis not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.suptitle(f'Distance Matrix Validation\n{base_name}', y=0.995)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"{base_name}_validation.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved validation plots: {os.path.basename(output_file)}")
    plt.close()
    
    # =================================================================
    # Save summary statistics
    # =================================================================
    summary = {
        'base_name': base_name,
        'n_cells': len(metadata),
        'n_animals': metadata['animal_id'].nunique(),
        'age_min': metadata['age'].min(),
        'age_max': metadata['age'].max(),
        'mean_dist_same_animal': mean_same,
        'mean_dist_diff_animal': mean_diff,
        'ratio_diff_to_same': mean_diff / mean_same,
        'mean_nn_distance': actual_nn_dist,
        'mean_random_distance': random_dist,
        'ratio_random_to_nn': random_dist / actual_nn_dist,
        'k': k,
        'mean_same_animal_neighbors': same_animal_counts.mean(),
        'pct_same_animal_neighbors': (same_animal_counts.mean() / k * 100),
        'pct_diff_animal_neighbors': (diff_animal_counts.mean() / k * 100),
    }
    
    if os.path.exists(knn_path):
        summary['mean_age_diff_nn'] = knn_results['min_age_diff'].mean()
        summary['median_age_diff_nn'] = knn_results['min_age_diff'].median()
        summary['mean_age_diff_knn'] = knn_results['mean_age_diff'].mean()
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_dir, f"{base_name}_validation_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary statistics: {os.path.basename(summary_file)}")
    
    print(f"\n{'='*70}")
    print(f"✓ Validation complete - outputs saved to: {output_dir}/")
    print(f"{'='*70}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validate dissimilarity matrix and analyze neighbor composition'
    )
    parser.add_argument('--dist-matrix', type=str, required=True,
                        help='Path to distance matrix (.npy file)')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to cell metadata (.csv file)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of nearest neighbors to analyze (default: 10)')
    
    args = parser.parse_args()
    
    validate_dissimilarity_matrix(
        dist_matrix_path=args.dist_matrix,
        metadata_path=args.metadata,
        k=args.k
    )
