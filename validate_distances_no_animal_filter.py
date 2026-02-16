import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def validate_distances_no_animal_filter(dist_matrix_path, metadata_path, k=10):
    """
    Validate distance matrix WITHOUT excluding same-animal cells from neighbors
    """
    
    # Load data
    print(f"Loading distance matrix from {dist_matrix_path}...")
    dist_matrix = np.load(dist_matrix_path)
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)
    
    base_name = os.path.basename(dist_matrix_path).replace('_distance_matrix.npy', '')
    output_dir = os.path.dirname(dist_matrix_path)
    
    print(f"\nValidating: {base_name}")
    print(f"Cells: {len(metadata)}")
    print(f"Animals: {metadata['animal_id'].nunique()}")
    print(f"{'='*70}\n")
    
    # Test 1: Same-animal vs different-animal distances
    print("="*70)
    print("TEST 1: Same-animal vs Different-animal Distances")
    print("="*70)
    print("Computing pairwise distances...")
    
    same_animal_dists = []
    diff_animal_dists = []
    
    for i in range(0, len(metadata), max(1, len(metadata)//500)):
        for j in range(i+1, len(metadata)):
            if metadata.iloc[i]['animal_id'] == metadata.iloc[j]['animal_id']:
                same_animal_dists.append(dist_matrix[i, j])
            else:
                diff_animal_dists.append(dist_matrix[i, j])
        
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{len(metadata)} cells...")
    
    mean_same = np.mean(same_animal_dists)
    mean_diff = np.mean(diff_animal_dists)
    ratio = mean_diff / mean_same
    
    print("RESULTS:")
    print(f"  Same-animal pairs: {len(same_animal_dists):,}")
    print(f"  Different-animal pairs: {len(diff_animal_dists):,}")
    print(f"  Mean distance (same animal): {mean_same:.2f}")
    print(f"  Mean distance (different animal): {mean_diff:.2f}")
    print(f"  Ratio (diff/same): {ratio:.2f}x")
    print(f"  Difference: {((mean_diff - mean_same)/mean_same)*100:.1f}% more distant")
    
    # Test 2: Nearest neighbor vs random distances
    print(f"\n{'='*70}")
    print("TEST 2: Nearest Neighbor vs Random Distances")
    print("="*70)
    print("Computing nearest neighbors (INCLUDING same-animal cells)...")
    
    nn_dists = []
    for i in range(len(metadata)):
        d = dist_matrix[i, :].copy()
        d[i] = np.inf  # Exclude self only
        nn_dists.append(d.min())
    
    mean_nn = np.mean(nn_dists)
    mean_random = dist_matrix[np.triu_indices_from(dist_matrix, k=1)].mean()
    ratio_nn = mean_random / mean_nn
    
    print("RESULTS:")
    print(f"  Mean nearest neighbor distance: {mean_nn:.2f}")
    print(f"  Mean random pair distance: {mean_random:.2f}")
    print(f"  Ratio: {ratio_nn:.2f}x")
    
    if ratio_nn > 1.5:
        print(f"  ✓ PASS - Distances are meaningful")
    else:
        print(f"  ✗ WARNING - Distances may not be meaningful")
    
    # Test 3: Same-animal composition of k-NN
    print(f"\n{'='*70}")
    print(f"TEST 3: Same-animal vs Different-animal in k={k} Nearest Neighbors")
    print("="*70)
    print(f"Analyzing k={k} nearest neighbors for each cell...")
    
    same_animal_counts = []
    
    for i in range(len(metadata)):
        d = dist_matrix[i, :].copy()
        d[i] = np.inf  # Exclude self only
        
        knn_indices = np.argsort(d)[:k]
        knn_animals = metadata.iloc[knn_indices]['animal_id'].values
        cell_animal = metadata.iloc[i]['animal_id']
        
        same_count = (knn_animals == cell_animal).sum()
        same_animal_counts.append(same_count)
        
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{len(metadata)} cells...")
    
    same_animal_counts = np.array(same_animal_counts)
    mean_same_neighbors = same_animal_counts.mean()
    pct_same = (mean_same_neighbors / k) * 100
    pct_diff = 100 - pct_same
    
    print("RESULTS:")
    print(f"  Same animal neighbors:")
    print(f"    Mean: {mean_same_neighbors:.2f} / {k}")
    print(f"    Median: {np.median(same_animal_counts):.0f} / {k}")
    print(f"  Different animal neighbors:")
    print(f"    Mean: {k - mean_same_neighbors:.2f} / {k}")
    print(f"    Median: {k - np.median(same_animal_counts):.0f} / {k}")
    print(f"  {pct_same:.1f}% from same animal")
    print(f"  {pct_diff:.1f}% from different animals")
    
    print(f"\n  Distribution of same-animal neighbors:")
    for n in range(k+1):
        count = (same_animal_counts == n).sum()
        pct = count / len(same_animal_counts) * 100
        print(f"    {n:2d} same-animal neighbors: {count:5d} cells ({pct:5.1f}%)")
    
    # Test 4: Age differences
    print(f"\n{'='*70}")
    print(f"TEST 4: Age Differences in k={k} Nearest Neighbors")
    print("="*70)
    
    age_diffs_nn = []
    age_diffs_knn = []
    
    for i in range(len(metadata)):
        d = dist_matrix[i, :].copy()
        d[i] = np.inf
        
        knn_indices = np.argsort(d)[:k]
        knn_ages = metadata.iloc[knn_indices]['age'].values
        cell_age = metadata.iloc[i]['age']
        
        age_diffs = np.abs(knn_ages - cell_age)
        age_diffs_nn.append(age_diffs[0])  # Nearest neighbor
        age_diffs_knn.append(age_diffs.mean())  # Mean across k
    
    print("RESULTS:")
    print(f"  Age difference to nearest neighbor:")
    print(f"    Mean: {np.mean(age_diffs_nn):.2f} years")
    print(f"    Median: {np.median(age_diffs_nn):.2f} years")
    print(f"  Mean age difference across k={k} neighbors:")
    print(f"    Mean: {np.mean(age_diffs_knn):.2f} years")
    print(f"    Median: {np.median(age_diffs_knn):.2f} years")
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("Creating summary plots...")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Same vs Different animal distances
    ax = axes[0, 0]
    ax.hist([same_animal_dists[:10000], diff_animal_dists[:10000]], bins=50,
            label=['Same animal', 'Different animal'], alpha=0.7, edgecolor='black')
    ax.axvline(mean_same, color='blue', linestyle='--', linewidth=2, label=f'Same mean={mean_same:.0f}')
    ax.axvline(mean_diff, color='orange', linestyle='--', linewidth=2, label=f'Diff mean={mean_diff:.0f}')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
    ax.set_title(f'Test 1: Same vs Diff Animal\nRatio={ratio:.2f}x')
    ax.legend()
    
    # Plot 2: NN vs random distances
    ax = axes[0, 1]
    ax.hist([nn_dists, np.random.choice(dist_matrix[np.triu_indices_from(dist_matrix, k=1)], 
                                        size=min(10000, len(nn_dists)))],
            bins=50, label=['Nearest neighbor', 'Random pairs'], alpha=0.7, edgecolor='black')
    ax.axvline(mean_nn, color='blue', linestyle='--', linewidth=2, label=f'NN mean={mean_nn:.0f}')
    ax.axvline(mean_random, color='orange', linestyle='--', linewidth=2, label=f'Random mean={mean_random:.0f}')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
    ax.set_title(f'Test 2: NN vs Random\nRatio={ratio_nn:.2f}x')
    ax.legend()
    
    # Plot 3: Same-animal neighbor distribution
    ax = axes[1, 0]
    ax.hist(same_animal_counts, bins=range(k+2), edgecolor='black', align='left')
    ax.axvline(mean_same_neighbors, color='red', linestyle='--', linewidth=2,
              label=f'Mean={mean_same_neighbors:.1f}')
    ax.set_xlabel(f'# Same-Animal Neighbors (out of {k})')
    ax.set_ylabel('Count')
    ax.set_title(f'Test 3: Animal Clustering\n{pct_same:.1f}% same-animal neighbors')
    ax.legend()
    
    # Plot 4: Age differences
    ax = axes[1, 1]
    ax.hist([age_diffs_nn, age_diffs_knn], bins=50,
           label=['To NN', f'Mean of k={k}'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Age Difference (years)')
    ax.set_ylabel('Count')
    ax.set_title(f'Test 4: Age Differences\nNN mean={np.mean(age_diffs_nn):.2f}y')
    ax.legend()
    
    plt.suptitle(f'Validation Results (No Animal Filter): {base_name}', y=0.995)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f'{base_name}_validation_no_animal_filter.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved validation plots: {plot_file}")
    plt.close()
    
    # Save summary
    summary = {
        'base_name': base_name,
        'n_cells': len(metadata),
        'n_animals': metadata['animal_id'].nunique(),
        'age_min': metadata['age'].min(),
        'age_max': metadata['age'].max(),
        'mean_dist_same_animal': mean_same,
        'mean_dist_diff_animal': mean_diff,
        'ratio_diff_to_same': ratio,
        'mean_nn_distance': mean_nn,
        'mean_random_distance': mean_random,
        'ratio_random_to_nn': ratio_nn,
        'mean_same_animal_neighbors': mean_same_neighbors,
        'pct_same_animal_neighbors': pct_same,
        'pct_diff_animal_neighbors': pct_diff,
        'mean_age_diff_nn': np.mean(age_diffs_nn),
        'median_age_diff_nn': np.median(age_diffs_nn),
        'mean_age_diff_knn': np.mean(age_diffs_knn)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_dir, f'{base_name}_validation_summary_no_animal_filter.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary statistics: {summary_file}")
    
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validate distance matrix WITHOUT animal filter for KNN'
    )
    parser.add_argument('--dist-matrix', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--k', type=int, default=10)
    
    args = parser.parse_args()
    
    validate_distances_no_animal_filter(args.dist_matrix, args.metadata, args.k)
