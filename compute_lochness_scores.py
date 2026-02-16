import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import argparse
import os

def compute_lochness_scores(dist_matrix_path, metadata_path, output_dir, 
                            k=10, age_threshold=10, n_permutations=100):
    """
    Compute lochNESS scores for age enrichment in cell neighborhoods
    
    Parameters:
    -----------
    dist_matrix_path : str
        Path to distance matrix (.npy file)
    metadata_path : str
        Path to cell metadata (.csv file)
    output_dir : str
        Output directory (default: same as input)
    k : int
        Number of nearest neighbors (default: 10)
    age_threshold : float
        Age threshold to define "old" vs "young" (default: 10 years)
    n_permutations : int
        Number of permutations for significance testing (default: 100)
    """
    
    if output_dir is None:
        output_dir = os.path.dirname(dist_matrix_path)
    
    base_name = os.path.basename(dist_matrix_path).replace('_distance_matrix.npy', '')
    
    # Load data
    print(f"Loading distance matrix from {dist_matrix_path}...")
    dist_matrix = np.load(dist_matrix_path)
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)
    
    print(f"\n{'='*70}")
    print(f"lochNESS Analysis: {base_name}")
    print(f"{'='*70}")
    print(f"Cells: {len(metadata)}")
    print(f"Age range: {metadata['age'].min():.2f} - {metadata['age'].max():.2f}")
    print(f"Age threshold: {age_threshold} years")
    print(f"k-nearest neighbors: {k}")
    
    # Define age groups
    metadata['age_group'] = ['old' if age >= age_threshold else 'young' 
                             for age in metadata['age']]
    
    n_old = (metadata['age_group'] == 'old').sum()
    n_young = (metadata['age_group'] == 'young').sum()
    prop_old = n_old / len(metadata)
    prop_young = n_young / len(metadata)
    
    print(f"\nAge groups:")
    print(f"  Young (<{age_threshold}y): {n_young} cells ({prop_young*100:.1f}%)")
    print(f"  Old (≥{age_threshold}y): {n_old} cells ({prop_old*100:.1f}%)")
    
    # Compute lochNESS scores
    print(f"\n{'='*70}")
    print("Computing lochNESS scores...")
    print(f"{'='*70}")
    
    lochness_scores = []
    observed_old = []
    expected_old = []
    
    for i in range(len(metadata)):
        # Get distances for this cell
        distances = dist_matrix[i, :].copy()
        cell_animal = metadata.iloc[i]['animal_id']
        cell_age_group = metadata.iloc[i]['age_group']
        
        # Exclude self and same animal
        distances[i] = np.inf
        same_animal_mask = metadata['animal_id'] == cell_animal
        distances[same_animal_mask] = np.inf
        
        # Find k nearest neighbors
        knn_indices = np.argsort(distances)[:k]
        
        # Count old vs young neighbors
        knn_age_groups = metadata.iloc[knn_indices]['age_group'].values
        n_old_neighbors = (knn_age_groups == 'old').sum()
        n_young_neighbors = (knn_age_groups == 'young').sum()
        
        # Expected number of old neighbors (based on global proportion)
        expected_n_old = k * prop_old
        
        # lochNESS score: (observed - expected) / expected
        # Avoid division by zero
        if expected_n_old > 0:
            lochness = (n_old_neighbors - expected_n_old) / expected_n_old
        else:
            lochness = 0
        
        lochness_scores.append(lochness)
        observed_old.append(n_old_neighbors)
        expected_old.append(expected_n_old)
        
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{len(metadata)} cells...")
    
    # Add to metadata
    metadata['lochness_score'] = lochness_scores
    metadata['observed_old_neighbors'] = observed_old
    metadata['expected_old_neighbors'] = expected_old
    
    print(f"\nlochNESS score distribution:")
    print(f"  Mean: {np.mean(lochness_scores):.3f}")
    print(f"  Median: {np.median(lochness_scores):.3f}")
    print(f"  Std: {np.std(lochness_scores):.3f}")
    print(f"  Min: {np.min(lochness_scores):.3f}")
    print(f"  Max: {np.max(lochness_scores):.3f}")
    
    # Classify cells
    metadata['lochness_category'] = 'neutral'
    metadata.loc[metadata['lochness_score'] > 0.5, 'lochness_category'] = 'old_enriched'
    metadata.loc[metadata['lochness_score'] < -0.5, 'lochness_category'] = 'young_enriched'
    
    n_old_enriched = (metadata['lochness_category'] == 'old_enriched').sum()
    n_young_enriched = (metadata['lochness_category'] == 'young_enriched').sum()
    n_neutral = (metadata['lochness_category'] == 'neutral').sum()
    
    print(f"\nCell classification (|lochNESS| > 0.5):")
    print(f"  Old-enriched: {n_old_enriched} ({n_old_enriched/len(metadata)*100:.1f}%)")
    print(f"  Young-enriched: {n_young_enriched} ({n_young_enriched/len(metadata)*100:.1f}%)")
    print(f"  Neutral: {n_neutral} ({n_neutral/len(metadata)*100:.1f}%)")
    
    # Statistical testing via permutation
    print(f"\n{'='*70}")
    print(f"Permutation testing ({n_permutations} permutations)...")
    print(f"{'='*70}")
    
    # Permute age labels and recompute lochNESS
    null_lochness = []
    
    for perm in range(n_permutations):
        # Shuffle ages while keeping animal structure
        metadata_perm = metadata.copy()
        metadata_perm['age'] = np.random.permutation(metadata['age'].values)
        metadata_perm['age_group'] = ['old' if age >= age_threshold else 'young' 
                                      for age in metadata_perm['age']]
        
        perm_lochness = []
        for i in range(len(metadata_perm)):
            distances = dist_matrix[i, :].copy()
            cell_animal = metadata_perm.iloc[i]['animal_id']
            
            distances[i] = np.inf
            same_animal_mask = metadata_perm['animal_id'] == cell_animal
            distances[same_animal_mask] = np.inf
            
            knn_indices = np.argsort(distances)[:k]
            knn_age_groups = metadata_perm.iloc[knn_indices]['age_group'].values
            n_old_neighbors = (knn_age_groups == 'old').sum()
            
            expected_n_old = k * prop_old
            if expected_n_old > 0:
                lochness = (n_old_neighbors - expected_n_old) / expected_n_old
            else:
                lochness = 0
            perm_lochness.append(lochness)
        
        null_lochness.extend(perm_lochness)
        
        if (perm+1) % 10 == 0:
            print(f"  Completed {perm+1}/{n_permutations} permutations...")
    
    null_lochness = np.array(null_lochness)
    
    # Calculate p-values for each cell
    p_values = []
    for score in lochness_scores:
        if score >= 0:
            p_val = (null_lochness >= score).sum() / len(null_lochness)
        else:
            p_val = (null_lochness <= score).sum() / len(null_lochness)
        p_values.append(p_val)
    
    metadata['lochness_pvalue'] = p_values
    metadata['lochness_significant'] = metadata['lochness_pvalue'] < 0.05
    
    n_significant = metadata['lochness_significant'].sum()
    print(f"\nSignificant cells (p < 0.05): {n_significant}/{len(metadata)} ({n_significant/len(metadata)*100:.1f}%)")
    
    # Save results
    output_file = os.path.join(output_dir, f"{base_name}_lochness_scores.csv")
    metadata.to_csv(output_file, index=False)
    print(f"\n✓ Saved lochNESS scores: {output_file}")
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("Creating visualizations...")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: lochNESS distribution
    ax = axes[0, 0]
    ax.hist(lochness_scores, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='Neutral')
    ax.axvline(0.5, color='orange', linestyle='--', label='Threshold')
    ax.axvline(-0.5, color='orange', linestyle='--')
    ax.set_xlabel('lochNESS Score')
    ax.set_ylabel('Count')
    ax.set_title(f'lochNESS Distribution\n(mean={np.mean(lochness_scores):.3f})')
    ax.legend()
    
    # Plot 2: lochNESS vs cell age
    ax = axes[0, 1]
    scatter = ax.scatter(metadata['age'], metadata['lochness_score'], 
                        c=metadata['lochness_score'], cmap='RdBu_r', 
                        alpha=0.6, vmin=-2, vmax=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3)
    ax.axhline(-0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Cell Age (years)')
    ax.set_ylabel('lochNESS Score')
    ax.set_title('lochNESS vs Cell Age')
    plt.colorbar(scatter, ax=ax, label='lochNESS')
    
    # Plot 3: Observed vs Expected old neighbors
    ax = axes[0, 2]
    ax.scatter(metadata['expected_old_neighbors'], metadata['observed_old_neighbors'], 
              alpha=0.5, s=20)
    max_val = max(metadata['expected_old_neighbors'].max(), metadata['observed_old_neighbors'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Expected')
    ax.set_xlabel('Expected # old neighbors')
    ax.set_ylabel('Observed # old neighbors')
    ax.set_title('Observed vs Expected')
    ax.legend()
    
    # Plot 4: Category breakdown
    ax = axes[1, 0]
    categories = metadata['lochness_category'].value_counts()
    colors = {'old_enriched': 'red', 'neutral': 'gray', 'young_enriched': 'blue'}
    ax.bar(categories.index, categories.values, 
          color=[colors.get(cat, 'gray') for cat in categories.index],
          edgecolor='black')
    ax.set_ylabel('Number of cells')
    ax.set_title('Cell Classification')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 5: Null distribution vs observed
    ax = axes[1, 1]
    ax.hist(null_lochness, bins=50, alpha=0.5, label='Null (permuted)', edgecolor='black')
    ax.hist(lochness_scores, bins=50, alpha=0.5, label='Observed', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('lochNESS Score')
    ax.set_ylabel('Count')
    ax.set_title('Observed vs Null Distribution')
    ax.legend()
    
    # Plot 6: P-value distribution
    ax = axes[1, 2]
    ax.hist(metadata['lochness_pvalue'], bins=50, edgecolor='black')
    ax.axvline(0.05, color='red', linestyle='--', label='p=0.05')
    ax.set_xlabel('P-value')
    ax.set_ylabel('Count')
    ax.set_title(f'P-value Distribution\n({n_significant} significant)')
    ax.legend()
    
    plt.suptitle(f'lochNESS Analysis: {base_name}\nAge threshold={age_threshold}y, k={k}', 
                y=0.995)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"{base_name}_lochness_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plots: {plot_file}")
    plt.close()
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Mean lochNESS: {np.mean(lochness_scores):.3f}")
    print(f"Cells with old-enriched neighborhoods: {n_old_enriched} ({n_old_enriched/len(metadata)*100:.1f}%)")
    print(f"Cells with young-enriched neighborhoods: {n_young_enriched} ({n_young_enriched/len(metadata)*100:.1f}%)")
    print(f"Statistically significant cells: {n_significant} ({n_significant/len(metadata)*100:.1f}%)")
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute lochNESS scores for age enrichment analysis'
    )
    parser.add_argument('--dist-matrix', type=str, required=True,
                        help='Path to distance matrix (.npy file)')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to cell metadata (.csv file)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as dist-matrix)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of nearest neighbors (default: 10)')
    parser.add_argument('--age-threshold', type=float, default=10,
                        help='Age threshold for old vs young (default: 10)')
    parser.add_argument('--n-permutations', type=int, default=100,
                        help='Number of permutations for significance (default: 100)')
    
    args = parser.parse_args()
    
    compute_lochness_scores(
        dist_matrix_path=args.dist_matrix,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        k=args.k,
        age_threshold=args.age_threshold,
        n_permutations=args.n_permutations
    )
